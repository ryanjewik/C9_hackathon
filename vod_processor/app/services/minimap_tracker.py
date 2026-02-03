"""
Minimap Tracker - Tracks player positions on the minimap using blob detection and tracking.
"""

import cv2
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

try:
    from filterpy.kalman import KalmanFilter
    HAS_KALMAN = True
except ImportError:
    HAS_KALMAN = False


@dataclass
class PlayerPosition:
    """Represents a player's position on the minimap."""
    timestamp_ms: float
    player_id: str
    x: float  # Normalized 0-1
    y: float  # Normalized 0-1
    team: str  # "ally" or "enemy"
    color: str  # Detected color
    confidence: float = 1.0


@dataclass
class TrackedBlob:
    """A blob being tracked across frames."""
    blob_id: str
    player_id: Optional[str] = None
    team: str = "unknown"
    color: str = "unknown"
    positions: List[Tuple[float, float, float]] = field(default_factory=list)  # (x, y, timestamp)
    last_seen_ms: float = 0
    kalman_filter: Any = None


class MinimapTracker:
    """
    Tracks player positions on the minimap.
    
    Uses:
    - Color thresholding to detect team-colored blobs
    - Kalman filtering for smooth tracking
    - Identity persistence across frames
    """
    
    # Color ranges for different blob types (HSV)
    BLOB_COLORS = {
        "green": {  # Ally (teammate/self)
            "lower": np.array([35, 100, 100]),
            "upper": np.array([85, 255, 255]),
            "team": "ally"
        },
        "blue": {  # Ally (observed through wall)
            "lower": np.array([90, 100, 100]),
            "upper": np.array([130, 255, 255]),
            "team": "ally"
        },
        "yellow": {  # Spike carrier or special
            "lower": np.array([20, 100, 100]),
            "upper": np.array([35, 255, 255]),
            "team": "ally"
        },
        "red": {  # Enemy (revealed)
            "lower1": np.array([0, 100, 100]),
            "upper1": np.array([10, 255, 255]),
            "lower2": np.array([160, 100, 100]),
            "upper2": np.array([180, 255, 255]),
            "team": "enemy"
        },
    }
    
    # Tracking parameters
    MAX_BLOBS = 12  # 10 players + spike + extra
    BLOB_MIN_AREA = 20
    BLOB_MAX_AREA = 800
    TRACK_TIMEOUT_MS = 2000  # Remove tracks not seen for this long
    MATCH_DISTANCE_THRESHOLD = 0.1  # Normalized distance for matching
    
    def __init__(self):
        self.tracked_blobs: Dict[str, TrackedBlob] = {}
        self.next_blob_id = 0
        self._frame_count = 0
    
    def process(self, timestamp_ms: float, minimap_image: np.ndarray) -> List[PlayerPosition]:
        """
        Process a minimap frame and return detected positions.
        
        Args:
            timestamp_ms: Current timestamp
            minimap_image: BGR image of the minimap ROI
            
        Returns:
            List of detected player positions
        """
        h, w = minimap_image.shape[:2]
        hsv = cv2.cvtColor(minimap_image, cv2.COLOR_BGR2HSV)
        
        # Detect all colored blobs
        detected_blobs = []
        
        for color_name, color_def in self.BLOB_COLORS.items():
            if color_name == "red":
                # Red wraps around in HSV
                mask1 = cv2.inRange(hsv, color_def["lower1"], color_def["upper1"])
                mask2 = cv2.inRange(hsv, color_def["lower2"], color_def["upper2"])
                mask = cv2.bitwise_or(mask1, mask2)
            else:
                mask = cv2.inRange(hsv, color_def["lower"], color_def["upper"])
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for c in contours:
                area = cv2.contourArea(c)
                if self.BLOB_MIN_AREA < area < self.BLOB_MAX_AREA:
                    M = cv2.moments(c)
                    if M["m00"] > 0:
                        cx = M["m10"] / M["m00"]
                        cy = M["m01"] / M["m00"]
                        
                        # Normalize coordinates
                        nx = cx / w
                        ny = cy / h
                        
                        detected_blobs.append({
                            "x": nx,
                            "y": ny,
                            "color": color_name,
                            "team": color_def["team"],
                            "area": area,
                        })
        
        # Match detected blobs to existing tracks
        positions = self._update_tracks(timestamp_ms, detected_blobs)
        
        # Clean up old tracks
        self._cleanup_tracks(timestamp_ms)
        
        self._frame_count += 1
        
        return positions
    
    def _update_tracks(
        self,
        timestamp_ms: float,
        detected_blobs: List[Dict]
    ) -> List[PlayerPosition]:
        """Update tracked blobs with new detections."""
        positions = []
        matched_blobs = set()
        matched_tracks = set()
        
        # Match detections to existing tracks
        for blob in detected_blobs:
            best_track_id = None
            best_distance = float('inf')
            
            for track_id, track in self.tracked_blobs.items():
                if track_id in matched_tracks:
                    continue
                
                # Only match same color/team
                if track.color != blob["color"]:
                    continue
                
                # Calculate distance to last position
                if track.positions:
                    last_x, last_y, _ = track.positions[-1]
                    dist = np.sqrt((blob["x"] - last_x) ** 2 + (blob["y"] - last_y) ** 2)
                    
                    if dist < best_distance and dist < self.MATCH_DISTANCE_THRESHOLD:
                        best_distance = dist
                        best_track_id = track_id
            
            if best_track_id:
                # Update existing track
                track = self.tracked_blobs[best_track_id]
                track.positions.append((blob["x"], blob["y"], timestamp_ms))
                track.last_seen_ms = timestamp_ms
                
                # Update Kalman filter if available
                if track.kalman_filter and HAS_KALMAN:
                    track.kalman_filter.update([blob["x"], blob["y"]])
                
                matched_tracks.add(best_track_id)
                matched_blobs.add(id(blob))
                
                positions.append(PlayerPosition(
                    timestamp_ms=timestamp_ms,
                    player_id=track.player_id or track.blob_id,
                    x=blob["x"],
                    y=blob["y"],
                    team=blob["team"],
                    color=blob["color"],
                    confidence=0.9
                ))
            else:
                # Create new track
                blob_id = f"blob_{self.next_blob_id}"
                self.next_blob_id += 1
                
                new_track = TrackedBlob(
                    blob_id=blob_id,
                    team=blob["team"],
                    color=blob["color"],
                    positions=[(blob["x"], blob["y"], timestamp_ms)],
                    last_seen_ms=timestamp_ms,
                )
                
                # Initialize Kalman filter
                if HAS_KALMAN:
                    new_track.kalman_filter = self._create_kalman_filter(blob["x"], blob["y"])
                
                self.tracked_blobs[blob_id] = new_track
                
                positions.append(PlayerPosition(
                    timestamp_ms=timestamp_ms,
                    player_id=blob_id,
                    x=blob["x"],
                    y=blob["y"],
                    team=blob["team"],
                    color=blob["color"],
                    confidence=0.7  # Lower confidence for new tracks
                ))
        
        return positions
    
    def _create_kalman_filter(self, initial_x: float, initial_y: float):
        """Create a Kalman filter for position tracking."""
        if not HAS_KALMAN:
            return None
        
        kf = KalmanFilter(dim_x=4, dim_z=2)
        
        # State: [x, y, vx, vy]
        kf.x = np.array([initial_x, initial_y, 0, 0])
        
        # State transition matrix
        kf.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Measurement matrix
        kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Measurement noise
        kf.R *= 0.01
        
        # Process noise
        kf.Q *= 0.001
        
        # Initial covariance
        kf.P *= 0.1
        
        return kf
    
    def _cleanup_tracks(self, current_time_ms: float):
        """Remove tracks that haven't been seen recently."""
        to_remove = []
        
        for track_id, track in self.tracked_blobs.items():
            if current_time_ms - track.last_seen_ms > self.TRACK_TIMEOUT_MS:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.tracked_blobs[track_id]
    
    def assign_player_identities(
        self,
        timestamp_ms: float,
        player_identities: Dict[str, str]
    ):
        """
        Assign player identities to tracked blobs.
        
        Called at round start when players are in known spawn positions.
        
        Args:
            timestamp_ms: Current timestamp
            player_identities: Dict mapping spawn position to player name
        """
        # This would use spawn clustering logic to match
        # tracks to known player positions at round start
        pass
    
    def get_positions_at_time(self, timestamp_ms: float) -> List[PlayerPosition]:
        """Get all tracked positions at a specific timestamp."""
        positions = []
        
        for track in self.tracked_blobs.values():
            # Find position closest to requested timestamp
            if not track.positions:
                continue
            
            best_pos = None
            best_diff = float('inf')
            
            for x, y, t in track.positions:
                diff = abs(t - timestamp_ms)
                if diff < best_diff:
                    best_diff = diff
                    best_pos = (x, y, t)
            
            if best_pos and best_diff < 1000:  # Within 1 second
                positions.append(PlayerPosition(
                    timestamp_ms=best_pos[2],
                    player_id=track.player_id or track.blob_id,
                    x=best_pos[0],
                    y=best_pos[1],
                    team=track.team,
                    color=track.color,
                ))
        
        return positions
    
    def export_tracks(self) -> List[Dict[str, Any]]:
        """Export all tracks for debugging/analysis."""
        tracks = []
        
        for track_id, track in self.tracked_blobs.items():
            tracks.append({
                "blob_id": track.blob_id,
                "player_id": track.player_id,
                "team": track.team,
                "color": track.color,
                "positions": [
                    {"x": x, "y": y, "t_ms": t}
                    for x, y, t in track.positions
                ],
                "last_seen_ms": track.last_seen_ms,
            })
        
        return tracks
