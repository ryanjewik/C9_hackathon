"""
VOD Processor - Main processing pipeline.
Extracts game events from VALORANT VODs.
"""

import os
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

import cv2
import numpy as np

from app.schemas import JobStatus
from config import (
    get_settings,
    ROI_CONFIG,
    DETECTOR_FPS,
    TEAM_COLORS,
    KILLFEED_ROW_HEIGHT_RANGE,
    KILLFEED_MAX_ROWS,
    KILL_DEDUP_WINDOW_MS,
    KILLFEED_ROW_ROIS,
    KILLFEED_NUM_ROWS,
    KILLFEED_EXTENDED_ROWS,
    OCR_NAME_CORRECTIONS,
)


@dataclass
class Event:
    """Represents a detected game event."""
    t_ms: float
    type: str
    roi: str
    payload: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0


class RateGate:
    """Rate limiter for detector execution based on video timestamp."""
    
    def __init__(self, target_fps: float):
        self.target_fps = target_fps
        self.period_ms = 1000.0 / target_fps if target_fps > 0 else 0.0
        self.next_due_ms = 0.0
    
    def due(self, t_ms: float) -> bool:
        if self.period_ms <= 0:
            return True
        if t_ms >= self.next_due_ms:
            self.next_due_ms = t_ms + self.period_ms
            return True
        return False


def roi_to_px(frame_w: int, frame_h: int, roi_norm: Tuple[float, float, float, float]) -> Tuple[int, int, int, int]:
    """Convert normalized ROI coordinates to pixel coordinates."""
    x, y, w, h = roi_norm
    px = int(x * frame_w)
    py = int(y * frame_h)
    pw = int(w * frame_w)
    ph = int(h * frame_h)
    return px, py, pw, ph


def crop(frame: np.ndarray, roi_px: Tuple[int, int, int, int]) -> np.ndarray:
    """Crop frame using pixel coordinates."""
    x, y, w, h = roi_px
    return frame[y:y+h, x:x+w]


class VODProcessor:
    """
    Main VOD processing pipeline.
    Processes video frames and extracts game events.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self._job_manager = None
        self._player_matcher = None
        self._round_winners = None  # Optional: list of team codes that won each round
    
    def set_job_manager(self, job_manager):
        """Set the job manager for status updates."""
        self._job_manager = job_manager
    
    def set_round_winners(self, round_winners: List[str]):
        """
        Set the list of round winners for accurate score computation.
        
        Args:
            round_winners: List of team codes (e.g., ["NRG", "NRG", "FNC", ...])
                          where index 0 is round 1 winner, etc.
        """
        self._round_winners = round_winners
        print(f"[VOD] Round winners set: {len(round_winners)} rounds", flush=True)
    
    def _compute_score_from_winners(self, round_number: int) -> Dict[str, int]:
        """
        Compute the score after a given round using known round winners.
        
        Returns dict with left_team and right_team scores, or None if round_winners not set.
        """
        if not self._round_winners or round_number < 1:
            return None
        
        left_team = getattr(self, '_left_team_code', None) or "left"
        right_team = getattr(self, '_right_team_code', None) or "right"
        
        # Sum up wins for each team up to and including this round
        left_score = sum(1 for i in range(min(round_number, len(self._round_winners))) 
                       if self._round_winners[i] == left_team)
        right_score = sum(1 for i in range(min(round_number, len(self._round_winners))) 
                       if self._round_winners[i] == right_team)
        
        return {left_team: left_score, right_team: right_score}
    
    def _get_display_score(self, round_number: int) -> Dict[str, Any]:
        """
        Get the display score (left/right) for a given round.
        
        IMPORTANT: Team POSITIONS stay fixed throughout the match!
        - Left team is ALWAYS on the LEFT side of the HUD  
        - Right team is ALWAYS on the RIGHT side of the HUD
        - Positions do NOT swap at halftime - only colors change
        
        Returns dict with left_score, right_score, left_team, right_team.
        """
        scores = self._compute_score_from_winners(round_number)
        if not scores:
            return None
        
        left_team = getattr(self, '_left_team_code', None) or "left"
        right_team = getattr(self, '_right_team_code', None) or "right"
        
        # Positions are ALWAYS fixed based on configured teams
        return {
            "left_score": scores.get(left_team, 0),
            "right_score": scores.get(right_team, 0),
            "left_team": left_team,
            "right_team": right_team,
        }
    
    def process_vod(
        self,
        job_id: str,
        video_path: str,
        output_dir: str,
        match_players: Optional[List[str]] = None,
        left_team: Optional[str] = None,
        right_team: Optional[str] = None,
        map_name: Optional[str] = None,
        left_player_pool: Optional[List[str]] = None,
        right_player_pool: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Process a VOD file and extract timeline data.
        
        Args:
            job_id: Unique job identifier
            video_path: Path to the VOD file
            output_dir: Directory to save output files
            match_players: Optional list of player names to filter for (legacy, bypasses OCR)
            left_team: Team code for left side of HUD (e.g., "NRG")
            right_team: Team code for right side of HUD (e.g., "FNC")
            map_name: Map name (e.g., "Abyss")
            left_player_pool: All players who have ever played for left team (for OCR validation)
            right_player_pool: All players who have ever played for right team (for OCR validation)
            
        Returns:
            Dictionary with processing results
        """
        from app.services.job_manager import JobManager
        from app.services.db_player_matcher import DatabasePlayerMatcher
        
        # Store team codes for use in team detection
        self._left_team_code = left_team
        self._right_team_code = right_team
        self._map_name = map_name
        
        # Store player pools for OCR validation
        self._left_player_pool = left_player_pool
        self._right_player_pool = right_player_pool
        
        print(f"[{job_id}] Team codes: left={left_team}, right={right_team}, map={map_name}")
        if left_player_pool:
            print(f"[{job_id}] Left player pool ({len(left_player_pool)}): {left_player_pool}")
        if right_player_pool:
            print(f"[{job_id}] Right player pool ({len(right_player_pool)}): {right_player_pool}")
        
        # Get job manager if not set
        if self._job_manager is None:
            self._job_manager = JobManager()
        
        # Initialize player matcher for fuzzy name matching
        # IMPORTANT: Set team codes BEFORE calling set_match_players so DB roster can be loaded
        self._player_matcher = DatabasePlayerMatcher()
        self._player_matcher._left_team_code = left_team
        self._player_matcher._right_team_code = right_team
        
        # Update status to processing
        self._job_manager.update_job_status(
            job_id, JobStatus.PROCESSING, "Opening video file..."
        )
        
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration_ms = (total_frames / fps) * 1000 if fps > 0 else 0
            
            print(f"[{job_id}] Video: {frame_width}x{frame_height} @ {fps:.2f}fps, {total_frames} frames")
            
            # AUTO-DETECT team tags from HUD if not provided
            print(f"[{job_id}] Checking team auto-detect: left_team={left_team!r}, right_team={right_team!r}")
            if not left_team or not right_team:
                print(f"[{job_id}] Starting team tag auto-detection...")
                self._job_manager.update_job_status(
                    job_id, JobStatus.PROCESSING, "Detecting team tags from HUD..."
                )
                try:
                    detected_left, detected_right = self._detect_team_tags_from_hud(cap, fps)
                    print(f"[{job_id}] Team detection returned: left={detected_left!r}, right={detected_right!r}")
                except Exception as e:
                    print(f"[{job_id}] Team detection FAILED with exception: {e}")
                    import traceback
                    traceback.print_exc()
                    detected_left, detected_right = None, None
                
                if detected_left and not left_team:
                    self._left_team_code = detected_left
                    self._player_matcher._left_team_code = detected_left
                    print(f"[{job_id}] Auto-detected left team: {detected_left}")
                    
                if detected_right and not right_team:
                    self._right_team_code = detected_right
                    self._player_matcher._right_team_code = detected_right
                    print(f"[{job_id}] Auto-detected right team: {detected_right}")
                
                # Load player pools from database for detected teams
                if (self._left_team_code or self._right_team_code) and (not left_player_pool and not right_player_pool):
                    print(f"[{job_id}] Loading player pools from database for detected teams...")
                    from app.services.db_player_matcher import load_match_players_from_db
                    try:
                        db_left_pool, db_right_pool = load_match_players_from_db(
                            self._left_team_code or "",
                            self._right_team_code or ""
                        )
                        if db_left_pool:
                            self._left_player_pool = db_left_pool
                            print(f"[{job_id}] Loaded {len(db_left_pool)} players for {self._left_team_code}: {db_left_pool}")
                        if db_right_pool:
                            self._right_player_pool = db_right_pool
                            print(f"[{job_id}] Loaded {len(db_right_pool)} players for {self._right_team_code}: {db_right_pool}")
                        # Update local refs for _extract_players_from_video
                        left_player_pool = self._left_player_pool
                        right_player_pool = self._right_player_pool
                    except Exception as e:
                        print(f"[{job_id}] Failed to load player pools from DB: {e}")
                
                # Reset video position after team detection
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Extract player names from first few seconds
            self._job_manager.update_job_status(
                job_id, JobStatus.PROCESSING, "Extracting player names..."
            )
            self._extract_players_from_video(cap, fps, match_players, left_player_pool, right_player_pool)
            
            # Reset video to start
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Initialize detectors with player matcher
            detectors = self._create_detectors()
            
            # Initialize frame state detector
            frame_state_detector = FrameStateDetector()
            
            # Pass player matcher and team codes to killfeed detector
            print(f"[DEBUG] Setting player matcher on {len(detectors)} detectors, matcher={self._player_matcher is not None}, left={self._left_team_code}, right={self._right_team_code}")
            killfeed_detector = None
            top_hud_detector = None
            for d in detectors:
                print(f"[DEBUG] Detector {d.__class__.__name__}: has set_player_matcher={hasattr(d, 'set_player_matcher')}")
                if hasattr(d, 'set_player_matcher'):
                    d.set_player_matcher(self._player_matcher, self._left_team_code, self._right_team_code)
                    print(f"[DEBUG] Called set_player_matcher on {d.__class__.__name__}")
                if isinstance(d, KillfeedDetector):
                    killfeed_detector = d
                if isinstance(d, TopHUDDetector):
                    top_hud_detector = d
            
            # Connect TopHUD halftime detection to KillfeedDetector
            if top_hud_detector and killfeed_detector:
                def on_halftime_change(is_halftime: bool, timestamp_ms: float):
                    if not is_halftime:
                        # Halftime ended - notify killfeed detector to resume
                        killfeed_detector.end_halftime_early(timestamp_ms)
                top_hud_detector.add_halftime_listener(on_halftime_change)
            
            # Pre-compute ROI pixel coordinates
            roi_px_cache = {
                name: roi_to_px(frame_width, frame_height, roi_norm)
                for name, roi_norm in ROI_CONFIG.items()
            }
            
            # Processing loop
            all_events: List[Event] = []
            frame_idx = 0
            sample_interval = max(1, int(fps / self.settings.frame_sample_fps))
            skipped_replay_frames = 0
            skipped_transition_frames = 0
            prev_frame_state = "GAMEPLAY"
            
            self._job_manager.update_job_status(
                job_id, JobStatus.PROCESSING, "Processing frames..."
            )
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames based on sample interval
                if frame_idx % sample_interval != 0:
                    frame_idx += 1
                    continue
                
                t_ms = (frame_idx / fps) * 1000 if fps > 0 else 0
                
                # Detect frame state (GAMEPLAY, REPLAY, TRANSITION)
                replay_roi = crop(frame, roi_px_cache.get("replay_indicator", (0, 0, 1, 1)))
                score_bar_roi = crop(frame, roi_px_cache.get("score_bar", (0, 0, 1, 1)))
                left_panels_roi = crop(frame, roi_px_cache.get("left_panels", (0, 0, 1, 1)))
                right_panels_roi = crop(frame, roi_px_cache.get("right_panels", (0, 0, 1, 1)))
                
                frame_state = frame_state_detector.detect_state(
                    frame, replay_roi, score_bar_roi, left_panels_roi, right_panels_roi
                )
                
                # Emit frame state change event
                if frame_state != prev_frame_state:
                    all_events.append(Event(
                        t_ms=t_ms,
                        type="FRAME_STATE_CHANGE",
                        roi="frame",
                        payload={
                            "from_state": prev_frame_state,
                            "to_state": frame_state,
                        },
                        confidence=0.9
                    ))
                    prev_frame_state = frame_state
                
                # Skip non-gameplay frames
                if frame_state == "REPLAY":
                    skipped_replay_frames += 1
                    frame_idx += 1
                    continue
                elif frame_state == "TRANSITION":
                    skipped_transition_frames += 1
                    frame_idx += 1
                    continue
                
                # Run detectors (only during GAMEPLAY)
                frame_events = []
                for detector in detectors:
                    roi_name = detector.roi_name
                    if roi_name not in roi_px_cache:
                        continue
                    
                    roi_px = roi_px_cache[roi_name]
                    roi_frame = crop(frame, roi_px)
                    
                    if roi_frame.size == 0:
                        continue
                    
                    events = detector.process(t_ms, roi_frame)
                    frame_events.extend(events)
                
                # Process round transitions: notify KillfeedDetector of score changes
                # This enables halftime break detection and score validation
                for event in frame_events:
                    if event.type == "ROUND_TRANSITION":
                        round_num = event.payload.get("round_number", 0)
                        left_score = event.payload.get("left_score", 0)
                        right_score = event.payload.get("right_score", 0)
                        total_rounds_played = left_score + right_score
                        
                        # Update killfeed detector with round info
                        # IMPORTANT: set_round_start tells detector the round that just ENDED
                        # so it can properly assign kills in the buffer window
                        for d in detectors:
                            if hasattr(d, 'set_round_start'):
                                d.set_round_start(t_ms, round_num, left_score, right_score)
                            
                            # Check for halftime: After round 12 ends (total 12 rounds played)
                            # DELAY halftime pause by 5 seconds to capture final kills in killfeed
                            # The killfeed displays kills for ~5s, so kills at round end may
                            # appear in frames AFTER the score transition
                            if total_rounds_played == 12 and hasattr(d, 'set_halftime_start'):
                                # Delay halftime pause to capture end-of-round-12 kills
                                HALFTIME_DELAY_MS = 5000
                                d.set_halftime_start(t_ms + HALFTIME_DELAY_MS)
                
                all_events.extend(frame_events)
                
                frame_idx += 1
            
            cap.release()
            
            # Post-process events
            self._job_manager.update_job_status(
                job_id, JobStatus.PROCESSING, "Post-processing events..."
            )
            
            # Build timeline
            timeline = self._build_timeline(
                job_id=job_id,
                events=all_events,
                duration_ms=duration_ms,
                resolution=[frame_width, frame_height],
                fps=fps,
                filename=os.path.basename(video_path)
            )
            
            # Save outputs
            os.makedirs(output_dir, exist_ok=True)
            
            # Save events
            events_path = os.path.join(output_dir, f"{job_id}_events.json")
            with open(events_path, "w") as f:
                json.dump([asdict(e) for e in all_events], f, indent=2)
            
            # Save timeline
            timeline_path = os.path.join(output_dir, f"{job_id}_timeline.json")
            with open(timeline_path, "w") as f:
                json.dump(timeline, f, indent=2)
            
            # Save kill summary
            kill_events = [e for e in all_events if e.type == "KILL_EVENT"]
            round_transitions = [e for e in all_events if e.type == "ROUND_TRANSITION"]
            summary = self._build_kill_summary(kill_events, round_transitions)
            summary_path = os.path.join(output_dir, f"{job_id}_summary.json")
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            
            # Log skipped frames
            if skipped_replay_frames > 0 or skipped_transition_frames > 0:
                print(f"[{job_id}] Skipped frames - Replay: {skipped_replay_frames}, Transition: {skipped_transition_frames}")
            
            # Update job status
            self._job_manager.update_job_status(
                job_id, JobStatus.COMPLETED,
                f"Processing complete. {len(all_events)} events detected. (Skipped: {skipped_replay_frames} replay, {skipped_transition_frames} transition frames)"
            )
            self._job_manager.add_output_file(job_id, events_path)
            self._job_manager.add_output_file(job_id, timeline_path)
            self._job_manager.add_output_file(job_id, summary_path)
            
            return {
                "job_id": job_id,
                "status": "completed",
                "events_count": len(all_events),
                "kills_count": len(kill_events),
                "skipped_replay_frames": skipped_replay_frames,
                "skipped_transition_frames": skipped_transition_frames,
                "output_files": [events_path, timeline_path, summary_path]
            }
            
        except Exception as e:
            import traceback
            error_msg = f"Processing failed: {str(e)}"
            print(f"[{job_id}] ERROR: {error_msg}")
            traceback.print_exc()
            
            self._job_manager.update_job_status(
                job_id, JobStatus.FAILED, error_msg, error=str(e)
            )
            
            return {
                "job_id": job_id,
                "status": "failed",
                "error": str(e)
            }
    
    def _create_detectors(self) -> List['BaseDetector']:
        """Create detector instances."""
        return [
            KillfeedDetector("killfeed", DETECTOR_FPS["killfeed"]),
            TopHUDDetector("top_hud", DETECTOR_FPS["top_hud"]),
            MinimapDetector("minimap", DETECTOR_FPS["minimap"]),
        ]
    
    def _get_team_name_from_color(self, color: str, round_number: int) -> str:
        """
        Map killfeed color to team name, accounting for halftime COLOR swap.
        
        IMPORTANT: Team POSITIONS stay fixed throughout the match!
        - Left team is ALWAYS on the LEFT side of the HUD
        - Right team is ALWAYS on the RIGHT side of the HUD
        
        Only the COLORS swap:
        - First half (rounds 1-12): left_team = teal, right_team = orange
        - Second half (rounds 13-24): left_team = orange, right_team = teal
        - Overtime (rounds 25+): Colors swap EVERY ROUND
        """
        # Use configured team codes, fallback to NRG/FNC for backward compatibility
        left_team = getattr(self, '_left_team_code', None) or "NRG"
        right_team = getattr(self, '_right_team_code', None) or "FNC"
        
        if round_number <= 12:
            # First half: left_team = teal, right_team = orange
            return left_team if color == "teal" else right_team
        elif round_number <= 24:
            # Second half: left_team = orange, right_team = teal (colors swapped)
            return left_team if color == "orange" else right_team
        else:
            # Overtime: colors swap every round
            # Round 25 starts same as second half, then swaps each round
            ot_round = round_number - 24  # 1, 2, 3, 4...
            if ot_round % 2 == 1:
                # Odd OT rounds (25, 27, 29...): same as second half
                return left_team if color == "orange" else right_team
            else:
                # Even OT rounds (26, 28, 30...): same as first half
                return left_team if color == "teal" else right_team
    
    def _build_timeline(
        self,
        job_id: str,
        events: List[Event],
        duration_ms: float,
        resolution: List[int],
        fps: float,
        filename: str,
    ) -> Dict[str, Any]:
        """Build the timeline structure from events."""
        # Build round boundaries first to determine which round each kill is in
        round_transitions = sorted(
            [e for e in events if e.type == "ROUND_TRANSITION"],
            key=lambda e: e.t_ms
        )
        
        # Create time->round mapping
        def get_round_for_time(t_ms: float) -> int:
            """Get the round number for a given timestamp."""
            round_num = 1
            for trans in round_transitions:
                if t_ms >= trans.t_ms:
                    round_num = trans.payload.get("round_number", round_num + 1)
                else:
                    break
            return round_num
        
        # Extract unique teams from kill events (with halftime correction)
        teams = set()
        for e in events:
            if e.type == "KILL_EVENT":
                round_num = get_round_for_time(e.t_ms)
                killer_color = e.payload.get("killer_team", "")
                victim_color = e.payload.get("victim_team", "")
                if killer_color:
                    teams.add(self._get_team_name_from_color(killer_color, round_num))
                if victim_color:
                    teams.add(self._get_team_name_from_color(victim_color, round_num))
        
        # Build rounds from ROUND_TRANSITION events
        round_transitions = [e for e in events if e.type == "ROUND_TRANSITION"]
        
        # Group events into rounds
        kill_events = [e for e in events if e.type == "KILL_EVENT"]
        
        # Total rounds = number of transitions + 1 (for the starting round)
        # If no transitions detected, we have at least 1 round
        total_rounds = len(round_transitions) + 1 if round_transitions else 1
        
        return {
            "metadata": {
                "vod_id": job_id,
                "filename": filename,
                "duration_ms": duration_ms,
                "resolution": resolution,
                "fps": fps,
                "teams": list(teams),
                "players": [],
                "total_rounds": total_rounds,
                "total_kills": len(kill_events),
            },
            "events": [asdict(e) for e in events],
            "rounds_with_kills": self._build_kill_timeline(kill_events, round_transitions),
            "player_states": [],
            "kill_summary": self._build_kill_summary(kill_events, round_transitions),
            "kill_timeline": self._build_flat_kill_timeline(kill_events, round_transitions),
        }
    
    def _build_kill_timeline(self, kill_events: List[Event], round_transitions: List[Event]) -> List[Dict[str, Any]]:
        """Build a kill timeline organized by rounds, including score tracking.
        
        IMPORTANT: ROUND_TRANSITION events mark the END of a round, not the start.
        - round_number=N in transition means round N just ENDED
        - Kills BEFORE this transition belong to round N
        - Kills shortly AFTER this transition (within buffer) ALSO belong to round N
          (because killfeed has display delay and we sample frames)
        - Kills well AFTER this transition belong to round N+1
        """
        # Sort round transitions by time
        sorted_transitions = sorted(round_transitions, key=lambda e: e.t_ms)
        
        # Buffer: kills within 5s AFTER a transition still count as the ending round
        # This accounts for:
        # - Killfeed display delay (~1-2s)
        # - Frame sampling (~166ms at 6 FPS)  
        # - OCR processing latency
        ROUND_BOUNDARY_BUFFER_MS = 5000
        
        # Build NON-OVERLAPPING round boundaries
        # Round N: from previous_transition_end to this_transition + buffer
        round_boundaries = []
        
        for i, transition in enumerate(sorted_transitions):
            round_num = transition.payload.get("round_number", i + 1)
            
            # Start of this round is either:
            # - Time 0 for round 1
            # - Previous transition + buffer for subsequent rounds
            if i == 0:
                start_ms = 0
            else:
                prev_trans = sorted_transitions[i - 1]
                start_ms = prev_trans.t_ms + ROUND_BOUNDARY_BUFFER_MS
            
            # End of this round is this transition + buffer
            end_ms = transition.t_ms + ROUND_BOUNDARY_BUFFER_MS
            
            # Get scores (scores AFTER round ended)
            left_score = transition.payload.get("left_score", -1)
            right_score = transition.payload.get("right_score", -1)
            
            round_boundaries.append((start_ms, end_ms, round_num, left_score, right_score))
        
        # Add final round (after last transition, goes to infinity)
        if sorted_transitions:
            last_trans = sorted_transitions[-1]
            last_round_num = last_trans.payload.get("round_number", len(sorted_transitions))
            final_round_start = last_trans.t_ms + ROUND_BOUNDARY_BUFFER_MS
            final_round_num = last_round_num + 1
            # Use last known scores
            final_left = last_trans.payload.get("left_score", -1)
            final_right = last_trans.payload.get("right_score", -1)
            round_boundaries.append((final_round_start, float('inf'), final_round_num, final_left, final_right))
        
        # If no round transitions detected, treat everything as round 1
        if not round_boundaries:
            round_boundaries = [(0, float('inf'), 1, 0, 0)]
        
        # Group kills by round
        rounds_data = []
        for start_ms, end_ms, round_num, left_score, right_score in round_boundaries:
            round_kills = []
            for e in kill_events:
                if start_ms <= e.t_ms < end_ms:
                    t_sec = e.t_ms / 1000
                    mins = int(t_sec // 60)
                    secs = t_sec % 60
                    timestamp = f"{mins}:{secs:05.2f}"
                    
                    # Get color from payload and convert to team name with halftime correction
                    killer_color = e.payload.get("killer_team", "unknown")
                    victim_color = e.payload.get("victim_team", "unknown")
                    killer_team = self._get_team_name_from_color(killer_color, round_num) if killer_color != "unknown" else "unknown"
                    victim_team = self._get_team_name_from_color(victim_color, round_num) if victim_color != "unknown" else "unknown"
                    
                    round_kills.append({
                        "t_ms": e.t_ms,
                        "t_seconds": round(t_sec, 1),
                        "timestamp": timestamp,
                        "killer": e.payload.get("killer_name", "Unknown"),
                        "killer_team": killer_team,
                        "killer_color": killer_color,  # Keep original color for debugging
                        "victim": e.payload.get("victim_name", "Unknown"),
                        "victim_team": victim_team,
                        "victim_color": victim_color,  # Keep original color for debugging
                        "weapon": e.payload.get("weapon", "unknown"),
                        "headshot": e.payload.get("is_headshot", False),
                    })
            
            # Sort kills within the round
            round_kills.sort(key=lambda x: x["t_ms"])
            
            if round_kills:  # Only include rounds with kills
                # Calculate round timestamp
                round_start_sec = start_ms / 1000
                round_mins = int(round_start_sec // 60)
                round_secs = round_start_sec % 60
                
                # Use configured team codes
                left_team = getattr(self, '_left_team_code', None) or "left"
                right_team = getattr(self, '_right_team_code', None) or "right"
                
                rounds_data.append({
                    "round_number": round_num,
                    "score": {
                        "left": left_score,
                        "right": right_score,
                        "left_team": left_team,
                        "right_team": right_team,
                    },
                    "round_start_ms": start_ms,
                    "round_start_timestamp": f"{round_mins}:{round_secs:05.2f}",
                    "kills_count": len(round_kills),
                    "kills": round_kills,
                })
        
        return rounds_data
    
    def _get_round_for_time(self, t_ms: float, round_transitions: List[Event]) -> int:
        """Get the round number for a given timestamp.
        
        IMPORTANT: ROUND_TRANSITION events mark the END of a round, not the start.
        - round_number=N in transition means round N just ENDED
        - Kills within BUFFER after transition still belong to round N
        - Kills well after transition (past buffer) belong to round N+1
        
        Note: 5s buffer accounts for frame sampling (~166ms at 6 FPS), OCR latency,
        and killfeed display delay.
        """
        ROUND_BOUNDARY_BUFFER_MS = 5000
        sorted_transitions = sorted(round_transitions, key=lambda e: e.t_ms)
        
        # Start in round 1
        round_num = 1
        
        for trans in sorted_transitions:
            round_num_ended = trans.payload.get("round_number", round_num)
            trans_end_with_buffer = trans.t_ms + ROUND_BOUNDARY_BUFFER_MS
            
            # If time is PAST this transition + buffer, we've moved to the next round
            if t_ms >= trans_end_with_buffer:
                round_num = round_num_ended + 1
                # Continue checking - might be even further rounds
            else:
                # Time is before the buffer expires - we're in round_num_ended (or earlier)
                # The kill belongs to the round that just ended (within buffer)
                # or we haven't reached this transition yet
                if t_ms >= trans.t_ms:
                    # We're in the buffer window - kill belongs to the ending round
                    return round_num_ended
                else:
                    # We're before this transition - we're in round_num
                    return round_num
        
        return round_num
    
    def _build_flat_kill_timeline(self, kill_events: List[Event], round_transitions: List[Event]) -> List[Dict[str, Any]]:
        """Build a flat chronological kill timeline (for backward compatibility)."""
        timeline = []
        for e in kill_events:
            t_sec = e.t_ms / 1000
            mins = int(t_sec // 60)
            secs = t_sec % 60
            timestamp = f"{mins}:{secs:05.2f}"
            
            # Get round number for halftime correction
            round_num = self._get_round_for_time(e.t_ms, round_transitions)
            killer_color = e.payload.get("killer_team", "unknown")
            victim_color = e.payload.get("victim_team", "unknown")
            killer_team = self._get_team_name_from_color(killer_color, round_num) if killer_color != "unknown" else "unknown"
            victim_team = self._get_team_name_from_color(victim_color, round_num) if victim_color != "unknown" else "unknown"
            
            timeline.append({
                "t_ms": e.t_ms,
                "t_seconds": round(t_sec, 1),
                "timestamp": timestamp,
                "round_number": round_num,
                "killer": e.payload.get("killer_name", "Unknown"),
                "killer_team": killer_team,
                "killer_color": killer_color,
                "victim": e.payload.get("victim_name", "Unknown"),
                "victim_team": victim_team,
                "victim_color": victim_color,
                "weapon": e.payload.get("weapon", "unknown"),
                "headshot": e.payload.get("is_headshot", False),
            })
        timeline.sort(key=lambda x: x["t_ms"])
        return timeline
    
    def _build_kill_summary(self, kill_events: List[Event], round_transitions: List[Event]) -> Dict[str, Any]:
        """Build a summary of kills with team assignments."""
        kills_by_player: Dict[str, Dict[str, Any]] = {}
        deaths_by_player: Dict[str, Dict[str, Any]] = {}
        headshots: int = 0
        
        # Initialize team kills with configured team codes
        left_team = getattr(self, '_left_team_code', None) or "left"
        right_team = getattr(self, '_right_team_code', None) or "right"
        team_kills: Dict[str, int] = {left_team: 0, right_team: 0}
        
        for e in kill_events:
            killer = e.payload.get("killer_name", "Unknown")
            victim = e.payload.get("victim_name", "Unknown")
            is_headshot = e.payload.get("is_headshot", False)
            
            # Get round number for halftime correction
            round_num = self._get_round_for_time(e.t_ms, round_transitions)
            killer_color = e.payload.get("killer_team", "unknown")
            victim_color = e.payload.get("victim_team", "unknown")
            killer_team = self._get_team_name_from_color(killer_color, round_num) if killer_color != "unknown" else "unknown"
            victim_team = self._get_team_name_from_color(victim_color, round_num) if victim_color != "unknown" else "unknown"
            
            # Track kills by player with team
            if killer not in kills_by_player:
                kills_by_player[killer] = {"kills": 0, "team": killer_team}
            kills_by_player[killer]["kills"] += 1
            
            # Track deaths by player with team
            if victim not in deaths_by_player:
                deaths_by_player[victim] = {"deaths": 0, "team": victim_team}
            deaths_by_player[victim]["deaths"] += 1
            
            # Track team kills
            if killer_team in team_kills:
                team_kills[killer_team] += 1
            
            if is_headshot:
                headshots += 1
        
        return {
            "total_kills": len(kill_events),
            "total_headshots": headshots,
            "kills_by_team": team_kills,
            "kills_by_player": kills_by_player,
            "deaths_by_player": deaths_by_player,
        }
    
    def _detect_team_tags_from_hud(
        self,
        cap: cv2.VideoCapture,
        fps: float,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Detect team tags from the top HUD by OCR-ing the team tag regions.
        Returns (left_team_tag, right_team_tag) or (None, None) if detection fails.
        
        IMPROVED: First finds frames with valid game score (0-0 or higher) to ensure
        we're sampling during actual gameplay, not intro/lobby screens.
        """
        import re
        from collections import Counter
        
        # Initialize EasyOCR directly for better control
        try:
            import easyocr
            import os
            use_gpu = os.environ.get('USE_GPU', 'false').lower() == 'true'
            ocr_reader = easyocr.Reader(['en'], gpu=use_gpu, verbose=False)
            print(f"[TeamTagDetector] Initialized EasyOCR with GPU={use_gpu}")
        except Exception as e:
            print(f"[TeamTagDetector] Failed to initialize EasyOCR: {e}")
            return None, None
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Get ROI pixel coordinates
        left_tag_roi = ROI_CONFIG.get("top_left_team_tag")
        right_tag_roi = ROI_CONFIG.get("top_right_team_tag")
        left_score_roi = ROI_CONFIG.get("top_left_score")
        right_score_roi = ROI_CONFIG.get("top_right_score")
        
        if not left_tag_roi or not right_tag_roi:
            print("[TeamTagDetector] Team tag ROIs not configured")
            return None, None
        
        left_tag_px = roi_to_px(frame_width, frame_height, left_tag_roi)
        right_tag_px = roi_to_px(frame_width, frame_height, right_tag_roi)
        left_score_px = roi_to_px(frame_width, frame_height, left_score_roi) if left_score_roi else None
        right_score_px = roi_to_px(frame_width, frame_height, right_score_roi) if right_score_roi else None
        
        original_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        
        # STEP 1: Find frames with valid game score (indicates match has started)
        # Scan from 1 minute to 15 minutes to find gameplay
        print(f"[TeamTagDetector] Scanning for gameplay frames with valid score...")
        
        valid_gameplay_frames = []
        scan_times_sec = list(range(60, 900, 15))  # Every 15 seconds from 1-15 min
        
        for t_sec in scan_times_sec:
            frame_num = int(t_sec * fps)
            if frame_num >= total_frames:
                break
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Check if this frame has a valid score visible
            if left_score_px and right_score_px:
                left_score_img = crop(frame, left_score_px)
                right_score_img = crop(frame, right_score_px)
                
                # OCR the scores
                left_score = self._ocr_single_digit(left_score_img, ocr_reader)
                right_score = self._ocr_single_digit(right_score_img, ocr_reader)
                
                # Valid if both scores are readable numbers 0-20
                if left_score is not None and right_score is not None:
                    valid_gameplay_frames.append((t_sec, frame_num, left_score, right_score))
                    if len(valid_gameplay_frames) >= 10:  # Found enough valid frames
                        break
        
        if not valid_gameplay_frames:
            print(f"[TeamTagDetector] No valid gameplay frames found, falling back to fixed times")
            # Fallback to fixed sample times
            valid_gameplay_frames = [(t, int(t * fps), -1, -1) for t in [180, 240, 300, 360, 420]]
        else:
            print(f"[TeamTagDetector] Found {len(valid_gameplay_frames)} valid gameplay frames")
            for t_sec, frame_num, ls, rs in valid_gameplay_frames[:3]:
                print(f"  t={t_sec}s: score {ls}-{rs}")
        
        # STEP 2: OCR team tags from valid gameplay frames
        left_detections = []
        right_detections = []
        
        for t_sec, frame_num, _, _ in valid_gameplay_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Extract and OCR team tags with enhanced preprocessing
            left_img = crop(frame, left_tag_px)
            left_tag = self._ocr_team_tag_enhanced(left_img, ocr_reader)
            if left_tag:
                left_detections.append(left_tag)
                print(f"[TeamTagDetector] t={t_sec}s: left='{left_tag}'")
            
            right_img = crop(frame, right_tag_px)
            right_tag = self._ocr_team_tag_enhanced(right_img, ocr_reader)
            if right_tag:
                right_detections.append(right_tag)
                print(f"[TeamTagDetector] t={t_sec}s: right='{right_tag}'")
        
        # Restore original position
        cap.set(cv2.CAP_PROP_POS_FRAMES, original_pos)
        
        # Find most common detection for each side
        left_team = None
        right_team = None
        
        if left_detections:
            left_counter = Counter(left_detections)
            left_team = left_counter.most_common(1)[0][0]
            print(f"[TeamTagDetector] Left team detections: {left_counter} -> '{left_team}'")
        else:
            print(f"[TeamTagDetector] No left team detections!")
        
        if right_detections:
            right_counter = Counter(right_detections)
            right_team = right_counter.most_common(1)[0][0]
            print(f"[TeamTagDetector] Right team detections: {right_counter} -> '{right_team}'")
        else:
            print(f"[TeamTagDetector] No right team detections!")
        
        return left_team, right_team
    
    def _ocr_single_digit(self, img: np.ndarray, ocr_reader) -> Optional[int]:
        """OCR a single digit/number from a score region."""
        if img is None or img.size == 0:
            return None
        try:
            results = ocr_reader.readtext(img, allowlist='0123456789')
            if results and len(results) > 0:
                text = results[0][1]
                conf = results[0][2]
                if conf >= 0.5:
                    return int(text)
        except:
            pass
        return None
    
    def _ocr_team_tag_enhanced(self, img: np.ndarray, ocr_reader) -> Optional[str]:
        """OCR a team tag with enhanced preprocessing for small white text."""
        import re
        
        if img is None or img.size == 0:
            return None
        
        # Scale up significantly for small text (team tags are ~30px tall)
        # Use scale 6x - testing showed this gives best results (91.7% conf for DRX)
        scale = 6
        scaled = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # Convert to grayscale
        if len(scaled.shape) == 3:
            gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
        else:
            gray = scaled
        
        # Multiple preprocessing approaches - prioritize approaches that work best
        preprocessed_images = []
        
        # 1. Original scaled color (BEST for DRX detection - 91.7% conf)
        # EasyOCR handles color well and this avoids D->O threshold errors
        preprocessed_images.append(scaled)
        
        # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        preprocessed_images.append(enhanced)
        
        # 3. Grayscale only (no threshold to avoid D->O errors)
        preprocessed_images.append(gray)
        
        # NOTE: Removed threshold-based approaches as they cause D->O misreads
        # (threshold makes 'D' look like 'O' by filling in the curve)
        
        results = []
        
        # Known team tags for fuzzy matching
        known_tags = {'NRG', 'FNC', 'SEN', 'C9', 'TL', 'TSM', '100T', 'G2', 'VIT', 
                     'FUT', 'LEV', 'KRU', 'LOUD', 'PRX', 'DRX', 'T1', 'GEN', 'EDG',
                     'FPX', 'TH', 'KC', 'BBL', 'FUR', 'MIBR', 'NIP', 'EG', 'FNATIC',
                     'SENTINELS', 'CLOUD9', 'GENG', 'HERETICS', 'PAPER', 'REX'}
        
        for img_version in preprocessed_images:
            try:
                # Use EasyOCR with allowlist for uppercase letters and numbers
                ocr_results = ocr_reader.readtext(
                    img_version, 
                    allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                    paragraph=False,
                    min_size=5
                )
                for bbox, text, conf in ocr_results:
                    if text and conf >= 0.3:
                        clean = text.upper().strip()
                        results.append((clean, conf))
            except Exception as e:
                pass
        
        if not results:
            return None
        
        # Sort by confidence
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Common OCR misreads for team tags
        ocr_corrections = {
            'MH': 'TH',   # T often misread as M
            'WH': 'TH',   # T misread as W
            'YRG': 'NRG', # N misread as Y
            'MRG': 'NRG', # N misread as M
            'IL': 'TL',   # T misread as I
            '1L': 'TL',   # T misread as 1
            'T1': 'TL',   # L misread as 1
            'PRK': 'PRX', # X misread as K
            'DRK': 'DRX', # X misread as K
            'ORX': 'DRX', # D misread as O
            'OPX': 'DRX', # D misread as O, R as P
        }
        
        # First, look for exact matches to known tags
        for text, conf in results:
            clean = re.sub(r'[^A-Z0-9]', '', text)
            # Apply OCR corrections first
            if clean in ocr_corrections:
                corrected = ocr_corrections[clean]
                if corrected in known_tags:
                    print(f"[TeamTagOCR] Corrected '{clean}' -> '{corrected}'")
                    return corrected
            if clean in known_tags:
                return clean
        
        # Try fuzzy matching against known tags
        for text, conf in results:
            clean = re.sub(r'[^A-Z0-9]', '', text)
            if 2 <= len(clean) <= 6:
                # Check for close matches (1-2 char difference)
                for tag in known_tags:
                    if len(tag) == len(clean):
                        diff = sum(1 for a, b in zip(tag, clean) if a != b)
                        if diff <= 1:  # Allow 1 character difference
                            return tag
        
        # Last resort: return the highest confidence short string
        for text, conf in results:
            clean = re.sub(r'[^A-Z0-9]', '', text)
            if 2 <= len(clean) <= 5 and conf >= 0.5:
                return clean
        
        return None

    def _extract_players_from_video(
        self,
        cap: cv2.VideoCapture, 
        fps: float,
        match_players: Optional[List[str]] = None,
        left_player_pool: Optional[List[str]] = None,
        right_player_pool: Optional[List[str]] = None,
    ):
        """
        Extract player names from the first few seconds of video.
        Uses the player card slots on left and right sides of the HUD.
        
        If match_players is provided, expects format like:
          - Single string: "NRG:Ethan,Brawk,Mada;FNC:Boaster,Alfajer" (semicolon separates teams)
          - Or list of individual names
          
        If player pools are provided (left_player_pool, right_player_pool), OCR results
        will be validated against these pools to find the actual 5 players in the match.
        """
        from urllib.parse import unquote
        
        if match_players:
            # URL-decode the player string if it contains encoded characters
            decoded_players = [unquote(p) for p in match_players]
            
            # Handle case where it's passed as a list with the format string
            if len(decoded_players) == 1 and ';' in decoded_players[0]:
                # Single string with both teams: "NRG:Ethan,Brawk;FNC:Boaster,Alfajer"
                team_str = decoded_players[0]
            elif len(decoded_players) > 1 and ';' in decoded_players[-1]:
                # Multiple strings where team separator leaked into parsing
                team_str = ','.join(decoded_players)
            else:
                team_str = None
            
            if team_str and ';' in team_str:
                # Parse "TeamA:Player1,Player2;TeamB:Player3,Player4" format
                teams_data = team_str.split(';')
                left = []
                right = []
                
                for i, team_data in enumerate(teams_data[:2]):  # Only take first 2 teams
                    if ':' in team_data:
                        team_name, players_str = team_data.split(':', 1)
                        players = [p.strip() for p in players_str.split(',') if p.strip()]
                    else:
                        team_name = f"Team{i+1}"
                        players = [p.strip() for p in team_data.split(',') if p.strip()]
                    
                    if i == 0:
                        left = players
                    else:
                        right = players
                
                print(f"Using provided player list:")
                print(f"  Left team: {left}")
                print(f"  Right team: {right}")
                print(f"Manually set {len(left)} left, {len(right)} right players")
                self._player_matcher.set_match_players(left, right)
                return
            
            # Fallback: Group players by team prefix (e.g., "NRG Ethan")
            team_groups: Dict[str, List[str]] = {}
            for player in decoded_players:
                # Extract team prefix (first word before space)
                parts = player.strip().split()
                if len(parts) >= 2:
                    team_prefix = parts[0].upper()
                    team_groups.setdefault(team_prefix, []).append(player.strip())
                else:
                    # Single word name - put in 'Other' group
                    team_groups.setdefault("Other", []).append(player.strip())
            
            # Get the two main teams
            teams = list(team_groups.keys())
            if len(teams) >= 2:
                # Use first two team groups
                team1, team2 = teams[0], teams[1]
                left = team_groups[team1]
                right = team_groups[team2]
            else:
                # Fall back to splitting in half
                mid = len(decoded_players) // 2
                left = decoded_players[:mid] if mid > 0 else []
                right = decoded_players[mid:] if mid > 0 else decoded_players
            
            print(f"Using provided player list (fallback):")
            print(f"  Left team: {left}")
            print(f"  Right team: {right}")
            self._player_matcher.set_match_players(left, right)
            return
        
        from app.services.player_name_extractor import PlayerNameExtractor
        
        extractor = PlayerNameExtractor()
        
        # Sample frames from first 30 seconds to find player names
        sample_frames = []
        max_sample_time = 30 * fps  # 30 seconds
        sample_interval = int(fps * 2)  # Every 2 seconds
        
        frame_idx = 0
        while frame_idx < max_sample_time:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            sample_frames.append(frame)
            frame_idx += sample_interval
        
        if not sample_frames:
            print("WARNING: No frames found for player extraction")
            return
        
        # Extract players from sampled frames
        all_left_players = []
        all_right_players = []
        
        for frame in sample_frames:
            left, right = extractor.extract_players_from_frame(frame)
            all_left_players.extend([p.name for p in left])
            all_right_players.extend([p.name for p in right])
        
        # Get most common names for each team
        def get_top_names(names: List[str], n: int = 5) -> List[str]:
            """Get the N most frequent names."""
            from collections import Counter
            counts = Counter([name for name in names if name and len(name) >= 2])
            return [name for name, _ in counts.most_common(n * 2)][:n]
        
        def validate_against_pool(ocr_names: List[str], pool: List[str]) -> List[str]:
            """
            Validate OCR names against a pool of known players.
            Returns validated names that match the pool.
            """
            if not pool:
                return ocr_names
            
            from difflib import SequenceMatcher
            
            validated = []
            pool_lower = {p.lower(): p for p in pool}
            
            for ocr_name in ocr_names:
                ocr_lower = ocr_name.lower().strip()
                
                # Exact match
                if ocr_lower in pool_lower:
                    validated.append(pool_lower[ocr_lower])
                    continue
                
                # Fuzzy match against pool
                best_match = None
                best_score = 0.0
                
                for pool_name in pool:
                    score = SequenceMatcher(None, ocr_lower, pool_name.lower()).ratio()
                    if score > best_score and score > 0.6:
                        best_score = score
                        best_match = pool_name
                
                if best_match:
                    print(f"  OCR '{ocr_name}' matched to '{best_match}' (score: {best_score:.2f})")
                    if best_match not in validated:
                        validated.append(best_match)
                else:
                    print(f"  OCR '{ocr_name}' no match in pool (best score: {best_score:.2f})")
            
            return validated
        
        left_team_players = get_top_names(all_left_players)
        right_team_players = get_top_names(all_right_players)
        
        print(f"Extracted raw OCR players from HUD:")
        print(f"  Left team raw: {left_team_players}")
        print(f"  Right team raw: {right_team_players}")
        
        # Validate against player pools if provided
        if left_player_pool:
            print(f"Validating left team against pool ({len(left_player_pool)} candidates)...")
            left_team_players = validate_against_pool(left_team_players, left_player_pool)
        if right_player_pool:
            print(f"Validating right team against pool ({len(right_player_pool)} candidates)...")
            right_team_players = validate_against_pool(right_team_players, right_player_pool)
        
        if left_team_players or right_team_players:
            print(f"Final validated players:")
            print(f"  Left team: {left_team_players}")
            print(f"  Right team: {right_team_players}")
            self._player_matcher.set_match_players(left_team_players, right_team_players)
        else:
            print("WARNING: Could not extract player names from HUD")
            # IMPORTANT: Still call set_match_players to load team rosters from database
            # This is critical for matching players when HUD extraction fails
            self._player_matcher.set_match_players([], [])


# ======================================
# Frame State Detector
# ======================================
class FrameStateDetector:
    """
    Detects the current frame state to avoid processing non-gameplay frames.
    
    States:
    - GAMEPLAY: Normal gameplay with full HUD (should process)
    - REPLAY: Replay/highlight mode (skip to avoid duplicates)
    - TRANSITION: Player cam, pre-match, etc. (skip - no HUD)
    """
    
    def __init__(self):
        self._ocr_reader = None
        self._ocr_initialized = False
        self._last_state = "GAMEPLAY"
        self._state_count = 0  # For hysteresis
    
    def _init_ocr(self):
        """Lazily initialize OCR for replay text detection."""
        if self._ocr_initialized:
            return
        try:
            import easyocr
            import os
            use_gpu = os.environ.get('USE_GPU', 'false').lower() == 'true'
            self._ocr_reader = easyocr.Reader(['en'], gpu=use_gpu, verbose=False)
        except Exception:
            self._ocr_reader = None
        self._ocr_initialized = True
    
    def detect_state(
        self, 
        frame: np.ndarray,
        replay_roi: np.ndarray,
        score_bar_roi: np.ndarray,
        left_panels_roi: np.ndarray,
        right_panels_roi: np.ndarray,
    ) -> str:
        """
        Detect the current frame state.
        
        Returns:
            "GAMEPLAY" - Normal gameplay, process events
            "REPLAY" - Replay mode, skip killfeed to avoid duplicates
            "TRANSITION" - Non-gameplay screen, skip all processing
        """
        # Check for REPLAY or CLUTCH indicator (skip killfeed during these overlays)
        if self._detect_replay_or_clutch_text(replay_roi):
            return "REPLAY"
        
        # Check if standard HUD is present
        if not self._has_standard_hud(score_bar_roi, left_panels_roi, right_panels_roi):
            return "TRANSITION"
        
        return "GAMEPLAY"
    
    def _detect_replay_or_clutch_text(self, replay_roi: np.ndarray) -> bool:
        """
        Detect if "REPLAY" or "CLUTCH" text is visible in the bottom-right corner.
        Uses OCR to look for these overlay texts.
        
        Both REPLAY and CLUTCH overlays indicate segments where we should skip
        killfeed processing to avoid duplicate detection:
        - REPLAY: Obviously replay footage
        - CLUTCH: Often shown during replay highlights of clutch moments
        
        Returns True if either REPLAY or CLUTCH text is detected.
        """
        if replay_roi.size == 0:
            return False
        
        h, w = replay_roi.shape[:2]
        
        # Look for high-contrast white text on dark background
        gray = cv2.cvtColor(replay_roi, cv2.COLOR_BGR2GRAY)
        
        # The REPLAY/CLUTCH text is typically white/light on dark
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        white_ratio = np.sum(thresh > 0) / thresh.size
        
        # If there's significant white text, try OCR
        if white_ratio > 0.05 and white_ratio < 0.6:
            self._init_ocr()
            if self._ocr_reader:
                try:
                    # Use broader allowlist to detect both REPLAY and CLUTCH
                    results = self._ocr_reader.readtext(
                        replay_roi, 
                        detail=0,
                        paragraph=True,
                        allowlist='REPLAYreplayCLUTCHclutch'
                    )
                    for text in results:
                        if isinstance(text, str):
                            text_upper = text.upper().replace(" ", "")
                            # CLUTCH = skip to avoid replay duplicates
                            if "CLUTCH" in text_upper:
                                return True
                            # REPLAY = replay footage, skip to avoid duplicates
                            if "REPLAY" in text_upper or "REPLA" in text_upper:
                                return True
                except Exception:
                    pass
        
        # Removed heuristic method - it was causing false positives
        # The dark banner + white text pattern matches too many things
        # (player cards, health bars, etc.)
        
        return False
    
    def _has_standard_hud(
        self,
        score_bar_roi: np.ndarray,
        left_panels_roi: np.ndarray, 
        right_panels_roi: np.ndarray,
    ) -> bool:
        """
        Check if standard gameplay HUD elements are present.
        During player cams or transitions, these regions will be dark or have
        non-standard content.
        """
        if score_bar_roi.size == 0:
            return False
        
        # Check score bar - should have bright/colorful elements (timer, scores)
        score_gray = cv2.cvtColor(score_bar_roi, cv2.COLOR_BGR2GRAY)
        score_brightness = np.mean(score_gray)
        
        # Score bar should have moderate to high brightness during gameplay
        if score_brightness < 40:
            return False
        
        # Check for presence of team-colored elements in player panels
        # During gameplay, these should have teal/orange colored health bars
        left_has_color = self._has_team_colors(left_panels_roi)
        right_has_color = self._has_team_colors(right_panels_roi)
        
        # At least one side should have team colors visible
        if not left_has_color and not right_has_color:
            # Give some slack - maybe panels are covered by ability effects
            # Check if score bar has expected structure
            score_edges = cv2.Canny(score_gray, 50, 150)
            edge_density = np.sum(score_edges > 0) / score_edges.size
            
            # Score bar should have some edge structure from numbers/icons
            if edge_density < 0.02:
                return False
        
        return True
    
    def _has_team_colors(self, panel_roi: np.ndarray) -> bool:
        """Check if panel region has team colors (teal or orange)."""
        if panel_roi.size == 0:
            return False
        
        hsv = cv2.cvtColor(panel_roi, cv2.COLOR_BGR2HSV)
        
        # Check for teal (H: 75-105)
        teal_mask = cv2.inRange(hsv, (75, 80, 100), (105, 255, 255))
        teal_ratio = np.sum(teal_mask > 0) / teal_mask.size
        
        # Check for orange (H: 0-25)
        orange_mask = cv2.inRange(hsv, (0, 80, 100), (25, 255, 255))
        orange_ratio = np.sum(orange_mask > 0) / orange_mask.size
        
        # Should have at least some team color pixels
        return teal_ratio > 0.005 or orange_ratio > 0.005


# ======================================
# Detector Base Class
# ======================================
class BaseDetector:
    """Base class for all detectors."""
    
    def __init__(self, roi_name: str, target_fps: float):
        self.roi_name = roi_name
        self.gate = RateGate(target_fps)
        self._prev_gray = None
    
    def should_run(self, t_ms: float) -> bool:
        """Check if detector should run at this timestamp."""
        return self.gate.due(t_ms)
    
    def process(self, t_ms: float, roi_frame: np.ndarray) -> List[Event]:
        """Process a frame and return detected events."""
        if not self.should_run(t_ms):
            return []
        return self._detect(t_ms, roi_frame)
    
    def _detect(self, t_ms: float, roi_frame: np.ndarray) -> List[Event]:
        """Override in subclasses to implement detection logic."""
        raise NotImplementedError


# ======================================
# Killfeed Detector
# ======================================
class KillfeedDetector(BaseDetector):
    """Detects kill events from the killfeed region."""
    
    # Per-round dedup: A player can only die ONCE per round in Valorant
    # Track when each victim last died to prevent duplicate detections
    ROUND_DEDUP_WINDOW_MS = 90000  # 90 seconds - longer than any round duration
    
    def __init__(self, roi_name: str, target_fps: float):
        super().__init__(roi_name, target_fps)
        self.recent_signatures: List[Tuple[float, str, str, str, str, int]] = []  # (t_ms, killer_team, victim_team, killer_name, victim_name, row_idx)
        self._ocr_reader = None
        self._ocr_initialized = False
        self._player_matcher = None
        
        # Change detection to skip OCR when killfeed hasn't changed
        self._prev_hash = None
        self._no_change_count = 0
        self._SKIP_THRESHOLD = 3  # Skip OCR if unchanged for 3 consecutive frames
        self._left_team_code = None
        self._right_team_code = None
        
        # Per-row hash cache to skip OCR on unchanged rows
        self._row_hashes: Dict[int, int] = {}  # row_index -> perceptual hash
        
        # NEW: Per-round kill tracking to prevent replay/duplicate detection
        # Key: normalized victim name, Value: (last_death_timestamp_ms, killer_name)
        self._victim_last_death: Dict[str, Tuple[float, str]] = {}
        # Track current round start time (set via set_round_start)
        self._current_round_start_ms: float = 0.0
        # Track current round number for halftime detection (starts at 1, not 0)
        self._current_round_number: int = 1
        # In VALORANT, halftime is at round 12 (rounds 1-12 = first half, 13+ = second half)
        self._HALFTIME_ROUND = 12
        
        # Score validation: Track expected scores to reject replay/ad clips
        self._expected_left_score: int = 0
        self._expected_right_score: int = 0
        # Set when entering halftime break (between rounds 12 and 13)
        self._in_halftime_break: bool = False
        self._halftime_start_ms: float = 0.0
        # Halftime break in broadcasts can be 3-5 minutes (including ads/replays/analysis)
        # Using 300 seconds (5 min) as max - will exit early when round 13 is detected
        self._HALFTIME_DURATION_MS: float = 300000  # 5 minutes max
        # Post-halftime cooldown: filter replay kills that appear immediately after halftime
        # Replays show first-half highlights, need short cooldown before detecting new kills
        self._POST_HALFTIME_COOLDOWN_MS: float = 2000  # 2 seconds
        self._halftime_end_ms: float = 0.0
        # Scheduled halftime start (delayed from transition to capture final kills)
        self._halftime_scheduled_ms: float = 0.0
        # Track the round transition time for accurate round display in buffer window
        self._last_transition_ms: float = 0.0
        self._last_transition_round: int = 0
    
    def set_round_start(self, timestamp_ms: float, round_number: int = 0, left_score: int = 0, right_score: int = 0):
        """
        Called when a new round starts - clears per-round tracking and updates expected scores.
        
        Args:
            timestamp_ms: Round start timestamp
            round_number: The round that just ENDED (1-indexed)
            left_score: Expected left team score (rounds won)
            right_score: Expected right team score (rounds won)
        """
        total_rounds_played = left_score + right_score
        
        # Exit halftime when we see the first round of second half (round 13 = total 13)
        if self._in_halftime_break and total_rounds_played > self._HALFTIME_ROUND:
            self._in_halftime_break = False
            self._halftime_end_ms = timestamp_ms
            duration = timestamp_ms - self._halftime_start_ms
            print(f"[KillfeedDetector] Halftime break ended at {timestamp_ms:.0f}ms (duration: {duration/1000:.1f}s) - detection RESUMED (replays filtered by deduplication)")
        
        self._current_round_start_ms = timestamp_ms
        # Track transition for buffer-aware round display
        self._last_transition_ms = timestamp_ms
        self._last_transition_round = round_number  # The round that just ended
        # round_number is the round that ENDED, so current round being PLAYED is +1
        self._current_round_number = round_number + 1
        self._expected_left_score = left_score
        self._expected_right_score = right_score
        
        # Clear victim death tracking for new round (players respawn)
        self._victim_last_death.clear()
        print(f"[KillfeedDetector] Round {round_number} ended, now in round {self._current_round_number} at {timestamp_ms:.0f}ms (score: {left_score}-{right_score}) - cleared death tracking")
    
    def set_halftime_start(self, timestamp_ms: float):
        """Called when halftime break begins (after round 12 ends).
        
        Note: timestamp_ms should be delayed from the transition time to allow
        capturing kills that appear in the killfeed after the score changes.
        """
        self._halftime_scheduled_ms = timestamp_ms
        print(f"[KillfeedDetector] Halftime break scheduled at {timestamp_ms:.0f}ms - will pause when reached")
    
    def end_halftime_early(self, timestamp_ms: float):
        """Called when game HUD becomes visible again (scores showing after halftime).
        
        This allows us to capture round 13 kills instead of waiting for the score change.
        """
        if self._in_halftime_break:
            self._in_halftime_break = False
            self._halftime_end_ms = timestamp_ms
            duration = timestamp_ms - self._halftime_start_ms
            print(f"[KillfeedDetector] Halftime break ended EARLY at {timestamp_ms:.0f}ms (duration: {duration/1000:.1f}s) - detection RESUMED with {self._POST_HALFTIME_COOLDOWN_MS/1000:.0f}s cooldown")
            return True
        return False
    
    def is_score_valid(self, detected_left_score: int, detected_right_score: int) -> bool:
        """
        Validate if a detected score is consistent with expected match state.
        
        This rejects kills from replay clips/ads that show impossible scores.
        For example: Actual match at 11-1, but replay shows 11-3 = invalid.
        
        Rules:
        1. Score sum must equal rounds played (current_round - 1)
        2. Each score must be <= current maximum possible
        3. Score should be close to expected values (within +/- 1 for timing issues)
        """
        expected_sum = self._expected_left_score + self._expected_right_score
        detected_sum = detected_left_score + detected_right_score
        
        # Rule 1: Score sum should match rounds completed (with small tolerance)
        if abs(detected_sum - expected_sum) > 1:
            return False
        
        # Rule 2: Neither score should exceed max possible
        max_score = max(self._expected_left_score, self._expected_right_score) + 1
        if detected_left_score > max_score or detected_right_score > max_score:
            return False
        
        return True
    
    def is_second_half(self) -> bool:
        """Check if we're in the second half (after halftime side swap)."""
        return self._current_round_number > self._HALFTIME_ROUND
    
    def is_colors_swapped(self) -> bool:
        """
        Check if colors are currently swapped from initial state.
        
        VALORANT color swap rules:
        - First half (rounds 1-12): NOT swapped
        - Second half (rounds 13-24): Swapped (after halftime)
        - Overtime (rounds 25+): Swap EVERY round
          - Round 25: Same as second half (swapped) 
          - Round 26: Swap back (not swapped)
          - Round 27: Swap again (swapped)
          - etc.
        """
        round_num = self._current_round_number
        
        if round_num <= 12:
            # First half: not swapped
            return False
        elif round_num <= 24:
            # Second half: swapped
            return True
        else:
            # Overtime: swap every round
            # Round 25 starts swapped (same as second half end)
            # Then alternates: 25=swapped, 26=not, 27=swapped, etc.
            ot_round = round_num - 24  # 1, 2, 3, 4...
            return ot_round % 2 == 1  # Odd OT rounds are swapped
    
    def get_actual_team(self, detected_color: str) -> str:
        """
        Convert detected color to actual team accounting for halftime and overtime swaps.
        
        In first half: teal = left team, orange = right team
        In second half: teal = right team, orange = left team (sides swapped)
        In overtime: colors swap EVERY round
        """
        if self.is_colors_swapped():
            # Colors are swapped - invert
            return "orange" if detected_color == "teal" else "teal"
        return detected_color
    
    def get_team_code_from_color(self, detected_color: str) -> Optional[str]:
        """
        Convert detected color to team CODE (e.g., 'NRG', 'FNC') for player matching.
        
        This is CRITICAL for correctly matching players who have played for multiple teams.
        For example, Crashies was on NRG but now plays for FNC.
        
        Args:
            detected_color: The color detected in killfeed ('teal', 'orange', etc.)
            
        Returns:
            Team code (e.g., 'NRG', 'FNC') or None if can't determine
        """
        # First, account for halftime/overtime color swaps
        actual_color = self.get_actual_team(detected_color)
        
        # Map color to side
        # In VALORANT standard layout: teal = attackers (left side), orange = defenders (right side)
        # After swap: teal = right, orange = left
        if actual_color in ('teal', 'green', 'cyan'):
            return self._left_team_code
        elif actual_color in ('orange', 'red', 'yellow'):
            return self._right_team_code
        
        return None
    
    def get_color_from_team_code(self, team_code: str) -> Optional[str]:
        """
        Convert team CODE (e.g., 'NRG', 'FNC') back to color for event tracking.
        
        This is the reverse of get_team_code_from_color, used when we extract
        the team from OCR text and need to update the color-based team tracking.
        
        Args:
            team_code: Team code (e.g., 'NRG', 'FNC')
            
        Returns:
            Color string ('teal' or 'orange') accounting for halftime swap
        """
        if not team_code:
            return None
        
        # Determine which side this team is on
        is_left_team = self._left_team_code and team_code.upper() == self._left_team_code.upper()
        is_right_team = self._right_team_code and team_code.upper() == self._right_team_code.upper()
        
        if not is_left_team and not is_right_team:
            return None
        
        # Base color mapping (first half): left = teal, right = orange
        # After halftime (round 13+): left = orange, right = teal
        # After OT round 24: swap every round
        round_num = getattr(self, '_current_round_number', 1)
        
        if round_num <= 12:
            # First half: left = teal, right = orange
            return "teal" if is_left_team else "orange"
        elif round_num <= 24:
            # Second half: left = orange, right = teal (swapped)
            return "orange" if is_left_team else "teal"
        else:
            # Overtime: swap every round
            ot_round = round_num - 24
            if ot_round % 2 == 1:
                # Odd OT (25, 27...): same as second half
                return "orange" if is_left_team else "teal"
            else:
                # Even OT (26, 28...): same as first half
                return "teal" if is_left_team else "orange"
    
    def set_player_matcher(self, player_matcher, left_team_code: str = None, right_team_code: str = None):
        """Set the player matcher for fuzzy name matching."""
        self._player_matcher = player_matcher
        self._left_team_code = left_team_code
        self._right_team_code = right_team_code
        print(f"KillfeedDetector: set team codes left={left_team_code}, right={right_team_code}")
    
    def _init_ocr(self):
        """Lazily initialize OCR using the enhanced OCR engine."""
        if self._ocr_initialized:
            return
        
        try:
            import os
            from app.services.ocr_engine import get_ocr_engine
            use_gpu = os.environ.get('USE_GPU', 'false').lower() == 'true'
            self._ocr_engine = get_ocr_engine(use_gpu=use_gpu)
            print(f"KillfeedDetector: Using OCR engine ({self._ocr_engine.backend}) with GPU={use_gpu}")
            # Keep backward compatibility
            self._ocr_reader = self._ocr_engine
        except Exception as e:
            print(f"KillfeedDetector: OCR engine unavailable ({e}), trying EasyOCR directly")
            try:
                import easyocr
                use_gpu = os.environ.get('USE_GPU', 'false').lower() == 'true'
                self._ocr_reader = easyocr.Reader(['en'], gpu=use_gpu, verbose=False)
                print(f"KillfeedDetector: Using EasyOCR fallback with GPU={use_gpu}")
            except Exception as e2:
                print(f"KillfeedDetector: EasyOCR also unavailable ({e2}), using color detection only")
                self._ocr_reader = None
        
        self._ocr_initialized = True
    
    def _compute_frame_hash(self, roi_frame: np.ndarray) -> int:
        """Compute a fast hash of the frame for change detection."""
        # Downsample and convert to grayscale for fast comparison
        small = cv2.resize(roi_frame, (32, 32), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        # Simple hash based on mean brightness in quadrants
        h, w = gray.shape
        q1 = np.mean(gray[:h//2, :w//2])
        q2 = np.mean(gray[:h//2, w//2:])
        q3 = np.mean(gray[h//2:, :w//2])
        q4 = np.mean(gray[h//2:, w//2:])
        return int(q1) * 1000000 + int(q2) * 10000 + int(q3) * 100 + int(q4)
    
    def _has_significant_change(self, roi_frame: np.ndarray) -> bool:
        """Check if the killfeed has changed significantly."""
        current_hash = self._compute_frame_hash(roi_frame)
        
        if self._prev_hash is None:
            self._prev_hash = current_hash
            return True
        
        # Check if hash changed significantly
        hash_diff = abs(current_hash - self._prev_hash)
        changed = hash_diff > 500  # Threshold for significant change
        
        if changed:
            self._no_change_count = 0
            self._prev_hash = current_hash
            return True
        else:
            self._no_change_count += 1
            # Always process occasionally even if no change (in case we missed something)
            if self._no_change_count >= self._SKIP_THRESHOLD * 3:
                self._no_change_count = 0
                return True
            return self._no_change_count < self._SKIP_THRESHOLD
    
    def _compute_row_hash(self, row_img: np.ndarray) -> int:
        """Compute a perceptual hash (difference hash) for a single row to detect changes."""
        # Resize to 9x8 for difference hash
        small = cv2.resize(row_img, (9, 8), interpolation=cv2.INTER_AREA)
        # Convert to grayscale
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        # Compute difference hash: compare each pixel to its right neighbor
        # Creates 8x8 = 64-bit hash
        diff = gray[:, 1:] > gray[:, :-1]
        # Convert boolean array to integer hash
        return hash(diff.tobytes())
    
    def _cleanup_signatures(self, t_ms: float):
        # Remove old signatures outside dedup window (use max of dedup and display window)
        window = max(KILL_DEDUP_WINDOW_MS, 5000)
        self.recent_signatures = [s for s in self.recent_signatures if t_ms - s[0] < window]
    
    def _detect(self, t_ms: float, roi_frame: np.ndarray) -> List[Event]:
        """Detect kills from killfeed."""
        # CHECK FOR SCHEDULED HALFTIME: Activate halftime pause when we reach scheduled time
        # This allows us to capture kills in the buffer window after round 12 ends
        if self._halftime_scheduled_ms > 0 and t_ms >= self._halftime_scheduled_ms and not self._in_halftime_break:
            self._in_halftime_break = True
            self._halftime_start_ms = t_ms
            self._halftime_scheduled_ms = 0  # Clear scheduled time
            print(f"[KillfeedDetector] Halftime break NOW ACTIVE at {t_ms:.0f}ms - killfeed detection PAUSED")
        
        # HALFTIME BREAK CHECK: Skip killfeed detection during halftime break
        # This prevents false positives from replay clips and advertisements
        if self._in_halftime_break:
            # Check if we've exceeded the halftime duration (fallback exit)
            if t_ms - self._halftime_start_ms > self._HALFTIME_DURATION_MS:
                print(f"[KillfeedDetector] Halftime break timeout at {t_ms:.0f}ms - resuming detection")
                self._in_halftime_break = False
            else:
                # Still in halftime break - skip all killfeed processing
                return []
        
        # POST-HALFTIME COOLDOWN: Filter replay highlights that appear right after halftime
        if self._halftime_end_ms > 0:
            time_since_halftime_end = t_ms - self._halftime_end_ms
            if time_since_halftime_end < self._POST_HALFTIME_COOLDOWN_MS:
                return []  # Still in cooldown - reject potential replay kills
        
        self._cleanup_signatures(t_ms)
        # Skip expensive OCR if killfeed hasn't changed
        if not self._has_significant_change(roi_frame):
            return []
        events = []
        h, w = roi_frame.shape[:2]
        # Segment rows using fixed positions for consistent extraction
        rows = self._segment_rows_fixed(roi_frame)
        
        KILLFEED_DISPLAY_WINDOW_MS = 5000  # Kills stay visible for 5s
        for actual_row_idx, y_start, y_end, row_img in rows:
            # Per-row change detection - skip OCR if this row hasn't changed
            row_hash = self._compute_row_hash(row_img)
            if actual_row_idx in self._row_hashes and self._row_hashes[actual_row_idx] == row_hash:
                continue  # Row unchanged, skip expensive OCR
            self._row_hashes[actual_row_idx] = row_hash
            
            entry = self._parse_row(row_img)
            if not entry:
                continue
            
            # Get values
            killer_team = entry.get("killer_team", "unknown")
            victim_team = entry.get("victim_team", "unknown")
            killer_name = entry.get("killer_name", "Unknown")
            victim_name = entry.get("victim_name", "Unknown")
            confidence = entry.get("confidence", 0.5)
            
            # Filter 1: Require minimum confidence
            if confidence < 0.7:
                # Debug: log filtered kills
                if killer_name != "Unknown" and victim_name != "Unknown":
                    print(f"[FILTERED-CONF] t={t_ms/1000:.1f}s: {killer_name} -> {victim_name} (conf={confidence:.2f})")
                continue
            
            # Filter 2: Require BOTH killer and victim names (every kill has both in this VOD)
            # Exception: fall damage would only have victim, but that's rare
            if killer_name == "Unknown" or victim_name == "Unknown":
                continue
            
            # Convert colors to team codes for team-aware player matching
            # This is CRITICAL for players who've been on multiple teams (e.g., Crashies: NRG -> FNC)
            killer_team_code = self.get_team_code_from_color(killer_team) if hasattr(self, 'get_team_code_from_color') else None
            victim_team_code = self.get_team_code_from_color(victim_team) if hasattr(self, 'get_team_code_from_color') else None
            
            # Use fuzzy database matching with team-specific player pools
            # match_killfeed_name returns (player_name, extracted_team_code_from_ocr)
            # The extracted_team_code takes PRIORITY over color-based detection
            killer_name_db, killer_ocr_team = (None, None)
            victim_name_db, victim_ocr_team = (None, None)
            
            if self._player_matcher:
                killer_name_db, killer_ocr_team = self._player_matcher.match_killfeed_name(killer_name, killer_team_code)
                victim_name_db, victim_ocr_team = self._player_matcher.match_killfeed_name(victim_name, victim_team_code)
                
                # If OCR detected a team tag (e.g., "FNC" from "FNC crashies"), use that
                # This overrides color-based team detection for players on multiple teams
                if killer_ocr_team:
                    killer_team_code = killer_ocr_team
                    # Also update killer_team color to match (for summary generation)
                    killer_team = self.get_color_from_team_code(killer_ocr_team) if hasattr(self, 'get_color_from_team_code') else killer_team
                if victim_ocr_team:
                    victim_team_code = victim_ocr_team
                    victim_team = self.get_color_from_team_code(victim_ocr_team) if hasattr(self, 'get_color_from_team_code') else victim_team

            killer_name_normalized = killer_name_db if killer_name_db else self._normalize_player_name(killer_name)
            victim_name_normalized = victim_name_db if victim_name_db else self._normalize_player_name(victim_name)

            # Filter 3: Skip if EITHER normalized name is Unknown (need both for valid kill)
            if killer_name_normalized == "Unknown" or victim_name_normalized == "Unknown":
                print(f"[FILTERED-UNK] t={t_ms/1000:.1f}s: {killer_name}->{killer_name_normalized} vs {victim_name}->{victim_name_normalized}")
                continue
            
            # Filter 4: If we have player matcher, require at least one name to match database
            # OR both names to be successfully normalized (not Unknown)
            # This filters out garbage OCR while allowing valid detections
            if self._player_matcher:
                has_db_match = killer_name_db is not None or victim_name_db is not None
                if not has_db_match:
                    # Neither matched database - both must be normalized successfully
                    # (we already checked they're not Unknown above, so this is fine)
                    pass  # Allow if both are valid normalized names

            # Check for duplicates using normalized names
            sig = (t_ms, killer_team, victim_team, killer_name_normalized, victim_name_normalized, actual_row_idx)
            if self._is_duplicate_scroll_aware(t_ms, sig, KILLFEED_DISPLAY_WINDOW_MS):
                continue
            self.recent_signatures.append(sig)
            
            # Determine round number for display - if within 5s buffer of last transition,
            # the kill belongs to the ending round, not the new round
            BUFFER_MS = 5000
            if self._last_transition_ms > 0 and (t_ms - self._last_transition_ms) < BUFFER_MS:
                # Within buffer window - kill belongs to the round that just ended
                display_round = self._last_transition_round
            else:
                # Past buffer - kill belongs to current round
                display_round = self._current_round_number
            
            # Log the accepted kill - just player names, no team prefix (OCR may include it)
            print(f"[KILL] t={t_ms/1000:.1f}s R{display_round} ROW {actual_row_idx+1}: {killer_name_normalized} killed {victim_name_normalized}")

            # Track this victim's death for per-round deduplication
            victim_key = victim_name_normalized.lower().strip() if victim_name_normalized != "Unknown" else None
            if victim_key:
                self._victim_last_death[victim_key] = (t_ms, killer_name_normalized)

            # Create kill event with normalized names
            events.append(Event(
                t_ms=t_ms,
                type="KILL_EVENT",
                roi=self.roi_name,
                payload={
                    "killer_name": killer_name_normalized,
                    "killer_team": killer_team,
                    "victim_name": victim_name_normalized,
                    "victim_team": victim_team,
                    "weapon": entry.get("weapon", "unknown"),
                    "is_headshot": entry.get("is_headshot", False),
                },
                confidence=confidence
            ))

            # Also emit death event with normalized name
            events.append(Event(
                t_ms=t_ms,
                type="DEATH_EVENT",
                roi=self.roi_name,
                payload={
                    "player_name": victim_name_normalized,
                    "player_team": entry.get("victim_team", "unknown"),
                    "killed_by": killer_name_normalized,
                },
                confidence=entry.get("confidence", 0.5)
            ))
        
        return events

    def _is_duplicate_scroll_aware(self, t_ms: float, sig: tuple, display_window_ms: int) -> bool:
        """
        Enhanced deduplication: deduplicate kills across all rows and time, accounting for scrolling.
        
        KEY INSIGHT: In VALORANT, a player can only die ONCE per round (~90 seconds max).
        So if we see the same VICTIM die again within a short window, it's definitely a duplicate.
        
        Dedup tiers (in priority order):
        1. VICTIM-FOCUSED: Same victim within 4s = duplicate (strongest signal, tightened)
        2. FULL MATCH: Same killer+victim within display window = duplicate
        3. PARTIAL MATCH: Either name very similar within tight window
        4. SWAP CHECK: Names swapped (OCR error) = duplicate
        """
        _, killer_team, victim_team, killer_name, victim_name, row_idx = sig
        
        # Apply OCR correction BEFORE comparison - this normalizes common OCR errors
        killer_corrected = self._correct_ocr_name(killer_name)
        victim_corrected = self._correct_ocr_name(victim_name)
        
        # Normalize names for comparison (strip team prefixes)
        killer_base = self._strip_team_prefix(killer_corrected)
        victim_base = self._strip_team_prefix(victim_corrected)
        
        for (sig_t, sig_kt, sig_vt, sig_kn, sig_vn, sig_row) in self.recent_signatures:
            time_diff = t_ms - sig_t
            if time_diff > display_window_ms:
                continue
            
            # Apply OCR correction to stored names too
            sig_killer_corrected = self._correct_ocr_name(sig_kn)
            sig_victim_corrected = self._correct_ocr_name(sig_vn)
            
            # Strip prefixes from stored names
            sig_killer_base = self._strip_team_prefix(sig_killer_corrected)
            sig_victim_base = self._strip_team_prefix(sig_victim_corrected)
            
            # EXACT MATCH after OCR correction = definitely duplicate
            if killer_base.lower() == sig_killer_base.lower() and victim_base.lower() == sig_victim_base.lower():
                return True
            
            # Calculate similarities using corrected base names
            killer_sim = self._name_similarity(killer_base, sig_killer_base)
            victim_sim = self._name_similarity(victim_base, sig_victim_base)
            
            # TIER 1: VICTIM-FOCUSED dedup (strongest - player can only die once per round)
            # Same victim within 3 seconds = DEFINITELY a duplicate (scrolling or repeated detection)
            # Tightened to 3s since we now have OCR correction reducing false matches
            if victim_sim > 0.70 and time_diff < 3000:
                # Don't require team match - team colors are unreliable
                return True
            
            # TIER 1b: Very high victim similarity within longer window (exact name match)
            if victim_sim > 0.90 and time_diff < display_window_ms:
                return True
            
            # TIER 2: Full match - both killer and victim similar
            if killer_sim > 0.70 and victim_sim > 0.70:
                # Don't require strict team match - colors can be misdetected
                return True
            
            # TIER 2b: Moderate similarity on both names within tight window
            if killer_sim > 0.55 and victim_sim > 0.55 and time_diff < 2500:
                return True
            
            # TIER 3: One name very similar + other moderately similar (OCR variation)
            if (killer_sim > 0.85 and victim_sim > 0.45 and time_diff < 2500) or \
               (victim_sim > 0.85 and killer_sim > 0.45 and time_diff < 2500):
                return True
            
            # TIER 4: Check for swapped killer/victim (rare OCR confusion)
            swap_killer_sim = self._name_similarity(killer_base, sig_victim_base)
            swap_victim_sim = self._name_similarity(victim_base, sig_killer_base)
            if swap_killer_sim > 0.70 and swap_victim_sim > 0.70:
                return True
                
        return False
    
    def _correct_ocr_name(self, name: str) -> str:
        """
        Apply OCR error corrections to player names.
        This normalizes common OCR mistakes before deduplication.
        Uses the DB player matcher for fuzzy matching to canonical names.
        """
        if not name or name == "Unknown":
            return name
        
        # Strip team prefix first
        base_name = self._strip_team_prefix(name)
        base_lower = base_name.lower().strip()
        
        # Try DB matcher first for fuzzy matching to canonical names
        if self._player_matcher:
            db_match, _ = self._player_matcher.match_killfeed_name(base_name)
            if db_match:
                # Preserve original team prefix if present
                if name != base_name:
                    prefix = name[:len(name) - len(base_name)]
                    return prefix + db_match
                return db_match
        
        # Fallback: Check for known OCR errors (legacy dictionary)
        if base_lower in OCR_NAME_CORRECTIONS:
            corrected = OCR_NAME_CORRECTIONS[base_lower]
            # Preserve original team prefix if present
            if name != base_name:
                prefix = name[:len(name) - len(base_name)]
                return prefix + corrected
            return corrected
        
        return name
    
    def _strip_team_prefix(self, name: str) -> str:
        """Strip team prefix (e.g., 'NRG ', 'FNC ') from player name for comparison."""
        if not name or name == "Unknown":
            return name
        # Common team prefixes in VALORANT esports
        prefixes = ['NRG ', 'FNC ', 'SEN ', 'C9 ', 'TL ', 'LOUD ', '100T ', 'DRX ', 'PRX ', 'FPX ', 
                    'TH ', 'LEV ', 'EDG ', 'KRU ', 'EG ', 'FURIA ', 'T1 ', 'GEN ', 'ZETA ', 'TS ']
        name_upper = name.upper()
        for prefix in prefixes:
            if name_upper.startswith(prefix):
                return name[len(prefix):].strip()
        return name
    
    def _segment_rows(self, roi_bgr: np.ndarray) -> List[Tuple[int, int, np.ndarray]]:
        """Segment killfeed into individual rows."""
        h, w = roi_bgr.shape[:2]
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        
        # Create combined mask for team colors
        teal_mask = cv2.inRange(hsv, 
                                np.array(TEAM_COLORS["teal"]["lower"]),
                                np.array(TEAM_COLORS["teal"]["upper"]))
        orange_mask = cv2.inRange(hsv,
                                  np.array(TEAM_COLORS["orange"]["lower"]),
                                  np.array(TEAM_COLORS["orange"]["upper"]))
        
        color_mask = cv2.bitwise_or(teal_mask, orange_mask)
        
        # Dilate to connect nearby regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        dilated = cv2.dilate(color_mask, kernel, iterations=1)
        
        # Project horizontally
        h_proj = np.sum(dilated, axis=1).astype(np.float32)
        
        # Find rows with content
        threshold = np.max(h_proj) * 0.15 if np.max(h_proj) > 0 else 0
        
        rows = []
        in_row = False
        row_start = 0
        
        for y in range(h):
            if h_proj[y] > threshold:
                if not in_row:
                    row_start = y
                    in_row = True
            else:
                if in_row:
                    row_height = y - row_start
                    if KILLFEED_ROW_HEIGHT_RANGE[0] <= row_height <= KILLFEED_ROW_HEIGHT_RANGE[1]:
                        rows.append((row_start, y, roi_bgr[row_start:y, :]))
                    in_row = False
        
        return rows[:KILLFEED_MAX_ROWS]
    
    def _segment_rows_fixed(self, roi_bgr: np.ndarray) -> List[Tuple[int, int, int, np.ndarray]]:
        """
        Segment killfeed into individual rows using fixed pixel positions.
        Uses smart processing: always check primary rows (1-5), only check
        extended rows (6-9) when ALL primary rows have content.
        
        Returns: List of (row_index, y_start, y_end, row_img) tuples.
        Row index is 0-based (row 0 = top kill, row 1 = second kill, etc.)
        
        CRITICAL FIX: Extended rows (5+) only valid if ALL lower rows have content.
        This prevents false positives from UI elements outside the killfeed.
        """
        h, w = roi_bgr.shape[:2]
        
        # Fixed row height based on extended rows (full 9 rows)
        row_height = h // KILLFEED_EXTENDED_ROWS
        total_pixels = row_height * w
        
        rows = []
        rows_with_content = set()  # Track which rows have content
        row_color_density = {}  # Track color density for extended row validation
        
        # Thresholds for color density validation
        # Real killfeed entries have solid colored backgrounds (team colors)
        # Both killer and victim names should have substantial team color pixels
        # Primary rows: need both colors present with meaningful amounts
        # Extended rows: need higher density to filter out UI elements
        MIN_COLOR_PIXELS_PRIMARY = 100  # Increased from 20 - need real text backgrounds
        MIN_COLOR_PIXELS_MINORITY = 50  # The minority color needs at least this many pixels
        MIN_COLOR_DENSITY_EXTENDED = 0.01  # At least 1% of row should be team color
        
        # First pass: check ALL rows and determine which have content
        for i in range(KILLFEED_EXTENDED_ROWS):
            y_start = i * row_height
            y_end = min((i + 1) * row_height, h)
            
            # Extract row image
            row_img = roi_bgr[y_start:y_end, :]
            
            # Check for team colors (killfeed indicator)
            hsv = cv2.cvtColor(row_img, cv2.COLOR_BGR2HSV)
            teal_mask = cv2.inRange(hsv, 
                                    np.array(TEAM_COLORS["teal"]["lower"]),
                                    np.array(TEAM_COLORS["teal"]["upper"]))
            orange_mask = cv2.inRange(hsv,
                                      np.array(TEAM_COLORS["orange"]["lower"]),
                                      np.array(TEAM_COLORS["orange"]["upper"]))
            
            teal_pixels = cv2.countNonZero(teal_mask)
            orange_pixels = cv2.countNonZero(orange_mask)
            
            # Calculate color density (fraction of row covered by team colors)
            color_density = (teal_pixels + orange_pixels) / total_pixels
            row_color_density[i] = color_density
            
            # Must have BOTH teal and orange colors for a valid kill entry
            # (killer name = one color, victim name = other color)
            # The majority color (larger text area) should have substantial pixels
            # The minority color should also be meaningful (not just noise)
            majority_pixels = max(teal_pixels, orange_pixels)
            minority_pixels = min(teal_pixels, orange_pixels)
            
            has_sufficient_colors = (
                majority_pixels > MIN_COLOR_PIXELS_PRIMARY and 
                minority_pixels > MIN_COLOR_PIXELS_MINORITY
            )
            
            if has_sufficient_colors:
                rows_with_content.add(i)
        
        # Early exit: if no rows have content, return empty
        if not rows_with_content:
            return []
        
        # EARLY EXIT: If rows 0, 1, AND 2 are all empty, killfeed is likely clear
        # This prevents false positives from isolated UI elements in lower rows
        if 0 not in rows_with_content and 1 not in rows_with_content and 2 not in rows_with_content:
            return []
        
        # Second pass: return rows that pass validation
        # RULE: A row is only valid if ALL lower-numbered rows also have content
        # This prevents false positives from UI elements (like pings, player labels)
        for i in range(KILLFEED_EXTENDED_ROWS):
            if i not in rows_with_content:
                continue
            
            y_start = i * row_height
            y_end = min((i + 1) * row_height, h)
            row_img = roi_bgr[y_start:y_end, :]
            
            # Check if all lower rows have content
            all_lower_have_content = all(j in rows_with_content for j in range(i))
            
            # For primary rows (0-4), allow gaps (kills might be fading)
            if i < KILLFEED_NUM_ROWS:
                rows.append((i, y_start, y_end, row_img))
                continue
            
            # For extended rows (5-8), apply strict validation:
            # 1. ALL lower rows must have content (no gaps)
            # 2. At least 4 primary rows must have content (real multi-kill scenario)
            # 3. Must meet minimum color density threshold (real killfeed has solid backgrounds)
            # 4. Must have detectable text (OCR check for extended rows)
            primary_rows_with_content = sum(1 for j in range(KILLFEED_NUM_ROWS) if j in rows_with_content)
            
            if not all_lower_have_content:
                continue
            if primary_rows_with_content < 4:
                continue
            if row_color_density.get(i, 0) < MIN_COLOR_DENSITY_EXTENDED:
                continue
            
            # Text validation for extended rows - must have readable text
            # Quick OCR check to ensure this is actually a killfeed entry
            try:
                gray = cv2.cvtColor(row_img, cv2.COLOR_BGR2GRAY)
                # Apply contrast enhancement
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
                enhanced = clahe.apply(gray)
                # Quick OCR to check for text presence
                ocr_result = self.ocr.ocr(enhanced, cls=False)
                has_text = False
                if ocr_result and ocr_result[0]:
                    for line in ocr_result[0]:
                        if line and len(line) >= 2:
                            text = line[1][0] if isinstance(line[1], tuple) else str(line[1])
                            conf = line[1][1] if isinstance(line[1], tuple) else 0.5
                            # Need at least 3 chars with decent confidence
                            if len(text) >= 3 and conf > 0.3:
                                has_text = True
                                break
                
                if not has_text:
                    continue
            except Exception:
                # If OCR fails, skip this extended row
                continue
            
            rows.append((i, y_start, y_end, row_img))
        
        return rows

    def _parse_row(self, row_img: np.ndarray) -> Optional[Dict[str, Any]]:
        """Parse a killfeed row to extract kill information."""
        h, w = row_img.shape[:2]
        hsv = cv2.cvtColor(row_img, cv2.COLOR_BGR2HSV)
        
        # Detect team colors
        teal_mask = cv2.inRange(hsv,
                                np.array(TEAM_COLORS["teal"]["lower"]),
                                np.array(TEAM_COLORS["teal"]["upper"]))
        orange_mask = cv2.inRange(hsv,
                                  np.array(TEAM_COLORS["orange"]["lower"]),
                                  np.array(TEAM_COLORS["orange"]["upper"]))
        
        teal_pixels = cv2.countNonZero(teal_mask)
        orange_pixels = cv2.countNonZero(orange_mask)
        
        # Need at least one team color
        if teal_pixels < 20 and orange_pixels < 20:
            return None
        
        # Find colored regions
        teal_regions = self._find_color_regions(teal_mask)
        orange_regions = self._find_color_regions(orange_mask)
        
        all_regions = []
        for x, _, rw, _ in teal_regions:
            all_regions.append({"color": "teal", "x": x, "center": x + rw // 2})
        for x, _, rw, _ in orange_regions:
            all_regions.append({"color": "orange", "x": x, "center": x + rw // 2})
        
        all_regions.sort(key=lambda r: r["x"])
        
        if len(all_regions) < 2:
            return None
        
        # In killfeed: leftmost color = killer team, rightmost = victim team
        killer_team = all_regions[0]["color"]
        victim_team = all_regions[-1]["color"]
        
        # Try OCR for names
        killer_name = "Unknown"
        victim_name = "Unknown"
        
        self._init_ocr()
        if self._ocr_reader:
            try:
                # SPEED OPTIMIZATION: Single-pass OCR with contrast only (fastest)
                if hasattr(self._ocr_reader, 'read_text_multipass'):
                    # Single-pass OCR - contrast only for speed
                    multipass_results = self._ocr_reader.read_text_multipass(
                        row_img, 
                        min_confidence=0.2,
                        strategies=['contrast']  # Single pass for speed
                    )
                    # Convert to standard format (results already scaled by preprocessing)
                    # The preprocessing scales 1.5x now (reduced from 3x for speed)
                    scale = 1.5  # Matches new preprocessing scale
                    results = []
                    for r in multipass_results:
                        # x position needs to be divided by scale since preprocessing enlarged the image
                        results.append((r.bbox, r.text, r.confidence))
                elif hasattr(self._ocr_reader, 'read_text'):
                    # Single-pass OCR engine
                    scale = 2
                    scaled = cv2.resize(row_img, None, fx=scale, fy=scale, 
                                       interpolation=cv2.INTER_LINEAR)
                    ocr_results = self._ocr_reader.read_text(scaled, min_confidence=0.3)
                    results = [(r.bbox, r.text, r.confidence) for r in ocr_results]
                else:
                    # Legacy EasyOCR direct usage
                    scale = 2
                    scaled = cv2.resize(row_img, None, fx=scale, fy=scale, 
                                       interpolation=cv2.INTER_LINEAR)
                    results = self._ocr_reader.readtext(scaled, paragraph=False)
                
                names = []
                for bbox, text, conf in results:
                    if conf > 0.2 and len(text.strip()) >= 2:
                        # Handle both tuple bbox (new) and list bbox (legacy)
                        if isinstance(bbox, tuple) and len(bbox) == 4:
                            x_center = (bbox[0] + bbox[2] / 2) / scale
                        else:
                            x_center = (bbox[0][0] + bbox[2][0]) / 2 / scale
                        raw_name = text.strip()
                        
                        # NOTE: Don't do early fuzzy matching here - let _detect() handle it
                        # with proper team codes to avoid matching players to wrong teams
                        # (e.g., matching 'doma' garbage to a player not in this match)
                        
                        names.append({"name": raw_name, "x": x_center, "conf": conf})
                
                names.sort(key=lambda n: n["x"])
                
                if len(names) >= 2:
                    killer_name = names[0]["name"]
                    victim_name = names[-1]["name"]
                elif len(names) == 1:
                    # Single name - determine position
                    if names[0]["x"] < w / 2:
                        killer_name = names[0]["name"]
                    else:
                        victim_name = names[0]["name"]
                        
            except Exception as e:
                print(f"[OCR] Error processing killfeed row: {e}")
        
        # Use player matcher to determine teams from names (more reliable than color detection)
        # In VALORANT broadcasts:
        # - Left side of HUD ("left" team side) = typically teal colored
        # - Right side of HUD ("right" team side) = typically orange colored
        # This matches the standard broadcast layout
        
        # DEBUG: Log team resolution attempt
        has_matcher = self._player_matcher is not None
        left_code = getattr(self, '_left_team_code', None)
        right_code = getattr(self, '_right_team_code', None)
        
        # Debug - show initial team detection from color
        # print(f"[TEAM DEBUG] Color detected: killer={killer_team}, victim={victim_team}, matcher={has_matcher}, codes=({left_code}, {right_code})")
        
        if self._player_matcher:
            if killer_name != "Unknown":
                killer_team_side = self._player_matcher.get_player_team(killer_name)
                if killer_team_side:
                    old_killer_team = killer_team
                    # Left HUD side = teal, Right HUD side = orange
                    killer_team = "teal" if killer_team_side == "left" else "orange"
                    if old_killer_team != killer_team:
                        print(f"[DEBUG TEAM OVERRIDE] killer '{killer_name}' team_side={killer_team_side} -> color={killer_team} (was {old_killer_team})")
            if victim_name != "Unknown":
                victim_team_side = self._player_matcher.get_player_team(victim_name)
                if victim_team_side:
                    old_victim_team = victim_team
                    # Left HUD side = teal, Right HUD side = orange
                    victim_team = "teal" if victim_team_side == "left" else "orange"
                    if old_victim_team != victim_team:
                        print(f"[DEBUG TEAM OVERRIDE] victim '{victim_name}' team_side={victim_team_side} -> color={victim_team} (was {old_victim_team})")
        
        # NOTE: We store RAW colors (teal/orange as detected) in the event payload.
        # The _get_team_name_from_color() function handles halftime/overtime color swaps
        # when building the summary/timeline. Do NOT swap colors here to avoid double-swapping.
        
        return {
            "killer_name": killer_name,
            "killer_team": killer_team,  # Raw color: teal or orange
            "victim_name": victim_name,
            "victim_team": victim_team,  # Raw color: teal or orange
            "weapon": "unknown",
            "is_headshot": False,
            "confidence": 0.7 if killer_name != "Unknown" and victim_name != "Unknown" else 0.4,
        }
    
    def _find_color_regions(self, mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Find contiguous color regions in a mask."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w > 8 and h > 4:
                regions.append((x, y, w, h))
        return regions
    
    def _is_duplicate(self, t_ms: float, sig: Tuple) -> bool:
        """
        Check if this kill is a duplicate of a recent one.
        
        Uses two-tier duplicate detection:
        1. PER-ROUND: A player can only die ONCE per round in Valorant
           - If we've seen this victim die already this round, it's a duplicate
           - This catches replays and multiple detections of the same killfeed entry
        2. SHORT-TERM: Standard dedup within KILL_DEDUP_WINDOW_MS
           - Handles OCR re-detection of same killfeed row
        """
        _, killer_team, victim_team, killer_name, victim_name = sig
        
        # ==== TIER 1: Per-round victim death tracking ====
        # In Valorant, a player can only die ONCE per round
        # If victim already died this round, it's definitely a duplicate
        victim_key = victim_name.lower().strip() if victim_name and victim_name != "Unknown" else None
        
        if victim_key:
            if victim_key in self._victim_last_death:
                last_death_time, last_killer = self._victim_last_death[victim_key]
                time_since_death = t_ms - last_death_time
                
                # If victim "died" again within round duration, it's a duplicate
                # Rounds last ~100 seconds max, but use 90s window to be safe
                if time_since_death < self.ROUND_DEDUP_WINDOW_MS:
                    # Check if same killer (definitely duplicate) or different killer
                    # Even different killer = duplicate if within same round
                    killer_key = killer_name.lower().strip() if killer_name else ""
                    killer_sim = self._name_similarity(killer_name, last_killer)
                    
                    # If same killer or very similar, definitely duplicate
                    if killer_sim > 0.5:
                        return True
                    
                    # Different killer but very short gap = replay showing kill
                    if time_since_death < 60000:  # 60 seconds - well within a round
                        return True
        
        # ==== TIER 2: Standard short-term dedup ====
        
        for (sig_t, sig_kt, sig_vt, sig_kn, sig_vn) in self.recent_signatures:
            time_diff = t_ms - sig_t
            
            if time_diff > KILL_DEDUP_WINDOW_MS:
                continue
            
            # Calculate name similarities
            killer_sim = self._name_similarity(killer_name, sig_kn)
            victim_sim = self._name_similarity(victim_name, sig_vn)
            
            # CRITICAL: Same victim AND same killer = definitely duplicate
            # A player can only die once per kill event
            # But require BOTH to match, not just victim (player can die multiple times per match)
            if victim_sim > 0.7 and killer_sim > 0.7:
                if time_diff < 3000:  # Within 3 seconds
                    return True
            
            # Same teams and similar names
            if sig_kt == killer_team and sig_vt == victim_team:
                # Very high similarity on both = same kill
                if killer_sim > 0.7 and victim_sim > 0.7:
                    return True
                # Good similarity on both
                elif killer_sim > 0.5 and victim_sim > 0.5:
                    if time_diff < 4000:
                        return True
                # One name is Unknown, the other matches well
                elif (killer_name == "Unknown" and victim_sim > 0.6) or \
                     (victim_name == "Unknown" and killer_sim > 0.6):
                    if time_diff < 3000:
                        return True
            
            # Check for SWAPPED names (killer<->victim confusion)
            killer_as_victim = self._name_similarity(killer_name, sig_vn)
            victim_as_killer = self._name_similarity(victim_name, sig_kn)
            
            if killer_as_victim > 0.6 and victim_as_killer > 0.6:
                if time_diff < 4000:
                    return True
            
            # Exact match on BOTH names within very short time = definitely duplicate
            # But only filter if BOTH names match well (not just one)
            if time_diff < 1500:
                if killer_sim > 0.8 and victim_sim > 0.8:
                    return True
            
            # Check for team swap (same names, different teams)
            if sig_kt != killer_team and sig_vt != victim_team:
                if killer_sim > 0.6 and victim_sim > 0.6:
                    if time_diff < 4000:
                        return True
        
        return False
    
    def _name_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between two names (0.0 to 1.0)."""
        if not name1 or not name2:
            return 0.0
        if name1 == "Unknown" or name2 == "Unknown":
            return 0.0
        
        # Normalize names first
        n1 = self._normalize_player_name(name1).lower().strip()
        n2 = self._normalize_player_name(name2).lower().strip()
        
        if n1 == n2:
            return 1.0
        if n1 in n2 or n2 in n1:
            return 0.85
        
        # Character overlap ratio
        common = sum(1 for c in n1 if c in n2)
        return common / max(len(n1), len(n2), 1)
    
    def _normalize_player_name(self, name: str) -> str:
        """
        Normalize OCR misreads to canonical player names.
        Maps common OCR errors to correct names.
        Returns "Unknown" for obvious OCR noise.
        """
        if not name or name == "Unknown":
            return "Unknown"
        
        name_stripped = name.strip()
        name_lower = name_stripped.lower()
        
        # ===== EARLY GARBAGE FILTERING =====
        # Filter known garbage prefixes that are OCR errors
        garbage_prefixes = ['ndc ', 'nde ', 'nid ', 'nide ', 'noc ', 'noe ', 'iv ', 'tip ']
        for prefix in garbage_prefixes:
            if name_lower.startswith(prefix):
                # Strip the garbage prefix and re-process
                name_stripped = name_stripped[len(prefix):].strip()
                name_lower = name_stripped.lower()
                break
        
        # Filter repetitive garbage patterns from Surya hallucinations
        garbage_patterns = [
            'the state of', 'the second', 'the same of',
            'the party of', 'the property of', 'the person',
            'column 2', 'column two', 'in column',
            'a contractor', 'a real property', 'a security',
            'the reserve', 'the residence', 'control of the',
            'name of persons', 'named in column',
            'math>', '<b>', '</b>', '<u>', '</u>',
            '----', '____', '. . .', '* * *',
            'management', 'services', 'alberta', 'valley',
            'to nac', 'mac - ', 'all alpe', ' a ',
        ]
        for pattern in garbage_patterns:
            if pattern in name_lower:
                return "Unknown"
        
        # Filter text that's too long (player names with prefix are < 20 chars)
        if len(name_stripped) > 25:
            return "Unknown"
        
        # Filter if mostly repetitive words
        words = name_lower.split()
        if len(words) >= 4:
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            if word_counts:
                max_freq = max(word_counts.values())
                if max_freq > len(words) * 0.4:  # >40% same word = garbage
                    return "Unknown"
        
        # Filter out obvious OCR noise
        # - Too short (less than 3 chars after stripping team prefix)
        # - Pure numbers
        # - Single letters or gibberish
        player_part = name_stripped
        for prefix in ['nrg ', 'fnc ', 'fng ', 'nag ', 'npg ', 'fne ']:
            if name_lower.startswith(prefix):
                player_part = name_stripped[4:]
                break
        
        # Reject obvious garbage
        if len(player_part) < 3:
            return "Unknown"
        if player_part.isdigit():
            return "Unknown"
        if not any(c.isalpha() for c in player_part):
            return "Unknown"
        # Reject if too many special characters
        alpha_count = sum(1 for c in player_part if c.isalpha())
        if alpha_count < len(player_part) * 0.5:
            return "Unknown"
        
        # ===== SCALABLE PLAYER MATCHING =====
        # Use the player matcher if available to find canonical names
        # This works for ANY match, not just specific players
        if self._player_matcher:
            # Try matching with the DB player pool (lowered threshold of 0.55)
            db_match, extracted_team = self._player_matcher.match_killfeed_name(name_stripped)
            if db_match:
                # Use extracted team from OCR if available, else try to find from player matcher
                if extracted_team:
                    return f"{extracted_team} {db_match}"
                team_side = self._player_matcher.get_player_team(db_match)
                if team_side == "left" and self._player_matcher._left_team_code:
                    return f"{self._player_matcher._left_team_code} {db_match}"
                elif team_side == "right" and self._player_matcher._right_team_code:
                    return f"{self._player_matcher._right_team_code} {db_match}"
                return db_match
        
        # ===== TEAM PREFIX NORMALIZATION =====
        # If we have a team prefix, normalize it and keep the player part
        # Support common OCR errors in team prefixes
        left_prefixes = ['nrg ', 'nag ', 'npg ', 'nng ']
        right_prefixes = ['fnc ', 'fng ', 'fne ', 'fnf ']
        
        for prefix in left_prefixes:
            if name_lower.startswith(prefix):
                player_part = name_stripped[4:] if len(name_stripped) > 4 else name_stripped
                if len(player_part) >= 3:
                    team_code = self._player_matcher._left_team_code if self._player_matcher else "NRG"
                    return f"{team_code} {player_part}"
                    
        for prefix in right_prefixes:
            if name_lower.startswith(prefix):
                player_part = name_stripped[4:] if len(name_stripped) > 4 else name_stripped
                if len(player_part) >= 3:
                    team_code = self._player_matcher._right_team_code if self._player_matcher else "FNC"
                    return f"{team_code} {player_part}"
        
        # If name has no recognizable team prefix and no DB match, reject as unknown
        # (helps filter out garbage OCR)
        return "Unknown"

    def _names_similar(self, name1: str, name2: str) -> bool:
        """Check if two names are similar (for deduplication)."""
        if not name1 or not name2:
            return False
        n1 = name1.lower().strip()
        n2 = name2.lower().strip()
        if n1 == n2:
            return True
        if n1 in n2 or n2 in n1:
            return True
        # Simple character overlap
        common = sum(1 for c in n1 if c in n2)
        return common / max(len(n1), len(n2)) > 0.6


# ======================================
# Top HUD Detector (Score Change Detection)
# ======================================
class TopHUDDetector(BaseDetector):
    """
    Detects round transitions by reading the score via OCR and detecting when it changes.
    
    Simplified approach: Direct EasyOCR with high confidence threshold.
    Uses fixed pixel ROIs for score extraction (calibrated for 1920x1080).
    
    Also tracks score visibility for halftime detection:
    - Score visible and stable for 3+ seconds = live match
    - Score not visible or invalid = halftime/replay/ads
    """
    
    # VALORANT score rules
    MAX_REALISTIC_SCORE = 20   # Max score in extreme overtime
    HALFTIME_TOTAL_ROUNDS = 12  # First half ends after 12 rounds
    
    def __init__(self, roi_name: str, target_fps: float):
        super().__init__(roi_name, target_fps)
        self._spike_planted = False
        
        # Track confirmed scores - start at 0-0
        self._confirmed_left_score = 0
        self._confirmed_right_score = 0
        self._last_score_change_ms = 0
        self._round_count = 0
        
        # OCR reader (lazy loaded)
        self._score_ocr_reader = None
        
        # Debounce: minimum 5 seconds between round transitions
        self._ROUND_DEBOUNCE_MS = 5000
        
        # Score visibility tracking for halftime detection
        self._last_valid_score_ms = 0  # Last time we saw a valid score
        self._consecutive_invalid_frames = 0  # How many frames without valid score
        self._score_stability_start_ms = 0  # When current stable score started
        self._SCORE_STABLE_THRESHOLD_MS = 3000  # Score must be stable for 3s to confirm live
        self._in_halftime = False
        self._halftime_listeners = []  # Callbacks for halftime state changes
        self._ROUND_DEBOUNCE_MS = 5000
    
    def add_halftime_listener(self, callback):
        """Add a callback to be notified of halftime state changes.
        
        Callback signature: callback(in_halftime: bool, timestamp_ms: float)
        """
        self._halftime_listeners.append(callback)
    
    def is_in_halftime(self) -> bool:
        """Check if we're currently in halftime break."""
        return self._in_halftime
    
    def _get_score_ocr_reader(self):
        """Get or initialize EasyOCR reader for score detection."""
        if self._score_ocr_reader is None:
            try:
                import easyocr
                import os
                use_gpu = os.environ.get('USE_GPU', 'false').lower() == 'true'
                self._score_ocr_reader = easyocr.Reader(['en'], gpu=use_gpu, verbose=False)
                print(f"[Score OCR] Initialized EasyOCR with GPU={use_gpu}", flush=True)
            except Exception as e:
                print(f"[Score OCR] EasyOCR unavailable: {e}", flush=True)
                self._score_ocr_reader = None
        return self._score_ocr_reader
    
    def _extract_score(self, score_roi: np.ndarray) -> Tuple[int, float]:
        """
        Extract a score number (0-20) from a score ROI using EasyOCR.
        Returns (score, confidence) or (-1, 0.0) if unable to read.
        Uses multiple preprocessing approaches for robustness.
        """
        try:
            if score_roi is None or score_roi.size == 0:
                return -1, 0.0
            
            ocr = self._get_score_ocr_reader()
            if ocr is None:
                return -1, 0.0
            
            h, w = score_roi.shape[:2]
            
            # Try multiple preprocessing approaches
            candidates = []
            
            # Method 1: Scale up 3x (small digits need scaling)
            scaled = cv2.resize(score_roi, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
            results = ocr.readtext(scaled, allowlist='0123456789')
            if results:
                candidates.append((results[0][1], results[0][2], 'scaled'))
            
            # Method 2: Grayscale + threshold for white text
            gray = cv2.cvtColor(score_roi, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
            thresh_scaled = cv2.resize(thresh, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
            results = ocr.readtext(thresh_scaled, allowlist='0123456789')
            if results:
                candidates.append((results[0][1], results[0][2], 'thresh'))
            
            # Method 3: CLAHE contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(2, 2))
            enhanced = clahe.apply(gray)
            enhanced_scaled = cv2.resize(enhanced, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
            results = ocr.readtext(enhanced_scaled, allowlist='0123456789')
            if results:
                candidates.append((results[0][1], results[0][2], 'clahe'))
            
            # Pick best candidate by confidence
            best_score = -1
            best_conf = 0.0
            for text, conf, method in candidates:
                if conf >= 0.4:  # Lower threshold since we validate range
                    try:
                        score = int(text)
                        if 0 <= score <= self.MAX_REALISTIC_SCORE:
                            if conf > best_conf:
                                best_score = score
                                best_conf = conf
                    except ValueError:
                        pass
            
            if best_score >= 0:
                return best_score, best_conf
            
            return -1, 0.0
        except Exception as e:
            return -1, 0.0

    def _detect(self, t_ms: float, roi_frame: np.ndarray) -> List[Event]:
        """
        Detect round transitions by reading scores with direct EasyOCR.
        
        Uses ROI coordinates from settings, converted relative to top_hud region.
        Score ROIs in settings are normalized to full frame, so we convert them
        to be relative to top_hud which starts at (0.335, 0.005) with size (0.330, 0.200).
        """
        events = []
        h, w = roi_frame.shape[:2]
        
        # Get score ROIs from settings (normalized to full frame)
        # top_left_score: (0.417, 0.009, 0.036, 0.042) - full frame coords
        # top_right_score: (0.555, 0.009, 0.036, 0.042) - full frame coords
        # top_hud: (0.335, 0.005, 0.330, 0.200) - the region we receive as roi_frame
        
        # Convert full-frame normalized coords to top_hud relative coords
        top_hud_x, top_hud_y = 0.335, 0.005
        top_hud_w, top_hud_h = 0.330, 0.200
        
        # Left score: (0.417, 0.009, 0.036, 0.042)
        left_norm_x = (0.417 - top_hud_x) / top_hud_w
        left_norm_y = (0.009 - top_hud_y) / top_hud_h
        left_norm_w = 0.036 / top_hud_w
        left_norm_h = 0.042 / top_hud_h
        
        # Right score: (0.555, 0.009, 0.036, 0.042)
        right_norm_x = (0.555 - top_hud_x) / top_hud_w
        right_norm_y = (0.009 - top_hud_y) / top_hud_h
        right_norm_w = 0.036 / top_hud_w
        right_norm_h = 0.042 / top_hud_h
        
        # Convert to pixel coordinates
        left_x = int(left_norm_x * w)
        left_y = int(left_norm_y * h)
        score_w = int(left_norm_w * w)
        score_h = int(left_norm_h * h)
        
        right_x = int(right_norm_x * w)
        right_y = int(right_norm_y * h)
        
        # Ensure minimum dimensions
        score_w = max(score_w, 40)
        score_h = max(score_h, 30)
        
        # Clamp to valid bounds
        left_x = max(0, min(left_x, w - score_w))
        right_x = max(0, min(right_x, w - score_w))
        left_y = max(0, min(left_y, h - score_h))
        right_y = max(0, min(right_y, h - score_h))
        
        # Extract ROIs
        left_roi = roi_frame[left_y:left_y+score_h, left_x:left_x+score_w]
        right_roi = roi_frame[right_y:right_y+score_h, right_x:right_x+score_w]
        
        # Read scores with EasyOCR
        left_score, left_conf = self._extract_score(left_roi)
        right_score, right_conf = self._extract_score(right_roi)
        
        # Track score visibility for halftime detection
        score_visible = left_score >= 0 and right_score >= 0 and left_conf >= 0.5 and right_conf >= 0.5
        
        if score_visible:
            self._consecutive_invalid_frames = 0
            self._last_valid_score_ms = t_ms
            
            # Check if we're at halftime (total rounds = 12)
            current_total = left_score + right_score
            if current_total == self.HALFTIME_TOTAL_ROUNDS and not self._in_halftime:
                # We just completed first half - wait for stable second half score
                pass
            
            # Check for exit from halftime: 
            # IMPORTANT: Only end halftime if the score is VALID
            # At halftime, score was X-Y where X+Y=12. Valid post-halftime scores are:
            # 1. Same score X-Y (round 13 starting/in progress)
            # 2. X-(Y+1) or (X+1)-Y (round 13 just ended)
            # Invalid: swapped scores (Y-X), advertisement scores, etc.
            if self._in_halftime:
                halftime_min_duration_ms = 30000  # Halftime is at least 30 seconds
                time_in_halftime = t_ms - self._halftime_start_ms if hasattr(self, '_halftime_start_ms') else 0
                
                # Get the pre-halftime score for validation
                pre_halftime_left = self._confirmed_left_score
                pre_halftime_right = self._confirmed_right_score
                
                # Check if detected score is valid continuation from pre-halftime
                def is_valid_post_halftime_score(left, right, pre_left, pre_right):
                    """Validate that score is a valid continuation from halftime."""
                    # Score must be >= pre-halftime scores (can only gain points, not lose)
                    if left < pre_left or right < pre_right:
                        return False
                    # Total rounds can only increase by at most a few (not jump by many)
                    total_increase = (left + right) - (pre_left + pre_right)
                    if total_increase > 5:  # Allow some rounds to pass during halftime
                        return False
                    return True
                
                is_valid_score = is_valid_post_halftime_score(left_score, right_score, pre_halftime_left, pre_halftime_right)
                
                should_end_halftime = False
                if is_valid_score and time_in_halftime > halftime_min_duration_ms:
                    if current_total > self.HALFTIME_TOTAL_ROUNDS:
                        # Valid score change detected (round 13 ended)
                        should_end_halftime = True
                    elif current_total == self.HALFTIME_TOTAL_ROUNDS:
                        # Same halftime score visible again - round 13 is starting/in progress
                        should_end_halftime = True
                elif not is_valid_score:
                    # Invalid score detected during halftime - reset stability tracking
                    self._score_stability_start_ms = 0
                    if left_score != right_score:  # Don't spam logs for 0-0 type reads
                        print(f"[TopHUD] Invalid score {left_score}-{right_score} during halftime (expected continuation from {pre_halftime_left}-{pre_halftime_right}) at t={t_ms/1000:.1f}s")
                
                if should_end_halftime:
                    # Score visibility indicates live match resumed
                    if self._score_stability_start_ms == 0:
                        self._score_stability_start_ms = t_ms
                    elif t_ms - self._score_stability_start_ms >= self._SCORE_STABLE_THRESHOLD_MS:
                        # Score has been stable for 3+ seconds - halftime is over
                        self._in_halftime = False
                        print(f"[TopHUD] Halftime ended - stable score {left_score}-{right_score} detected at t={t_ms/1000:.1f}s")
                        # Notify listeners
                        for callback in self._halftime_listeners:
                            callback(False, t_ms)
        else:
            self._consecutive_invalid_frames += 1
            self._score_stability_start_ms = 0  # Reset stability tracking
            
            # If score not visible for too long after halftime score, we're in halftime break
            confirmed_total = self._confirmed_left_score + self._confirmed_right_score
            if confirmed_total == self.HALFTIME_TOTAL_ROUNDS and not self._in_halftime:
                # Score became unreadable after reaching halftime - enter halftime mode
                if self._consecutive_invalid_frames >= 5:  # ~2.5s at 2 FPS
                    self._in_halftime = True
                    self._halftime_start_ms = t_ms  # Track when halftime started
                    print(f"[TopHUD] Halftime started - score {self._confirmed_left_score}-{self._confirmed_right_score} no longer visible at t={t_ms/1000:.1f}s")
                    # Notify listeners
                    for callback in self._halftime_listeners:
                        callback(True, t_ms)
        
        # Only process if both scores are valid and confident
        if left_score >= 0 and right_score >= 0 and left_conf >= 0.5 and right_conf >= 0.5:
            # Check if score changed
            if left_score != self._confirmed_left_score or right_score != self._confirmed_right_score:
                # Validate the change makes sense
                total_old = self._confirmed_left_score + self._confirmed_right_score
                total_new = left_score + right_score
                
                # VALIDATION 1: Total score should only increase by exactly 1 (one round at a time)
                # This prevents halftime confusion where OCR might read swapped values
                rounds_added = total_new - total_old
                if rounds_added != 1:
                    # Skip invalid transitions (halftime visual glitch, OCR errors)
                    # Could be 0 (no change), negative (wrong), or >1 (skipped rounds)
                    pass
                else:
                    # VALIDATION 2: Individual scores should never DECREASE
                    # Each team's score can only stay same or +1
                    left_change = left_score - self._confirmed_left_score
                    right_change = right_score - self._confirmed_right_score
                    
                    # Valid transitions: (0,1) or (1,0) - one team wins the round
                    valid_transition = (
                        (left_change == 0 and right_change == 1) or
                        (left_change == 1 and right_change == 0)
                    )
                    
                    if valid_transition:
                        time_since_last = t_ms - self._last_score_change_ms
                        
                        # Debounce check
                        if time_since_last > self._ROUND_DEBOUNCE_MS or self._last_score_change_ms == 0:
                            print(f"[ROUND] Score: {self._confirmed_left_score}-{self._confirmed_right_score} -> {left_score}-{right_score} at t={t_ms/1000:.1f}s (conf: L={left_conf:.2f}, R={right_conf:.2f})", flush=True)
                            
                            # Emit round transition event
                            self._round_count += 1
                            events.append(Event(
                                t_ms=t_ms,
                                type="ROUND_TRANSITION",
                                roi=self.roi_name,
                                payload={
                                    "round_number": self._round_count,
                                    "left_score": left_score,
                                    "right_score": right_score,
                                }
                            ))
                            
                            self._confirmed_left_score = left_score
                            self._confirmed_right_score = right_score
                            self._last_score_change_ms = t_ms
        
        # Detect spike status (look for red/orange danger colors)
        hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
        danger_mask = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([20, 255, 255]))
        danger_pixels = cv2.countNonZero(danger_mask)
        
        spike_planted = danger_pixels > roi_frame.shape[0] * roi_frame.shape[1] * 0.01
        
        if spike_planted != self._spike_planted:
            if spike_planted:
                events.append(Event(
                    t_ms=t_ms,
                    type="SPIKE_PLANTED",
                    roi=self.roi_name,
                    payload={}
                ))
            self._spike_planted = spike_planted
        
        return events


# ======================================
# Minimap Detector
# ======================================
class MinimapDetector(BaseDetector):
    """Tracks player positions on the minimap."""
    
    def __init__(self, roi_name: str, target_fps: float):
        super().__init__(roi_name, target_fps)
        self._prev_player_count = 0
    
    def _detect(self, t_ms: float, roi_frame: np.ndarray) -> List[Event]:
        """Detect player positions on minimap."""
        events = []
        
        hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
        h, w = roi_frame.shape[:2]
        
        # Detect player blips (colored dots)
        # Green (teammates), Blue (ally observed), Red (enemies)
        
        green_mask = cv2.inRange(hsv, np.array([35, 100, 100]), np.array([85, 255, 255]))
        blue_mask = cv2.inRange(hsv, np.array([90, 100, 100]), np.array([130, 255, 255]))
        red_mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        # Count blips
        positions = []
        
        for color_name, mask in [("green", green_mask), ("blue", blue_mask), ("red", red_mask)]:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if 20 < area < 500:  # Filter by size
                    M = cv2.moments(c)
                    if M["m00"] > 0:
                        cx = M["m10"] / M["m00"]
                        cy = M["m01"] / M["m00"]
                        positions.append({
                            "x": cx / w,
                            "y": cy / h,
                            "color": color_name
                        })
        
        player_count = len(positions)
        
        # Emit event if player count changed significantly
        if abs(player_count - self._prev_player_count) >= 1:
            events.append(Event(
                t_ms=t_ms,
                type="MINIMAP_PLAYER_CHANGE",
                roi=self.roi_name,
                payload={
                    "prev_count": self._prev_player_count,
                    "curr_count": player_count,
                    "positions": positions
                }
            ))
        
        self._prev_player_count = player_count
        
        return events
