"""
Frame Sampler - Extracts frames from VODs at configurable rates.

Implementation based on architecture.md:
- Sample at 10-15 FPS for HUD parsing
- Maintain rolling buffer for retroactive sampling
- Generator-style API for memory efficiency
"""

import cv2
import numpy as np
from typing import Generator, Optional, Tuple, List
from dataclasses import dataclass
from collections import deque


@dataclass
class Frame:
    """A single video frame with metadata."""
    frame_id: int
    timestamp: float  # Seconds from VOD start
    image: np.ndarray
    
    @property
    def timestamp_ms(self) -> float:
        """Timestamp in milliseconds."""
        return self.timestamp * 1000


class FrameSampler:
    """
    Samples frames from a video at configurable rates.
    
    Features:
    - Base sampling at 10-15 FPS for HUD parsing
    - Rolling buffer for retroactive high-FPS sampling around events
    - Generator-style API to minimize memory usage
    - Adaptive sampling based on detected events
    """
    
    def __init__(
        self,
        video_path: str,
        base_fps: float = 12.0,
        buffer_size: int = 60,  # ~5 seconds at 12fps
    ):
        """
        Initialize the frame sampler.
        
        Args:
            video_path: Path to the video file
            base_fps: Target sampling rate (frames per second)
            buffer_size: Number of frames to keep in rolling buffer
        """
        self.video_path = video_path
        self.base_fps = base_fps
        self.buffer_size = buffer_size
        
        # Video properties (populated on open)
        self.video_fps: float = 0
        self.total_frames: int = 0
        self.duration: float = 0
        self.width: int = 0
        self.height: int = 0
        
        # State
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_buffer: deque = deque(maxlen=buffer_size)
        self._current_frame_idx: int = 0
        self._sample_interval: int = 1
    
    def open(self) -> bool:
        """
        Open the video file and read properties.
        
        Returns:
            True if successful, False otherwise
        """
        self._cap = cv2.VideoCapture(self.video_path)
        
        if not self._cap.isOpened():
            print(f"Error: Could not open video: {self.video_path}")
            return False
        
        self.video_fps = self._cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.total_frames / self.video_fps if self.video_fps > 0 else 0
        
        # Calculate sample interval
        self._sample_interval = max(1, int(self.video_fps / self.base_fps))
        
        print(f"Video opened: {self.width}x{self.height} @ {self.video_fps:.2f}fps")
        print(f"Duration: {self.duration:.2f}s, Total frames: {self.total_frames}")
        print(f"Sampling every {self._sample_interval} frames (~{self.video_fps/self._sample_interval:.1f} effective fps)")
        
        return True
    
    def close(self):
        """Close the video file."""
        if self._cap:
            self._cap.release()
            self._cap = None
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def frame_stream(self, start_time: float = 0) -> Generator[Frame, None, None]:
        """
        Generate frames at the configured sample rate.
        
        Args:
            start_time: Start time in seconds (default: 0)
            
        Yields:
            Frame objects at the sample rate
        """
        if not self._cap or not self._cap.isOpened():
            if not self.open():
                return
        
        # Seek to start position
        if start_time > 0:
            start_frame = int(start_time * self.video_fps)
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            self._current_frame_idx = start_frame
        else:
            self._current_frame_idx = 0
        
        while True:
            ret, image = self._cap.read()
            if not ret:
                break
            
            # Only yield at sample interval
            if self._current_frame_idx % self._sample_interval == 0:
                timestamp = self._current_frame_idx / self.video_fps
                
                frame = Frame(
                    frame_id=self._current_frame_idx,
                    timestamp=timestamp,
                    image=image,
                )
                
                # Add to rolling buffer
                self._frame_buffer.append(frame)
                
                yield frame
            
            self._current_frame_idx += 1
    
    def get_buffered_frames(self) -> List[Frame]:
        """
        Get all frames currently in the rolling buffer.
        Useful for retroactive analysis around detected events.
        
        Returns:
            List of Frame objects in chronological order
        """
        return list(self._frame_buffer)
    
    def get_frames_around_event(
        self,
        event_timestamp: float,
        before_seconds: float = 1.0,
        after_seconds: float = 1.0,
        high_fps: bool = True,
    ) -> List[Frame]:
        """
        Get frames around a specific timestamp for high-FPS analysis.
        
        For killfeed events, we want higher FPS sampling to capture
        the exact moment of the kill.
        
        Args:
            event_timestamp: Event time in seconds
            before_seconds: How many seconds before the event
            after_seconds: How many seconds after the event
            high_fps: If True, sample at video's native FPS
            
        Returns:
            List of Frame objects around the event
        """
        if not self._cap or not self._cap.isOpened():
            return []
        
        # Calculate frame range
        start_frame = max(0, int((event_timestamp - before_seconds) * self.video_fps))
        end_frame = min(self.total_frames, int((event_timestamp + after_seconds) * self.video_fps))
        
        # Sample interval for this extraction
        interval = 1 if high_fps else self._sample_interval
        
        # Save current position
        current_pos = self._cap.get(cv2.CAP_PROP_POS_FRAMES)
        
        frames = []
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for frame_idx in range(start_frame, end_frame, interval):
            ret, image = self._cap.read()
            if not ret:
                break
            
            frames.append(Frame(
                frame_id=frame_idx,
                timestamp=frame_idx / self.video_fps,
                image=image,
            ))
            
            # Skip frames if not reading every frame
            if interval > 1:
                for _ in range(interval - 1):
                    self._cap.read()
        
        # Restore position
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
        
        return frames
    
    def seek_to(self, timestamp: float) -> bool:
        """
        Seek to a specific timestamp.
        
        Args:
            timestamp: Time in seconds
            
        Returns:
            True if successful
        """
        if not self._cap:
            return False
        
        frame_idx = int(timestamp * self.video_fps)
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        self._current_frame_idx = frame_idx
        return True
    
    def get_frame_at(self, timestamp: float) -> Optional[Frame]:
        """
        Get a single frame at a specific timestamp.
        
        Args:
            timestamp: Time in seconds
            
        Returns:
            Frame object or None
        """
        if not self._cap:
            return None
        
        # Save current position
        current_pos = self._cap.get(cv2.CAP_PROP_POS_FRAMES)
        
        frame_idx = int(timestamp * self.video_fps)
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        ret, image = self._cap.read()
        
        # Restore position
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
        
        if not ret:
            return None
        
        return Frame(
            frame_id=frame_idx,
            timestamp=timestamp,
            image=image,
        )
    
    @property
    def progress(self) -> float:
        """Current progress as a percentage (0-100)."""
        if self.total_frames == 0:
            return 0
        return (self._current_frame_idx / self.total_frames) * 100
