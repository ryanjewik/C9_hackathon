"""
Map Detection Service

Detects the current map being played from the top-left corner of the broadcast.
The map indicator shows "CURRENT: <MAP_NAME>" in the series scoreboard area.

Valid maps (case-insensitive matching):
    abyss, bind, split, icebox, ascent, fracture, pearl, sunset, lotus, haven, corrode, breeze
"""

import re
import cv2
import numpy as np
from typing import Optional, List, Tuple
from difflib import SequenceMatcher


# Valid VALORANT map pool
VALID_MAPS = [
    "abyss",
    "bind",
    "split",
    "icebox",
    "ascent",
    "fracture",
    "pearl",
    "sunset",
    "lotus",
    "haven",
    "corrode",
    "breeze",
]

# ROI for map indicator region in top-left corner (normalized coords: x, y, w, h)
# This region captures the series scoreboard where "CURRENT: <MAP>" appears
# Shows: "LOTUS 13-6 | CURRENT: ABYSS | NEXT: ASCENT"
# Must match ROI_CONFIG["map_indicator"] in config/settings.py
MAP_INDICATOR_ROI = (0.0, 0.0, 0.32, 0.025)


class MapDetector:
    """
    Detects the current map from broadcast frames.
    
    Looks for "CURRENT:" text followed by a map name in the top-left
    corner of the screen where the series scoreboard is displayed.
    """
    
    def __init__(self):
        self._ocr_engine = None
        self._ocr_initialized = False
    
    def _init_ocr(self):
        """Lazily initialize OCR engine."""
        if self._ocr_initialized:
            return
        
        try:
            from vod_processor.app.services.ocr.ocr_engine import get_ocr_engine
            self._ocr_engine = get_ocr_engine(use_gpu=True)
            print(f"[MapDetector] OCR engine initialized ({self._ocr_engine.backend})")
        except Exception as e:
            print(f"[MapDetector] OCR init failed: {e}")
            self._ocr_engine = None
        
        self._ocr_initialized = True
    
    def _preprocess_roi(self, roi: np.ndarray, scale_factor: float = 4.0) -> np.ndarray:
        """
        Preprocess the ROI for better OCR readability.
        
        Args:
            roi: BGR image crop
            scale_factor: How much to upscale (default 4x for small text)
            
        Returns:
            Preprocessed image (still BGR for OCR)
        """
        # Upscale for better text recognition - 27px height becomes 108px
        h, w = roi.shape[:2]
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Keep color image - OCR often works better with it
        # Just apply slight sharpening
        kernel = np.array([[0, -1, 0],
                          [-1,  5, -1],
                          [0, -1, 0]])
        roi = cv2.filter2D(roi, -1, kernel)
        
        return roi
    
    def detect_map(self, frame: np.ndarray) -> Optional[str]:
        """
        Detect the current map from a video frame.
        
        Args:
            frame: BGR image (full frame)
            
        Returns:
            Detected map name (lowercase) or None if not detected
        """
        self._init_ocr()
        if self._ocr_engine is None:
            return None
        
        # Extract the map indicator region
        h, w = frame.shape[:2]
        x1 = int(MAP_INDICATOR_ROI[0] * w)
        y1 = int(MAP_INDICATOR_ROI[1] * h)
        x2 = int((MAP_INDICATOR_ROI[0] + MAP_INDICATOR_ROI[2]) * w)
        y2 = int((MAP_INDICATOR_ROI[1] + MAP_INDICATOR_ROI[3]) * h)
        
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            print(f"[MapDetector] Empty ROI: {x1},{y1} to {x2},{y2}")
            return None
        
        # Preprocess ROI for better OCR readability
        original_shape = roi.shape
        roi = self._preprocess_roi(roi)
        print(f"[MapDetector] ROI: {original_shape} -> {roi.shape} after preprocessing")
        
        # Run OCR on the region
        try:
            results = self._ocr_engine.read_text(roi, min_confidence=0.3)
            
            # Debug: show raw results
            print(f"[MapDetector] OCR returned {len(results)} results")
            for r in results:
                print(f"[MapDetector]   - '{r.text}' (conf: {r.confidence:.2f})")
            
            # Combine all detected text
            full_text = " ".join([r.text for r in results]).upper()
            print(f"[MapDetector] OCR text: {full_text}")
            
            # Look for "CURRENT" pattern - handle various OCR interpretations
            # Patterns: "CURRENT: LOTUS", "CURRENT LOTUS", "CURRENT:LOTUS"
            current_patterns = [
                r'CURRENT[:\s]+(\w+)',           # CURRENT: MAP or CURRENT MAP
                r'CURR[E3]NT[:\s]+(\w+)',        # OCR might mistake E for 3
                r'[CG]URRENT[:\s]+(\w+)',        # OCR might mistake C for G
            ]
            
            for pattern in current_patterns:
                current_match = re.search(pattern, full_text)
                if current_match:
                    potential_map = current_match.group(1).lower()
                    matched = self._match_map_name(potential_map)
                    if matched:
                        print(f"[MapDetector] Found CURRENT pattern, map: {matched}")
                        return matched
            
            # Fallback: look for any map name in the text with "CURRENT" nearby
            if "CURRENT" in full_text or "CURR" in full_text:
                for map_name in VALID_MAPS:
                    if map_name.upper() in full_text:
                        print(f"[MapDetector] Found map near CURRENT: {map_name}")
                        return map_name
            
            # Last resort: fuzzy match each word against valid maps
            words = re.findall(r'\b\w+\b', full_text)
            for word in words:
                matched = self._match_map_name(word.lower())
                if matched:
                    print(f"[MapDetector] Fuzzy matched word '{word}' to map: {matched}")
                    return matched
                    
        except Exception as e:
            print(f"[MapDetector] OCR error: {e}")
        
        return None
    
    def detect_map_from_frames(
        self, 
        frames: List[np.ndarray],
        min_consensus: int = 2
    ) -> Optional[str]:
        """
        Detect map from multiple frames using voting/consensus.
        
        Args:
            frames: List of BGR frames to analyze
            min_consensus: Minimum number of frames that must agree
            
        Returns:
            Most frequently detected map name, or None
        """
        detections = {}
        
        for frame in frames:
            detected = self.detect_map(frame)
            if detected:
                detections[detected] = detections.get(detected, 0) + 1
        
        if not detections:
            return None
        
        # Get the most common detection
        best_map = max(detections.keys(), key=lambda k: detections[k])
        
        if detections[best_map] >= min_consensus:
            return best_map
        
        # If no consensus but we have a single strong detection
        if len(detections) == 1 and detections[best_map] >= 1:
            return best_map
        
        return None
    
    def _match_map_name(self, text: str) -> Optional[str]:
        """
        Match input text to a valid map name using fuzzy matching.
        
        Args:
            text: Text to match (already lowercase)
            
        Returns:
            Matched map name or None
        """
        if not text or len(text) < 3:
            return None
        
        text = text.lower().strip()
        
        # Exact match
        if text in VALID_MAPS:
            return text
        
        # Check if text contains a map name
        for map_name in VALID_MAPS:
            if map_name in text or text in map_name:
                return map_name
        
        # Fuzzy match with threshold
        best_match = None
        best_score = 0.0
        
        for map_name in VALID_MAPS:
            score = SequenceMatcher(None, text, map_name).ratio()
            if score > best_score and score >= 0.7:
                best_score = score
                best_match = map_name
        
        return best_match


# Singleton instance
_map_detector: Optional[MapDetector] = None


def get_map_detector() -> MapDetector:
    """Get the global MapDetector instance."""
    global _map_detector
    if _map_detector is None:
        _map_detector = MapDetector()
    return _map_detector
