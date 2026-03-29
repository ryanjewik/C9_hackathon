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


def _build_orange_mask(hsv: np.ndarray) -> np.ndarray:
    """Build a complete orange mask that includes both hue ranges.
    
    Orange/red hues wrap around 0 in HSV:
      Primary:   H in [0, 25]  (standard orange)
      Secondary: H in [160, 180] (red/pink/magenta)
    Both ranges are needed to capture all orange-family colours in the
    broadcast (including self-kill rows that may have shifted hues).
    """
    mask1 = cv2.inRange(hsv,
                        np.array(TEAM_COLORS["orange"]["lower"]),
                        np.array(TEAM_COLORS["orange"]["upper"]))
    # Include secondary range if defined
    if "lower2" in TEAM_COLORS["orange"] and "upper2" in TEAM_COLORS["orange"]:
        mask2 = cv2.inRange(hsv,
                            np.array(TEAM_COLORS["orange"]["lower2"]),
                            np.array(TEAM_COLORS["orange"]["upper2"]))
        mask1 = cv2.bitwise_or(mask1, mask2)
    return mask1


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
        self._detected_map = None  # Detected map name from broadcast (e.g., "ABYSS")
    
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
        strict_roster: bool = False,
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
            strict_roster: If True, use ONLY the provided player pools for matching
                          (skip DB historical roster loading). Best when exact 5v5 rosters known.
            
        Returns:
            Dictionary with processing results
        """
        from vod_processor.app.services.io.job_manager import JobManager
        from vod_processor.app.services.db.db_player_matcher import DatabasePlayerMatcher
        
        # Store team codes for use in team detection
        self._left_team_code = left_team
        self._right_team_code = right_team
        self._map_name = map_name
        self._strict_roster = strict_roster
        
        # Store player pools for OCR validation
        self._left_player_pool = left_player_pool
        self._right_player_pool = right_player_pool
        
        print(f"[{job_id}] Team codes: left={left_team}, right={right_team}, map={map_name}")
        if strict_roster:
            print(f"[{job_id}] STRICT ROSTER MODE — matching only against provided players")
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
                left_candidates = []
                right_candidates = []
                try:
                    detected_left, detected_right, left_candidates, right_candidates = \
                        self._detect_team_tags_from_hud(cap, fps)
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
                    from vod_processor.app.services.db.db_player_matcher import load_match_players_from_db
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
                
                # --- Player-name validation: if detected team's player pool
                #     doesn't overlap with the names from the HUD, try the next
                #     best candidate tag (handles TL-vs-IL style ambiguity). ---
                self._validate_team_via_players(
                    job_id, cap, fps, left_candidates, right_candidates
                )
                left_player_pool = self._left_player_pool
                right_player_pool = self._right_player_pool
                
                # Reset video position after team detection
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Strict roster: skip HUD extraction, use provided rosters directly
            if strict_roster and left_player_pool and right_player_pool:
                print(f"[{job_id}] Strict roster: skipping HUD player extraction")
                self._player_matcher.set_match_players(
                    left_player_pool, right_player_pool, strict=True
                )
            else:
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
            
            # Enable crop saving on killfeed detector so icons are written to disk
            if killfeed_detector:
                crops_dir = os.path.join(output_dir, "crops")
                os.makedirs(crops_dir, exist_ok=True)
                killfeed_detector._crop_output_dir = crops_dir
                self._killfeed_detector = killfeed_detector  # expose for diagnostic access
                print(f"[{job_id}] Weapon icon crops will be saved to {crops_dir}")
            
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
            skipped_prematch_frames = 0
            match_started = False
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
                    frame, replay_roi, score_bar_roi, left_panels_roi, right_panels_roi, t_ms
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
                    
                    # Notify KillfeedDetector when entering REPLAY mode
                    # This triggers a lookback filter to remove kills just before replay started
                    if frame_state == "REPLAY" and prev_frame_state != "REPLAY":
                        for d in detectors:
                            if hasattr(d, 'on_replay_detected'):
                                d.on_replay_detected(t_ms)
                    
                    prev_frame_state = frame_state
                
                # Skip non-gameplay frames
                # REPLAY: Any overlay text visible (REPLAY/CLUTCH/THRIFTY/FLAWLESS)
                #         means replay/highlight footage — skip killfeed entirely
                if frame_state == "REPLAY":
                    skipped_replay_frames += 1
                    frame_idx += 1
                    continue
                elif frame_state == "TRANSITION":
                    skipped_transition_frames += 1
                    frame_idx += 1
                    continue
                
                # Gate: skip killfeed detection until match starts (0-0 score confirmed)
                if not match_started:
                    if top_hud_detector and top_hud_detector.has_confirmed_zero_zero():
                        match_started = True
                        print(f"[PROC] Match started at t={t_ms/1000:.1f}s — enabling killfeed detection", flush=True)
                    else:
                        # Still run TopHUD detector to look for 0-0 score, but skip killfeed
                        skipped_prematch_frames += 1
                        for detector in detectors:
                            if isinstance(detector, type(top_hud_detector)):
                                roi_name = detector.roi_name
                                if roi_name in roi_px_cache:
                                    roi_px = roi_px_cache[roi_name]
                                    roi_frame = crop(frame, roi_px)
                                    if roi_frame.size > 0:
                                        detector.process(t_ms, roi_frame)
                        frame_idx += 1
                        continue
                
                # Run detectors (only during GAMEPLAY after match started)
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

                # Periodically update job progress so frontend can show a progress bar
                try:
                    if self._job_manager:
                        # Update every ~100 frames to avoid excessive updates
                        if frame_idx % 100 == 0:
                            self._job_manager.update_progress(
                                job_id,
                                processed_frames=int(frame_idx),
                                total_frames=int(total_frames),
                                events_detected=len(all_events),
                            )
                except Exception:
                    # Don't let progress update failures stop processing
                    pass

                frame_idx += 1
            
            cap.release()
            
            # Flush any remaining pending kills from KillfeedDetector
            for d in detectors:
                if hasattr(d, '_pending_kills') and d._pending_kills:
                    print(f"[{job_id}] Flushing {len(d._pending_kills)} pending kill events at end of video")
                    all_events.extend(d._pending_kills)
                    d._pending_kills.clear()
            
            # Post-process events
            self._job_manager.update_job_status(
                job_id, JobStatus.PROCESSING, "Post-processing events..."
            )
            
            # Filter out ghost players (OCR artifacts that matched historical roster names)
            # A ghost player appears only 1-2 times total (kills + deaths) when real players
            # typically have many more interactions across a full match
            all_events, ghost_removed = self._filter_ghost_player_kills(job_id, all_events)
            
            # Delete orphan crop files for ghost-filtered events (Fix 10)
            if ghost_removed:
                orphan_count = 0
                for ev in ghost_removed:
                    _key = (
                        ev.payload.get("killer_name", "").lower(),
                        ev.payload.get("victim_name", "").lower(),
                        int(ev.t_ms),
                    )
                    for d in detectors:
                        for fpath in d._crop_file_paths.pop(_key, []):
                            try:
                                os.remove(fpath)
                                orphan_count += 1
                            except OSError:
                                pass
                if orphan_count:
                    print(f"[{job_id}] Ghost orphan cleanup: deleted {orphan_count} crop/diag files")
            
            # Collect REPLAY-removed kills from all detectors
            replay_removed = []
            for d in detectors:
                if hasattr(d, '_replay_removed_kills'):
                    replay_removed.extend(d._replay_removed_kills)
            
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
            
            # Save invalidated kills (ghost player + replay filter removals)
            if ghost_removed or replay_removed:
                invalidated = {
                    "ghost_removed": [asdict(e) for e in ghost_removed],
                    "replay_removed": [asdict(e) for e in replay_removed],
                }
                inv_path = os.path.join(output_dir, f"{job_id}_invalidated.json")
                with open(inv_path, "w") as f:
                    json.dump(invalidated, f, indent=2)
                print(f"[{job_id}] Saved {len(ghost_removed)} ghost + {len(replay_removed)} replay invalidated kills to {inv_path}")
            
            # Save kill summary
            kill_events = [e for e in all_events if e.type == "KILL_EVENT"]
            round_transitions = [e for e in all_events if e.type == "ROUND_TRANSITION"]
            summary = self._build_kill_summary(kill_events, round_transitions)
            summary_path = os.path.join(output_dir, f"{job_id}_summary.json")
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            
            # Log skipped frames
            if skipped_replay_frames > 0 or skipped_transition_frames > 0 or skipped_prematch_frames > 0:
                print(f"[{job_id}] Skipped frames - Prematch: {skipped_prematch_frames}, Replay: {skipped_replay_frames}, Transition: {skipped_transition_frames}")
            
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
                "map": self._detected_map,
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
    
    def _filter_ghost_player_kills(self, job_id: str, events: List[Event]) -> List[Event]:
        """
        Filter out kills involving 'ghost players' - OCR artifacts that happened to
        match a name in the historical roster but only appear 1-2 times in the match.
        
        Real players in a full VALORANT match typically have 10+ total interactions
        (kills + deaths combined). A player with only 1-2 appearances is likely an
        OCR error that coincidentally matched a roster name.
        
        We filter conservatively: only remove if player has â‰¤2 total appearances
        AND the player is ONLY a victim (never got a kill themselves).
        """
        from collections import defaultdict
        
        # Count total appearances (kills + deaths) per player
        kill_counts = defaultdict(int)
        death_counts = defaultdict(int)
        
        for e in events:
            if e.type == "KILL_EVENT":
                killer = e.payload.get("killer_name", "Unknown")
                victim = e.payload.get("victim_name", "Unknown")
                if killer and killer != "Unknown":
                    kill_counts[killer] += 1
                if victim and victim != "Unknown":
                    death_counts[victim] += 1
        
        # Identify ghost players: â‰¤2 total appearances AND only as victim (0 kills)
        ghost_players = set()
        for player in set(kill_counts.keys()) | set(death_counts.keys()):
            total_appearances = kill_counts[player] + death_counts[player]
            player_kills = kill_counts[player]
            
            # Ghost criteria: very few appearances AND never got a kill
            # (a player with even 1 kill is likely real since OCR read both killer + victim)
            if total_appearances <= 2 and player_kills == 0:
                ghost_players.add(player)
        
        if ghost_players:
            print(f"[{job_id}] Ghost player filter: identified {len(ghost_players)} potential ghost players: {ghost_players}")
        
        # Filter out kills involving ghost players
        filtered_events = []
        ghost_removed = []
        
        for e in events:
            if e.type == "KILL_EVENT":
                killer = e.payload.get("killer_name", "Unknown")
                victim = e.payload.get("victim_name", "Unknown")
                
                if victim in ghost_players:
                    # Remove this kill - the victim is a ghost player
                    t_sec = e.t_ms / 1000
                    print(f"[{job_id}] Ghost filter removed: {killer} killed {victim} at t={t_sec:.1f}s (ghost victim)")
                    ghost_removed.append(e)
                    continue
                    
                # Note: We don't filter by ghost killer because if OCR read the killer name,
                # it's more likely the victim was also read correctly
            
            filtered_events.append(e)
        
        if ghost_removed:
            print(f"[{job_id}] Ghost player filter: removed {len(ghost_removed)} kill events")
        
        return filtered_events, ghost_removed
    
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
            "map": self._detected_map,
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
    ) -> Tuple[Optional[str], Optional[str], list, list]:
        """
        Detect team tags from the top HUD by OCR-ing the team tag regions.
        Returns (left_team_tag, right_team_tag, left_candidates, right_candidates).
        
        *_candidates are lists of (tag, score) tuples ranked by score for
        player-name validation fallback.
        
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
        # Initialize PaddleOCR separately for a second opinion
        # get_ocr_engine() uses lazy init — we must trigger it before
        # accessing the internal _paddleocr_reader.
        paddle_ocr = None
        try:
            from app.services.ocr.ocr_engine import get_ocr_engine
            engine = get_ocr_engine()
            engine._lazy_init()  # Ensure backends are initialized
            paddle_ocr = getattr(engine, '_paddleocr_reader', None)
            # Suppress "angle classifier is not initialized" warnings
            import logging
            logging.getLogger('ppocr').setLevel(logging.ERROR)
            if paddle_ocr is not None:
                print(f"[TeamTagDetector] PaddleOCR raw reader acquired for second opinion")
            else:
                # Fallback: initialize PaddleOCR directly
                try:
                    from paddleocr import PaddleOCR
                    logging.getLogger('ppocr').setLevel(logging.ERROR)
                    use_gpu_paddle = os.environ.get('USE_GPU', 'false').lower() == 'true'
                    paddle_ocr = PaddleOCR(
                        use_angle_cls=True,
                        lang='en',
                        use_gpu=use_gpu_paddle,
                        show_log=False,
                    )
                    print(f"[TeamTagDetector] PaddleOCR initialized directly (GPU={use_gpu_paddle})")
                except Exception as e2:
                    print(f"[TeamTagDetector] PaddleOCR direct init failed: {e2}")
        except Exception as e:
            print(f"[TeamTagDetector] PaddleOCR not available: {e}")
        
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
                print(f"[TeamTagDetector] t={t_sec}s: left='{left_tag}' (easyocr)")
            
            right_img = crop(frame, right_tag_px)
            right_tag = self._ocr_team_tag_enhanced(right_img, ocr_reader)
            if right_tag:
                right_detections.append(right_tag)
                print(f"[TeamTagDetector] t={t_sec}s: right='{right_tag}' (easyocr)")
            
            # PaddleOCR pass — different engine often reads differently
            if paddle_ocr is not None:
                for side, img, detections in [
                    ('left', left_img, left_detections),
                    ('right', right_img, right_detections),
                ]:
                    try:
                        tag = self._ocr_team_tag_paddle(img, paddle_ocr)
                        if tag:
                            detections.append(tag)
                            print(f"[TeamTagDetector] t={t_sec}s: {side}='{tag}' (paddleocr)")
                    except Exception:
                        pass
        
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
        
        # --- Post-OCR correction: match against known team tags from DB ---
        known_tags = self._get_known_team_tags()
        left_candidates = []
        right_candidates = []
        if known_tags:
            print(f"[TeamTagDetector] Loaded {len(known_tags)} known team tags from DB for validation")
            if left_detections:
                left_candidates = self._resolve_tag_from_detections(left_detections, known_tags, return_ranked=True)
                if left_candidates:
                    corrected = left_candidates[0][0]
                    if corrected != left_team:
                        print(f"[TeamTagDetector] Left team corrected: '{left_team}' -> '{corrected}'")
                        left_team = corrected
            if right_detections:
                right_candidates = self._resolve_tag_from_detections(right_detections, known_tags, return_ranked=True)
                if right_candidates:
                    corrected = right_candidates[0][0]
                    if corrected != right_team:
                        print(f"[TeamTagDetector] Right team corrected: '{right_team}' -> '{corrected}'")
                        right_team = corrected
        
        return left_team, right_team, left_candidates, right_candidates
    
    def _find_team_by_hud_names(self, hud_names: List[str]) -> tuple:
        """Query the database directly to find which team a set of HUD player
        names belongs to.  Returns (team_tag, match_count) with the most
        matching players, or (None, 0) if no matches found.
        
        This is the nuclear fallback: when OCR-derived tag candidates all fail
        player-overlap checks, we bypass the tag entirely and let the player
        names speak for themselves.
        """
        import psycopg2
        # Filter to names that look like real player names (>= 3 chars, mostly alpha)
        valid_names = []
        for n in hud_names:
            stripped = n.strip()
            if len(stripped) < 3:
                continue
            alpha_ratio = sum(1 for c in stripped if c.isalpha()) / len(stripped)
            if alpha_ratio < 0.5:
                continue
            valid_names.append(stripped.lower())
        
        if not valid_names:
            return None
        
        try:
            host = os.environ.get('POSTGRES_HOST', 'localhost')
            if host == 'postgres':
                host = 'host.docker.internal'
            conn = psycopg2.connect(
                host=host,
                port=int(os.environ.get('POSTGRES_PORT', 5432)),
                user=os.environ.get('POSTGRES_USER', 'postgres'),
                password=os.environ.get('POSTGRES_PASSWORD', ''),
                dbname=os.environ.get('POSTGRES_DB', 'cloud9'),
            )
            cur = conn.cursor()
            placeholders = ','.join(['%s'] * len(valid_names))
            query = (
                f"SELECT UPPER(t.team_tag), COUNT(DISTINCT LOWER(p.nickname)) "
                f"FROM esports_players p "
                f"JOIN esports_teams t ON p.team_id = t.id "
                f"WHERE LOWER(p.nickname) IN ({placeholders}) "
                f"GROUP BY t.team_tag "
                f"ORDER BY 2 DESC LIMIT 5"
            )
            cur.execute(query, valid_names)
            rows = cur.fetchall()
            cur.close()
            conn.close()
            
            if rows:
                print(f"[TeamValidation] DB player-name search results: {rows}")
                best_tag, best_count = rows[0]
                if best_count >= 2:
                    return best_tag, best_count
                # If only 1 match and it's clearly the best, still accept
                if best_count == 1 and (len(rows) == 1 or rows[1][1] < best_count):
                    return best_tag, best_count
            return None, 0
        except Exception as e:
            print(f"[TeamValidation] DB player-name search failed: {e}")
            return None, 0

    def _validate_team_via_players(
        self,
        job_id: str,
        cap: cv2.VideoCapture,
        fps: float,
        left_candidates: list,
        right_candidates: list,
    ):
        """Validate detected teams by extracting HUD player names and checking
        against each candidate team's player pool from the database.
        
        Two-phase approach:
        1. For ambiguous sides (close candidate scores), check top candidates'
           rosters against HUD names — handles OCR confusion like TL/IL/1L.
        2. For ALL sides, verify the final team has player overlap. If not,
           run a database-wide player-name search as a nuclear fallback.
        """
        from vod_processor.app.services.db.db_player_matcher import load_match_players_from_db
        
        # Always extract HUD names — we need them for both phases
        hud_names = self._quick_extract_hud_names(cap, fps)
        if not hud_names['left'] and not hud_names['right']:
            print("[TeamValidation] Could not extract any HUD names, skipping validation")
            return
        
        print(f"[TeamValidation] HUD names — left: {hud_names['left']}, right: {hud_names['right']}")
        
        # ── Phase 1: Candidate-based validation (ambiguous sides only) ──
        sides_to_check = []
        for side, candidates, code_attr in [
            ('left', left_candidates, '_left_team_code'),
            ('right', right_candidates, '_right_team_code'),
        ]:
            if len(candidates) < 2:
                continue
            best_score = candidates[0][1]
            runner_up_score = candidates[1][1]
            if best_score <= 0:
                continue
            gap_pct = (best_score - runner_up_score) / best_score
            if gap_pct < 0.20:  # Top two are within 20%
                sides_to_check.append((side, candidates, code_attr))
                print(f"[TeamValidation] {side} team ambiguous — "
                      f"top={candidates[0][0]}({best_score:.1f}) vs "
                      f"runner={candidates[1][0]}({runner_up_score:.1f}), "
                      f"gap={gap_pct:.0%}")
        
        for side, candidates, code_attr in sides_to_check:
            current_tag = candidates[0][0]
            side_names = hud_names.get(side, [])
            if not side_names:
                continue
            
            # Check how many HUD names match each candidate's roster
            best_match_count = 0
            best_match_tag = current_tag
            
            for tag, score in candidates[:5]:  # Check top 5 candidates
                try:
                    if side == 'left':
                        tag_pool, _ = load_match_players_from_db(tag, "")
                    else:
                        _, tag_pool = load_match_players_from_db("", tag)
                except Exception:
                    tag_pool = []
                
                if not tag_pool:
                    continue
                
                # Count fuzzy matches (case-insensitive substring)
                tag_pool_lower = [p.lower() for p in tag_pool]
                match_count = 0
                for name in side_names:
                    name_lower = name.lower()
                    for pool_name in tag_pool_lower:
                        if name_lower == pool_name or name_lower in pool_name or pool_name in name_lower:
                            match_count += 1
                            break
                
                print(f"[TeamValidation] {side} candidate '{tag}': "
                      f"{match_count}/{len(side_names)} names match "
                      f"({len(tag_pool)} players in roster)")
                
                if match_count > best_match_count:
                    best_match_count = match_count
                    best_match_tag = tag
            
            if best_match_tag != current_tag and best_match_count > 0:
                print(f"[TeamValidation] {side} team CORRECTED (phase 1): "
                      f"'{current_tag}' -> '{best_match_tag}' "
                      f"(player overlap: {best_match_count}/{len(side_names)})")
                setattr(self, code_attr, best_match_tag)
                setattr(self._player_matcher, code_attr, best_match_tag)
                try:
                    if side == 'left':
                        new_pool, _ = load_match_players_from_db(best_match_tag, "")
                        self._left_player_pool = new_pool or []
                        print(f"[{job_id}] Reloaded {len(self._left_player_pool)} players for {best_match_tag}")
                    else:
                        _, new_pool = load_match_players_from_db("", best_match_tag)
                        self._right_player_pool = new_pool or []
                        print(f"[{job_id}] Reloaded {len(self._right_player_pool)} players for {best_match_tag}")
                except Exception as e:
                    print(f"[TeamValidation] Failed to reload pool for '{best_match_tag}': {e}")
        
        # ── Phase 2: Nuclear fallback — DB-wide player-name search ──
        # For each side, verify the current team actually has player overlap.
        # If not, search ALL teams in the database by player name.
        print(f"[TeamValidation] Entering Phase 2 — checking {len(hud_names.get('left',[]))} left / {len(hud_names.get('right',[]))} right HUD names")
        for side, code_attr, pool_attr in [
            ('left', '_left_team_code', '_left_player_pool'),
            ('right', '_right_team_code', '_right_player_pool'),
        ]:
            current_tag = getattr(self, code_attr, None)
            current_pool = getattr(self, pool_attr, None) or []
            side_names = hud_names.get(side, [])
            
            if not side_names or not current_tag:
                continue
            
            # Check if current team has any overlap with HUD names
            pool_lower = [p.lower() for p in current_pool]
            overlap = 0
            for name in side_names:
                name_lower = name.lower()
                for pool_name in pool_lower:
                    if name_lower == pool_name or name_lower in pool_name or pool_name in name_lower:
                        overlap += 1
                        break
            
            if overlap >= 2:
                print(f"[TeamValidation] {side} team '{current_tag}' verified: "
                      f"{overlap}/{len(side_names)} HUD names match roster")
                continue
            
            # Weak or no overlap — run DB-wide search to see if a better team exists
            if overlap == 0:
                print(f"[TeamValidation] {side} team '{current_tag}' has 0 player overlap "
                      f"with HUD names — running DB-wide player search...")
            else:
                print(f"[TeamValidation] {side} team '{current_tag}' has weak overlap "
                      f"({overlap}/{len(side_names)}) — running DB-wide player search to verify...")
            
            found_tag, found_count = self._find_team_by_hud_names(side_names)
            if found_tag and found_tag != current_tag and found_count > overlap:
                print(f"[TeamValidation] {side} team CORRECTED (phase 2 DB search): "
                      f"'{current_tag}' -> '{found_tag}' ({found_count} DB matches vs {overlap} current)")
                setattr(self, code_attr, found_tag)
                setattr(self._player_matcher, code_attr, found_tag)
                try:
                    if side == 'left':
                        new_pool, _ = load_match_players_from_db(found_tag, "")
                        self._left_player_pool = new_pool or []
                        print(f"[{job_id}] Reloaded {len(self._left_player_pool)} players for {found_tag}")
                    else:
                        _, new_pool = load_match_players_from_db("", found_tag)
                        self._right_player_pool = new_pool or []
                        print(f"[{job_id}] Reloaded {len(self._right_player_pool)} players for {found_tag}")
                except Exception as e:
                    print(f"[TeamValidation] Failed to reload pool for '{found_tag}': {e}")
            else:
                print(f"[TeamValidation] {side} DB-wide search found no better match "
                      f"(result={found_tag}), keeping '{current_tag}'")
    
    def _quick_extract_hud_names(
        self, cap: cv2.VideoCapture, fps: float
    ) -> Dict[str, List[str]]:
        """Extract player names from the HUD player card slots.
        
        Lightweight: samples just 3 gameplay frames and OCRs the 5 player
        name slots on each side.  Returns {'left': [...], 'right': [...]}.
        """
        from config.settings import ROI_CONFIG
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        
        # Grab frames at gameplay timestamps (3-5 min in)
        sample_times = [180, 240, 300]
        
        left_names = set()
        right_names = set()
        
        try:
            import easyocr
            ocr = easyocr.Reader(['en'], gpu=True, verbose=False)
        except Exception:
            cap.set(cv2.CAP_PROP_POS_FRAMES, original_pos)
            return {'left': [], 'right': []}
        
        # Player name slot ROIs (5 per side)
        left_slot_rois = [ROI_CONFIG.get(f"left_player_{i}") for i in range(1, 6)]
        right_slot_rois = [ROI_CONFIG.get(f"right_player_{i}") for i in range(1, 6)]
        
        # Filter out None ROIs
        left_slot_rois = [r for r in left_slot_rois if r]
        right_slot_rois = [r for r in right_slot_rois if r]
        
        if not left_slot_rois and not right_slot_rois:
            # Try the bottom HUD player name regions
            bottom_left_rois = [ROI_CONFIG.get(f"bottom_left_player_{i}") for i in range(1, 6)]
            bottom_right_rois = [ROI_CONFIG.get(f"bottom_right_player_{i}") for i in range(1, 6)]
            left_slot_rois = [r for r in bottom_left_rois if r]
            right_slot_rois = [r for r in bottom_right_rois if r]
        
        for t_sec in sample_times:
            frame_num = int(t_sec * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                continue
            
            for roi in left_slot_rois:
                px = roi_to_px(frame_width, frame_height, roi)
                slot_img = crop(frame, px)
                if slot_img is not None and slot_img.size > 0:
                    try:
                        results = ocr.readtext(slot_img, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789')
                        for r in results:
                            if r and len(r) >= 3 and r[2] >= 0.4 and len(r[1].strip()) >= 2:
                                left_names.add(r[1].strip())
                    except Exception:
                        pass
            
            for roi in right_slot_rois:
                px = roi_to_px(frame_width, frame_height, roi)
                slot_img = crop(frame, px)
                if slot_img is not None and slot_img.size > 0:
                    try:
                        results = ocr.readtext(slot_img, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789')
                        for r in results:
                            if r and len(r) >= 3 and r[2] >= 0.4 and len(r[1].strip()) >= 2:
                                right_names.add(r[1].strip())
                    except Exception:
                        pass
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, original_pos)
        return {'left': list(left_names), 'right': list(right_names)}
    
    def _get_known_team_tags(self) -> List[str]:
        """Query all known team tags from the esports_teams database table."""
        try:
            import psycopg2
            host = os.environ.get('POSTGRES_HOST', 'localhost')
            if host == 'postgres':
                host = 'host.docker.internal'
            conn = psycopg2.connect(
                host=host,
                port=int(os.environ.get('POSTGRES_PORT', 5432)),
                user=os.environ.get('POSTGRES_USER', 'postgres'),
                password=os.environ.get('POSTGRES_PASSWORD', ''),
                dbname=os.environ.get('POSTGRES_DB', 'cloud9'),
            )
            cur = conn.cursor()
            cur.execute("SELECT DISTINCT UPPER(team_tag) FROM esports_teams WHERE team_tag IS NOT NULL AND team_tag != ''")
            tags = [row[0] for row in cur.fetchall()]
            cur.close()
            conn.close()
            return tags
        except Exception as e:
            print(f"[TeamTagDetector] Could not load known team tags from DB: {e}")
            return []
    
    # OCR confusion groups — characters that look similar in small HUD fonts
    _CONFUSION_GROUPS = [
        {'T', '1', 'I', '7'},   # thin vertical stroke (NOT L — L has a foot)
        {'L', '1', 'I'},         # L can look like 1/I but less like T
        {'O', '0', 'D', 'Q'},
        {'S', '5'},
        {'B', '8'},
        {'G', '6'},
        {'Z', '2'},
        {'U', 'V'},
    ]
    
    @staticmethod
    def _chars_confusable(a: str, b: str) -> bool:
        """Check if two characters are commonly confused by OCR."""
        if a == b:
            return True
        for group in VODProcessor._CONFUSION_GROUPS:
            if a in group and b in group:
                return True
        return False
    
    def _resolve_tag_from_detections(
        self, raw_detections: List[str], known_tags: List[str],
        return_ranked: bool = False
    ) -> Optional[str]:
        """
        Given all per-frame raw OCR detections and a list of known DB tags,
        vote for the best known tag using confusion-aware matching.
        
        Each raw detection votes for every known tag it could be (all chars
        exact or confusable). Votes are weighted: exact char matches score
        higher than confusable matches, so 'TL' voting for known 'TL' beats
        '1L' voting for known 'TL'. This correctly distinguishes T1 vs TL
        even when OCR reads '11' or '1L'.
        
        If return_ranked=True, returns list of (tag, score) tuples sorted
        descending.  Otherwise returns the best tag string.
        """
        from collections import Counter
        
        # Tally weighted votes for each known tag
        tag_scores: Dict[str, float] = {}
        
        for raw in raw_detections:
            raw_upper = raw.upper()
            for known in known_tags:
                if len(known) != len(raw_upper):
                    continue
                
                # Check if every character is exact or confusable
                all_match = True
                exact_count = 0
                for a, b in zip(raw_upper, known):
                    if a == b:
                        exact_count += 1
                    elif not self._chars_confusable(a, b):
                        all_match = False
                        break
                
                if all_match:
                    score = exact_count + (len(raw_upper) - exact_count) * 0.5
                    tag_scores[known] = tag_scores.get(known, 0) + score
        
        if not tag_scores:
            counter = Counter(raw_detections)
            raw_best = counter.most_common(1)[0][0] if counter else None
            if return_ranked:
                return [(raw_best, 0.0)] if raw_best else []
            return raw_best
        
        sorted_scores = sorted(tag_scores.items(), key=lambda x: x[1], reverse=True)
        print(f"[TeamTagDetector] Tag candidates: {sorted_scores[:5]}")
        
        if return_ranked:
            return sorted_scores
        return sorted_scores[0][0]
    
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
        """OCR a team tag with enhanced preprocessing for small white text.

        Uses multiple preprocessing pipelines and both EasyOCR and PaddleOCR
        for consensus. Sharpening and morphological dilation help distinguish
        chars like T/1/I that share a thin vertical stroke.
        """
        import re

        if img is None or img.size == 0:
            return None

        try:
            # Upscale to help OCR on small fonts
            scale = 4
            scaled = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

            # Grayscale + CLAHE for all derived variants
            gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)

            preprocessed_images = []

            # 1) Color-scaled copy
            preprocessed_images.append(scaled)

            # 2) CLAHE enhanced
            enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            preprocessed_images.append(enhanced_bgr)

            # 3) Otsu threshold
            _, th = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            th_bgr = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
            preprocessed_images.append(th_bgr)

            # 4) Sharpened — helps preserve horizontal strokes (T vs 1)
            sharpen_kernel = np.array([[-1, -1, -1],
                                       [-1,  9, -1],
                                       [-1, -1, -1]], dtype=np.float32)
            sharpened = cv2.filter2D(scaled, -1, sharpen_kernel)
            preprocessed_images.append(sharpened)

            # 5) Dilated threshold — thickens strokes so T's crossbar is visible
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            dilated = cv2.dilate(th, kernel, iterations=1)
            dilated_bgr = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)
            preprocessed_images.append(dilated_bgr)

            # 6) Horizontal morphological close — reconnects T crossbar that
            #    thresholding may break, then dilate to thicken
            kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
            h_closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel_h)
            h_closed_bgr = cv2.cvtColor(h_closed, cv2.COLOR_GRAY2BGR)
            preprocessed_images.append(h_closed_bgr)

            results = []

            # --- EasyOCR pass ---
            for img_version in preprocessed_images:
                try:
                    ocr_results = ocr_reader.readtext(
                        img_version,
                        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                        paragraph=False,
                        min_size=5,
                        detail=1,
                    )
                except Exception:
                    ocr_results = []

                for entry in ocr_results:
                    if not entry:
                        continue
                    if isinstance(entry, tuple) and len(entry) >= 3:
                        text = str(entry[1]).upper().strip()
                        conf = float(entry[2]) if entry[2] is not None else 0.0
                    else:
                        text = str(entry).upper().strip()
                        conf = 0.0
                    if text:
                        clean = re.sub(r'[^A-Z0-9]', '', text)
                        if 1 < len(clean) <= 6:
                            results.append((clean, conf, "easyocr"))

            # --- PaddleOCR pass (second opinion via det=False) ---
            try:
                from app.services.ocr.ocr_engine import get_ocr_engine
                engine = get_ocr_engine()
                engine._lazy_init()
                paddle_reader = getattr(engine, '_paddleocr_reader', None)
                if paddle_reader is not None:
                    # Also add horizontal-closed variant for T crossbar
                    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
                    h_closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel_h)
                    h_closed_bgr = cv2.cvtColor(h_closed, cv2.COLOR_GRAY2BGR)
                    
                    for img_version in [sharpened, th_bgr, dilated_bgr, h_closed_bgr]:
                        try:
                            # det=False: skip detection, run recognition on whole crop
                            paddle_results = paddle_reader.ocr(img_version, det=False, cls=True)
                            if paddle_results and paddle_results[0]:
                                for item in paddle_results[0]:
                                    if item and len(item) >= 2:
                                        text = str(item[0]).upper().strip()
                                        conf = float(item[1]) if item[1] is not None else 0.0
                                        clean = re.sub(r'[^A-Z0-9]', '', text)
                                        if 1 < len(clean) <= 6:
                                            results.append((clean, conf, "paddleocr"))
                        except Exception:
                            continue
            except Exception:
                pass  # PaddleOCR not available or failed, continue with EasyOCR results

            if not results:
                return None

            # Return highest-confidence result (require modest confidence)
            results.sort(key=lambda x: x[1], reverse=True)
            best, best_conf, best_engine = results[0]
            if best_conf >= 0.35 or len(best) <= 3:
                return best
            return None
        except Exception:
            return None

    def _ocr_team_tag_paddle(self, img: np.ndarray, paddle_ocr) -> Optional[str]:
        """OCR a team tag using PaddleOCR with preprocessing tuned for small white text.
        
        Adds black padding around the crop so PaddleOCR's recognition model
        has border context and doesn't lose edge characters.  Tries both
        det=False (recognition-only) and det=True (full pipeline) on the
        6×-upscaled, padded image.
        """
        import re
        
        if img is None or img.size == 0:
            return None
        
        try:
            scale = 6
            scaled = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            
            # --- Add black padding (30px each side) so edge chars aren't clipped ---
            pad = 30
            padded = cv2.copyMakeBorder(scaled, pad, pad, pad, pad,
                                        cv2.BORDER_CONSTANT, value=(0, 0, 0))
            
            # Sharpen to preserve horizontal strokes (T crossbar)
            sharpen_kernel = np.array([[-1, -1, -1],
                                       [-1,  9, -1],
                                       [-1, -1, -1]], dtype=np.float32)
            sharpened = cv2.filter2D(padded, -1, sharpen_kernel)
            
            # Grayscale + high-contrast threshold
            gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Dilate slightly to thicken strokes
            kernel_sq = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            dilated = cv2.dilate(thresh, kernel_sq, iterations=1)
            dilated_bgr = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)
            
            # Horizontal-emphasis: close with wide kernel to preserve T crossbar
            kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
            h_closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_h)
            h_closed_bgr = cv2.cvtColor(h_closed, cv2.COLOR_GRAY2BGR)
            
            best_text = None
            best_conf = 0.0
            
            variant_names = ['sharp', 'dilated', 'h_closed']
            variants = [sharpened, dilated_bgr, h_closed_bgr]
            
            # Pass A: det=False (recognition-only on whole image)
            for vi, variant in enumerate(variants):
                try:
                    results = paddle_ocr.ocr(variant, det=False, cls=False)
                    if results and results[0]:
                        for item in results[0]:
                            if item and len(item) >= 2:
                                text = str(item[0]).upper().strip()
                                conf = float(item[1]) if item[1] is not None else 0.0
                                clean = re.sub(r'[^A-Z0-9]', '', text)
                                print(f"[PaddleOCR-tag] det=F {variant_names[vi]} raw='{text}' clean='{clean}' conf={conf:.3f}")
                                if 1 < len(clean) <= 6 and conf > best_conf:
                                    best_text = clean
                                    best_conf = conf
                except Exception as e:
                    print(f"[PaddleOCR-tag] det=F {variant_names[vi]} failed: {e}")
            
            # Pass B: det=True (full pipeline — image is large enough after 6× + padding)
            for vi, variant in enumerate(variants):
                try:
                    results = paddle_ocr.ocr(variant, det=True, cls=False)
                    if results and results[0]:
                        for line in results[0]:
                            if line and len(line) >= 2:
                                text = str(line[1][0]).upper().strip()
                                conf = float(line[1][1]) if line[1][1] is not None else 0.0
                                clean = re.sub(r'[^A-Z0-9]', '', text)
                                print(f"[PaddleOCR-tag] det=T {variant_names[vi]} raw='{text}' clean='{clean}' conf={conf:.3f}")
                                if 1 < len(clean) <= 6 and conf > best_conf:
                                    best_text = clean
                                    best_conf = conf
                except Exception as e:
                    print(f"[PaddleOCR-tag] det=T {variant_names[vi]} failed: {e}")
            
            if best_text and (best_conf >= 0.25 or len(best_text) <= 3):
                return best_text
            return None
        except Exception as e:
            print(f"[TeamTagDetector] _ocr_team_tag_paddle error: {e}")
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
        
        from vod_processor.app.services.ocr.player_name_extractor import PlayerNameExtractor
        
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
        
        # Detect map from later in the video where the series scoreboard is visible
        # The first 30 seconds often show intro/agent select without the series bar
        try:
            from vod_processor.app.services.state.map_detector import MapDetector
            map_detector = MapDetector()
            
            # Sample frames from 2-5 minutes into the video for map detection
            # This is when gameplay is happening and the series scoreboard is visible
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            map_sample_frames = []
            start_frame = int(120 * fps)  # Start at 2 minutes
            end_frame = min(int(300 * fps), total_frames - 1)  # End at 5 minutes or video end
            map_sample_interval = int(fps * 10)  # Every 10 seconds
            
            for frame_idx in range(start_frame, end_frame, map_sample_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    map_sample_frames.append(frame)
            
            if map_sample_frames:
                print(f"[MapDetector] Sampling {len(map_sample_frames)} frames from t=120-300s for map detection")
                detected_map = map_detector.detect_map_from_frames(map_sample_frames)
                if detected_map:
                    self._detected_map = detected_map
                    print(f"Detected map: {detected_map}")
                else:
                    print("WARNING: Could not detect map from broadcast")
            else:
                print("WARNING: No frames available for map detection")
        except Exception as e:
            import traceback
            print(f"WARNING: Map detection failed: {e}")
            traceback.print_exc()
        
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
    
    # How long to persist REPLAY state after detection (in frames at ~6fps = ~2 seconds)
    # Short persist to bridge frames where OCR misses between detections.
    REPLAY_PERSIST_FRAMES = 12
    
    def __init__(self):
        self._ocr_reader = None
        self._ocr_initialized = False
        self._last_state = "GAMEPLAY"
        self._state_count = 0  # For hysteresis
        self._replay_persist_counter = 0  # Persist REPLAY state for a few frames
        self._last_replay_detection_logged = False
    
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
        t_ms: float = 0,
    ) -> str:
        """
        Detect the current frame state.
        
        Returns:
            "GAMEPLAY" - Normal gameplay, process events
            "REPLAY" - Overlay text visible (REPLAY/CLUTCH/THRIFTY/FLAWLESS), skip killfeed
            "TRANSITION" - Non-gameplay screen, skip all processing
        """
        # Check for any overlay text (REPLAY, CLUTCH, THRIFTY, FLAWLESS)
        # ALL overlay text means we're in replay/highlight footage — skip killfeed
        detected = self._detect_replay_or_clutch_text(replay_roi, t_ms)
        
        if detected:
            # Reset persist counter on fresh detection
            self._replay_persist_counter = self.REPLAY_PERSIST_FRAMES
            return "REPLAY"
        
        # If we detected overlay text recently, persist REPLAY state for a few more frames
        # This handles cases where OCR misses a frame but replay is still showing
        if self._replay_persist_counter > 0:
            self._replay_persist_counter -= 1
            return "REPLAY"
        
        # Check if standard HUD is present
        if not self._has_standard_hud(score_bar_roi, left_panels_roi, right_panels_roi):
            return "TRANSITION"
        
        return "GAMEPLAY"
    
    def _detect_replay_or_clutch_text(self, replay_roi: np.ndarray, t_ms: float = 0) -> bool:
        """
        Detect if overlay text is visible in the bottom-right corner.
        
        ANY overlay text (REPLAY, CLUTCH, THRIFTY, FLAWLESS) means we're in
        replay/highlight footage and should skip killfeed processing.
        
        Returns True if any overlay text is detected.
        """
        if replay_roi.size == 0:
            return False
        
        h, w = replay_roi.shape[:2]
        
        # Look for high-contrast white text on dark background
        gray = cv2.cvtColor(replay_roi, cv2.COLOR_BGR2GRAY)
        
        # The overlay text is typically white/light on darker semi-transparent background
        # Use adaptive threshold for better detection across varying backgrounds
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        white_ratio = np.sum(thresh > 0) / thresh.size
        
        # If there's some white content, try OCR (lowered threshold from 0.05 to 0.02)
        _dbg_replay = (1280000 < t_ms < 1370000) or (1440000 < t_ms < 1480000) or (1935000 < t_ms < 1960000) or (2050000 < t_ms < 2090000)
        if _dbg_replay and not (white_ratio > 0.02 and white_ratio < 0.7):
            print(f"[REPLAY-DBG] t={t_ms/1000:.1f}s SKIPPED OCR: white_ratio={white_ratio:.3f} h={h} w={w}")
        if white_ratio > 0.02 and white_ratio < 0.7:
            self._init_ocr()
            if self._ocr_reader:
                try:
                    # First try without allowlist for better detection
                    results = self._ocr_reader.readtext(
                        replay_roi, 
                        detail=0,
                        paragraph=False,  # Don't merge - we want individual words
                    )
                    
                    # DEBUG: Log what OCR sees during known replay windows
                    _dbg_replay = (1280000 < t_ms < 1370000) or (1440000 < t_ms < 1480000) or (1935000 < t_ms < 1960000) or (2050000 < t_ms < 2090000)
                    if _dbg_replay and results:
                        print(f"[REPLAY-DBG] t={t_ms/1000:.1f}s white_ratio={white_ratio:.3f} OCR={results}")
                    elif _dbg_replay:
                        print(f"[REPLAY-DBG] t={t_ms/1000:.1f}s white_ratio={white_ratio:.3f} OCR=<empty>")
                    
                    for text in results:
                        if isinstance(text, str):
                            text_upper = text.upper().replace(" ", "").replace("_", "")
                            
                            # CLUTCH = replay/highlight overlay
                            if "CLUTCH" in text_upper or "CLUT" in text_upper:
                                if not self._last_replay_detection_logged:
                                    print(f"[FrameState] CLUTCH detected at t={t_ms/1000:.1f}s - entering REPLAY mode")
                                    self._last_replay_detection_logged = True
                                return True
                            
                            # REPLAY = replay footage
                            # Also match common OCR errors: REPLA, REPIAY, REPALY
                            if "REPLAY" in text_upper or "REPLA" in text_upper or "REPIAY" in text_upper:
                                if not self._last_replay_detection_logged:
                                    print(f"[FrameState] REPLAY detected at t={t_ms/1000:.1f}s - entering REPLAY mode")
                                    self._last_replay_detection_logged = True
                                return True
                            
                            # THRIFTY = round win overlay
                            if "THRIFTY" in text_upper or "THRIFT" in text_upper:
                                if not self._last_replay_detection_logged:
                                    print(f"[FrameState] THRIFTY detected at t={t_ms/1000:.1f}s - entering REPLAY mode")
                                    self._last_replay_detection_logged = True
                                return True
                            
                            # FLAWLESS = round win overlay (no deaths)
                            if "FLAWLESS" in text_upper or "FLAWLES" in text_upper:
                                if not self._last_replay_detection_logged:
                                    print(f"[FrameState] FLAWLESS detected at t={t_ms/1000:.1f}s - entering REPLAY mode")
                                    self._last_replay_detection_logged = True
                                return True
                                
                except Exception:
                    pass
        
        # If we get here without detection, reset the logged flag
        self._last_replay_detection_logged = False
        
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
        
        # Check for orange (including secondary red/pink range)
        orange_mask = _build_orange_mask(hsv)
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
        
        # Crop saving: when set, weapon icon crops are written to this directory
        self._crop_output_dir: Optional[str] = None
        self._crop_counter: int = 0
        self._crop_file_paths: dict = {}  # (killer_lower, victim_lower, int_t_ms) -> [path, ...]
        self._ult_diagnostics: list = []  # collected per-crop ult badge metrics
        # Deferred crop: wait 2 frames for the row to stabilize before cropping.
        # Key: (killer_norm, victim_norm, round) -> dict with row_img, ktr, vtl, etc.
        self._pending_crops: dict = {}
        self._CROP_DEFER_FRAMES: int = 2  # number of frames to wait before cropping
        
        # Scheduled halftime start (delayed from transition to capture final kills)
        self._halftime_scheduled_ms: float = 0.0
        # Track the round transition time for accurate round display in buffer window
        self._last_transition_ms: float = 0.0
        self._last_transition_round: int = 0
        
        # REPLAY lookback filter: When REPLAY is detected, invalidate kills from the
        # previous ~1 second because the replay overlay appears slightly AFTER the
        # killfeed starts showing replay content
        self._REPLAY_LOOKBACK_MS: float = 1500  # 1.5 seconds lookback
        self._pending_kills: List[Event] = []  # Buffer kills before confirming them
        self._replay_removed_kills: List[Event] = []  # Kills removed by REPLAY filter
    
    def on_replay_detected(self, replay_start_ms: float):
        """
        Called when REPLAY mode is detected. Filter out any pending kills that
        happened within the lookback window - they're likely from the start of
        the replay segment.
        """
        cutoff_ms = replay_start_ms - self._REPLAY_LOOKBACK_MS
        original_count = len(self._pending_kills)
        
        # Filter out kills within the lookback window
        kept = []
        removed = []
        for k in self._pending_kills:
            if k.t_ms < cutoff_ms:
                kept.append(k)
            else:
                removed.append(k)
        self._pending_kills = kept
        
        filtered_count = len(removed)
        if filtered_count > 0:
            self._replay_removed_kills.extend(removed)
            print(f"[KillfeedDetector] REPLAY lookback filter: removed {filtered_count} kill(s) from t={cutoff_ms/1000:.1f}s to t={replay_start_ms/1000:.1f}s")
    
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
            from vod_processor.app.services.ocr.ocr_engine import get_ocr_engine
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
        # ---- DEBUG: trace self-kill window ----
        # Enable debug for known self-kill timestamps:
        # R10 fall damage ~1455-1475s, R13 spike ~1940-1960s, R14 spike ~2055-2070s
        _DBG_SELF = (1455000 < t_ms < 1475000) or (1935000 < t_ms < 1960000) or (2055000 < t_ms < 2070000)
        if _DBG_SELF:
            print(f"[DBG-SELF] t={t_ms/1000:.1f}s _detect ENTERED")
        # ---- DEBUG: trace dead zone (R13-R25) kill pipeline ----
        _DBG_DEAD = (1695000 <= t_ms <= 3100000)
        if _DBG_DEAD and not hasattr(self, '_dbg_dead_counts'):
            self._dbg_dead_counts = {"hash_skip": 0, "parse_none": 0, "conf": 0, "unk_name": 0, "db_miss": 0, "dup": 0, "accepted": 0, "rows_seen": 0}
            self._dbg_dead_last_print = 0
        # Skip expensive OCR if killfeed hasn't changed
        if not self._has_significant_change(roi_frame):
            if _DBG_SELF:
                print(f"[DBG-SELF] t={t_ms/1000:.1f}s SKIPPED by _has_significant_change")
            return []
        events = []
        h, w = roi_frame.shape[:2]
        # Segment rows using fixed positions for consistent extraction
        self._dbg_self = _DBG_SELF
        rows = self._segment_rows_fixed(roi_frame)
        if _DBG_SELF:
            print(f"[DBG-SELF] t={t_ms/1000:.1f}s _segment_rows_fixed returned {len(rows)} rows: {[(r[0], r[1], r[2]) for r in rows]}")
        
        KILLFEED_DISPLAY_WINDOW_MS = 5500  # Kills visible ~5s + 500ms buffer for frame quantization
        for actual_row_idx, y_start, y_end, row_img in rows:
            # Per-row change detection - skip OCR if this row hasn't changed
            row_hash = self._compute_row_hash(row_img)
            if actual_row_idx in self._row_hashes and self._row_hashes[actual_row_idx] == row_hash:
                if _DBG_SELF:
                    print(f"[DBG-SELF] t={t_ms/1000:.1f}s ROW {actual_row_idx} skipped (row hash unchanged)")
                if _DBG_DEAD:
                    self._dbg_dead_counts["hash_skip"] += 1
                continue  # Row unchanged, skip expensive OCR
            if _DBG_DEAD:
                self._dbg_dead_counts["rows_seen"] += 1
            
            entry = self._parse_row(row_img)
            if _DBG_SELF:
                print(f"[DBG-SELF] t={t_ms/1000:.1f}s ROW {actual_row_idx} _parse_row -> {entry}")
            if not entry:
                if _DBG_DEAD:
                    self._dbg_dead_counts["parse_none"] += 1
                continue
            
            # NOTE: Do NOT store the row hash here.  We only seal the hash
            # after the kill is *accepted* or confirmed as a *duplicate* of an
            # already-accepted kill.  If the OCR produces garbled names that
            # fail database/confidence filters, we want to retry this row on
            # the next frame while the text is still at full opacity.
            
            # Get values
            killer_team = entry.get("killer_team", "unknown")
            victim_team = entry.get("victim_team", "unknown")
            killer_name = entry.get("killer_name", "Unknown")
            victim_name = entry.get("victim_name", "Unknown")
            confidence = entry.get("confidence", 0.5)
            
            # Filter 1: Require minimum confidence
            if confidence < 0.7:
                if _DBG_DEAD:
                    self._dbg_dead_counts["conf"] += 1
                continue
            
            # Filter 2: Require BOTH killer and victim names (every kill has both in this VOD)
            # Exception: fall damage would only have victim, but that's rare
            if killer_name == "Unknown" or victim_name == "Unknown":
                if _DBG_DEAD:
                    self._dbg_dead_counts["unk_name"] += 1
                    print(f"[DBG-DEAD] t={t_ms/1000:.1f}s R{actual_row_idx} Filter2: killer='{killer_name}' victim='{victim_name}'")
                continue
            
            # Convert colors to team codes for team-aware player matching
            # NOTE: _parse_row already overrides colors using the player matcher's
            # side-based logic (left→teal, right→orange), so the colors here are
            # already "canonical" (teal=left, orange=right) regardless of halftime.
            # We must NOT apply halftime swap again via get_team_code_from_color(),
            # which would double-swap and assign the wrong team pool.
            # Instead, map directly: teal → left team, orange → right team.
            if killer_team in ('teal', 'green', 'cyan'):
                killer_team_code = self._left_team_code
            elif killer_team in ('orange', 'red', 'yellow'):
                killer_team_code = self._right_team_code
            else:
                killer_team_code = None
            if victim_team in ('teal', 'green', 'cyan'):
                victim_team_code = self._left_team_code
            elif victim_team in ('orange', 'red', 'yellow'):
                victim_team_code = self._right_team_code
            else:
                victim_team_code = None
            
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
                if _DBG_DEAD:
                    self._dbg_dead_counts["unk_name"] += 1
                continue
            
            # Filter 4: If we have player matcher loaded, REQUIRE BOTH names to match database
            # This prevents ghost players like "Flame" from appearing (OCR garbage that 
            # coincidentally looks like a valid name but isn't in the match)
            if self._player_matcher:
                if killer_name_db is None or victim_name_db is None:
                    if _DBG_DEAD:
                        self._dbg_dead_counts["db_miss"] += 1
                        print(f"[DBG-DEAD] t={t_ms/1000:.1f}s Filter4: killer='{killer_name}'->db={killer_name_db}(team={killer_team_code}) victim='{victim_name}'->db={victim_name_db}(team={victim_team_code})")
                    # At least one name didn't match database - skip this kill
                    # This is stricter but prevents false positives when we know the player pool
                    continue

            # Check for duplicates using normalized names
            sig = (t_ms, killer_team, victim_team, killer_name_normalized, victim_name_normalized, actual_row_idx)
            if self._is_duplicate_scroll_aware(t_ms, sig, KILLFEED_DISPLAY_WINDOW_MS):
                if _DBG_DEAD:
                    self._dbg_dead_counts["dup"] += 1
                # Check if there's a pending (deferred) crop for this kill.
                # If so, update it with this frame's data (row is more stable now).
                pending_key = (killer_name_normalized.lower(), victim_name_normalized.lower())
                if pending_key in self._pending_crops:
                    pc = self._pending_crops[pending_key]
                    pc["row_img"] = row_img.copy()
                    pc["frames_seen"] += 1
                    if pc["frames_seen"] >= self._CROP_DEFER_FRAMES:
                        self._finalize_deferred_crop(pending_key)
                # Seal hash for confirmed duplicates
                self._row_hashes[actual_row_idx] = row_hash
                continue
            self.recent_signatures.append(sig)
            
            # Seal the row hash now that the kill is accepted.
            self._row_hashes[actual_row_idx] = row_hash
            
            # Determine round number for display - if within 5s buffer of last transition,
            # the kill belongs to the ending round, not the new round
            BUFFER_MS = 5000
            if self._last_transition_ms > 0 and (t_ms - self._last_transition_ms) < BUFFER_MS:
                # Within buffer window - kill belongs to the round that just ended
                display_round = self._last_transition_round
            else:
                # Past buffer - kill belongs to current round
                display_round = self._current_round_number
            
            if _DBG_DEAD:
                self._dbg_dead_counts["accepted"] += 1
            
            # Detect self-kill (fall damage, etc.) - killer and victim are the same player
            is_self_kill = (killer_name_normalized.lower().strip() == victim_name_normalized.lower().strip())

            # Log the accepted kill - just player names, no team prefix (OCR may include it)
            if is_self_kill:
                print(f"[KILL] t={t_ms/1000:.1f}s R{display_round} ROW {actual_row_idx+1}: {killer_name_normalized} SELF-KILL (fall damage)")
            else:
                print(f"[KILL] t={t_ms/1000:.1f}s R{display_round} ROW {actual_row_idx+1}: {killer_name_normalized} killed {victim_name_normalized}")

            # Track this victim's death for per-round deduplication
            victim_key = victim_name_normalized.lower().strip() if victim_name_normalized != "Unknown" else None
            if victim_key:
                self._victim_last_death[victim_key] = (t_ms, killer_name_normalized)

            # Defer weapon/ability icon crop  — store for 2 frames to let the
            # row stabilize (animation flash fades, text fully loads).
            ktr = entry.get("killer_text_right")
            vtl = entry.get("victim_text_left")
            pending_key = (killer_name_normalized.lower(), victim_name_normalized.lower())
            self._pending_crops[pending_key] = {
                "row_img": row_img.copy(),
                "ktr": ktr,
                "vtl": vtl,
                "t_ms": t_ms,
                "display_round": display_round,
                "killer_name": killer_name_normalized,
                "victim_name": victim_name_normalized,
                "is_self_kill": is_self_kill,
                "actual_row_idx": actual_row_idx,
                "frames_seen": 0,  # will increment on dup frames
            }
            print(f"[CROP-DEFER] crop pending for {killer_name_normalized} -> {victim_name_normalized} ktr={ktr} vtl={vtl} (waiting {self._CROP_DEFER_FRAMES} frames)")

            entry["weapon_icon"] = None

            # Create kill event with normalized names
            kill_event = Event(
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
                    "is_self_kill": is_self_kill,
                },
                confidence=confidence
            )
            
            # Buffer kill in pending list for REPLAY lookback filtering
            self._pending_kills.append(kill_event)

            # Also emit death event with normalized name
            death_event = Event(
                t_ms=t_ms,
                type="DEATH_EVENT",
                roi=self.roi_name,
                payload={
                    "player_name": victim_name_normalized,
                    "player_team": entry.get("victim_team", "unknown"),
                    "killed_by": killer_name_normalized,
                },
                confidence=entry.get("confidence", 0.5)
            )
            self._pending_kills.append(death_event)
        
        # Flush confirmed kills (those older than the lookback window)
        # This allows recent kills to be filtered if REPLAY is detected
        flush_cutoff_ms = t_ms - self._REPLAY_LOOKBACK_MS
        confirmed_events = [k for k in self._pending_kills if k.t_ms < flush_cutoff_ms]
        self._pending_kills = [k for k in self._pending_kills if k.t_ms >= flush_cutoff_ms]
        events.extend(confirmed_events)
        
        # Finalize any deferred crops that have been pending long enough
        # (safety net in case dup path was never hit, e.g. row disappeared)
        stale_keys = [k for k, v in self._pending_crops.items()
                      if v["frames_seen"] >= self._CROP_DEFER_FRAMES
                      or (t_ms - v["t_ms"]) > 500]  # 500ms max wait
        for pk in stale_keys:
            self._finalize_deferred_crop(pk)
        
        # ---- DEBUG: periodic summary for dead zone ----
        if _DBG_DEAD and hasattr(self, '_dbg_dead_counts'):
            if t_ms - self._dbg_dead_last_print > 30000:  # Every 30s
                c = self._dbg_dead_counts
                print(f"[DBG-DEAD-SUMMARY] t={t_ms/1000:.1f}s rows_seen={c['rows_seen']} hash_skip={c['hash_skip']} parse_none={c['parse_none']} conf={c['conf']} unk_name={c['unk_name']} db_miss={c['db_miss']} dup={c['dup']} accepted={c['accepted']}")
                self._dbg_dead_last_print = t_ms
        
        return events

    def _finalize_deferred_crop(self, pending_key: tuple) -> None:
        """Finalize a deferred crop using the stored (stabilized) row image."""
        pc = self._pending_crops.pop(pending_key, None)
        if pc is None:
            return
        
        row_img = pc["row_img"]
        ktr = pc["ktr"]
        vtl = pc["vtl"]
        t_ms = pc["t_ms"]
        
        print(f"[CROP-DEFER] finalizing crop for {pc['killer_name']} -> {pc['victim_name']} (frames_seen={pc['frames_seen']})")
        
        # Seal the row hash now that we're done with this row
        actual_row_idx = pc.get("actual_row_idx")
        if actual_row_idx is not None:
            row_hash = self._compute_row_hash(row_img)
            self._row_hashes[actual_row_idx] = row_hash
        
        icon_img = None
        try:
            icon_img = self._extract_weapon_icon(
                row_img,
                killer_text_right=ktr,
                victim_text_left=vtl,
            )
        except Exception as _crop_err:
            import traceback; traceback.print_exc()
            print(f"[CROP-ERR] _extract_weapon_icon failed: {_crop_err}")

        # Check for ult badge
        if icon_img is not None and self._last_crop_bounds is not None and vtl is not None:
            weapon_right = self._last_crop_bounds[1]
            try:
                badge = self._maybe_extract_ult_badge(row_img, weapon_right, vtl)
                if badge is not None:
                    icon_img = badge
                    self._last_crop_method = "ult_badge"
            except Exception:
                pass

        crop_method = getattr(self, '_last_crop_method', None) or "unknown"

        if icon_img is not None and self._crop_output_dir:
            self._crop_counter += 1
            method_dir = os.path.join(self._crop_output_dir, crop_method)
            os.makedirs(method_dir, exist_ok=True)
            crop_path = os.path.join(
                method_dir,
                f"crop_{self._crop_counter:05d}_t{int(t_ms)}ms.png"
            )
            cv2.imwrite(crop_path, icon_img)

            # Track crop paths for ghost orphan cleanup
            _crop_key = (pc["killer_name"].lower(), pc["victim_name"].lower(), int(t_ms))
            self._crop_file_paths.setdefault(_crop_key, []).append(crop_path)

            # Diagnostic: save annotated full row alongside crop
            try:
                diag_dir = os.path.join(self._crop_output_dir, "diag")
                os.makedirs(diag_dir, exist_ok=True)
                diag_row = row_img.copy()
                rh, rw = diag_row.shape[:2]
                if ktr is not None:
                    ktr_px = int(round(ktr))
                    cv2.line(diag_row, (ktr_px, 0), (ktr_px, rh), (0, 0, 255), 2)
                if vtl is not None:
                    vtl_px = int(round(vtl))
                    cv2.line(diag_row, (vtl_px, 0), (vtl_px, rh), (255, 0, 0), 2)
                last_bounds = getattr(self, '_last_crop_bounds', None)
                if last_bounds is not None:
                    cx0, cx1 = last_bounds
                    cv2.rectangle(diag_row, (cx0, 0), (cx1, rh), (0, 255, 0), 2)
                last_zone = getattr(self, '_last_search_zone', None)
                if last_zone is not None:
                    sz0, sz1 = last_zone
                    cv2.rectangle(diag_row, (sz0, 2), (sz1, rh - 2), (0, 255, 255), 1)
                ult_bounds = getattr(self, '_last_ult_badge_bounds', None)
                if ult_bounds is not None:
                    ub0, ub1 = ult_bounds
                    cv2.rectangle(diag_row, (ub0, 0), (ub1, rh), (255, 255, 0), 2)
                diag_path = os.path.join(diag_dir, f"row_{self._crop_counter:05d}_t{int(t_ms)}ms.png")
                cv2.imwrite(diag_path, diag_row)
                self._crop_file_paths.setdefault(_crop_key, []).append(diag_path)
            except Exception:
                pass

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
            if victim_sim > 0.70 and time_diff < 3000:
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
        victim_key = victim_name.lower().strip() if victim_name and victim_name != "Unknown" else None
        
        if victim_key:
            if victim_key in self._victim_last_death:
                last_death_time, last_killer = self._victim_last_death[victim_key]
                time_since_death = t_ms - last_death_time
                
                if time_since_death < self.ROUND_DEDUP_WINDOW_MS:
                    killer_key = killer_name.lower().strip() if killer_name else ""
                    killer_sim = self._name_similarity(killer_name, last_killer)
                    
                    if killer_sim > 0.5:
                        return True
                    
                    if time_since_death < 60000:
                        return True
        
        # ==== TIER 2: Standard short-term dedup ====
        for (sig_t, sig_kt, sig_vt, sig_kn, sig_vn) in self.recent_signatures:
            time_diff = t_ms - sig_t
            
            if time_diff > KILL_DEDUP_WINDOW_MS:
                continue
            
            killer_sim = self._name_similarity(killer_name, sig_kn)
            victim_sim = self._name_similarity(victim_name, sig_vn)
            
            if victim_sim > 0.7 and killer_sim > 0.7:
                if time_diff < 3000:
                    return True
            
            if sig_kt == killer_team and sig_vt == victim_team:
                if killer_sim > 0.7 and victim_sim > 0.7:
                    return True
                elif killer_sim > 0.5 and victim_sim > 0.5:
                    if time_diff < 4000:
                        return True
                elif (killer_name == "Unknown" and victim_sim > 0.6) or \
                     (victim_name == "Unknown" and killer_sim > 0.6):
                    if time_diff < 3000:
                        return True
            
            killer_as_victim = self._name_similarity(killer_name, sig_vn)
            victim_as_killer = self._name_similarity(victim_name, sig_kn)
            
            if killer_as_victim > 0.6 and victim_as_killer > 0.6:
                if time_diff < 4000:
                    return True
            
            if time_diff < 1500:
                if killer_sim > 0.8 and victim_sim > 0.8:
                    return True
            
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
        
        n1 = self._normalize_player_name(name1).lower().strip()
        n2 = self._normalize_player_name(name2).lower().strip()
        
        if n1 == n2:
            return 1.0
        
        len_ratio = min(len(n1), len(n2)) / max(len(n1), len(n2), 1)
        if (n1 in n2 or n2 in n1) and len_ratio > 0.6:
            return 0.85
        
        common = sum(1 for c in n1 if c in n2)
        return common / max(len(n1), len(n2), 1)
    
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
        garbage_prefixes = ['ndc ', 'nde ', 'nid ', 'nide ', 'noc ', 'noe ', 'iv ', 'tip ']
        for prefix in garbage_prefixes:
            if name_lower.startswith(prefix):
                name_stripped = name_stripped[len(prefix):].strip()
                name_lower = name_stripped.lower()
                break
        
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
        
        if len(name_stripped) > 25:
            return "Unknown"
        
        words = name_lower.split()
        if len(words) >= 4:
            word_counts: dict = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            if word_counts:
                max_freq = max(word_counts.values())
                if max_freq > len(words) * 0.4:
                    return "Unknown"
        
        player_part = name_stripped
        for prefix in ['nrg ', 'fnc ', 'fng ', 'nag ', 'npg ', 'fne ']:
            if name_lower.startswith(prefix):
                player_part = name_stripped[4:]
                break
        
        if len(player_part) < 3:
            return "Unknown"
        if player_part.isdigit():
            return "Unknown"
        if not any(c.isalpha() for c in player_part):
            return "Unknown"
        alpha_count = sum(1 for c in player_part if c.isalpha())
        if alpha_count < len(player_part) * 0.5:
            return "Unknown"
        
        if self._player_matcher:
            db_match, extracted_team = self._player_matcher.match_killfeed_name(name_stripped)
            if db_match:
                if extracted_team:
                    return f"{extracted_team} {db_match}"
                team_side = self._player_matcher.get_player_team(db_match)
                if team_side == "left" and self._player_matcher._left_team_code:
                    return f"{self._player_matcher._left_team_code} {db_match}"
                elif team_side == "right" and self._player_matcher._right_team_code:
                    return f"{self._player_matcher._right_team_code} {db_match}"
                return db_match
        
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
        common = sum(1 for c in n1 if c in n2)
        return common / max(len(n1), len(n2)) > 0.6
    
    def _segment_rows(self, roi_bgr: np.ndarray) -> List[Tuple[int, int, np.ndarray]]:
        """Segment killfeed into individual rows."""
        h, w = roi_bgr.shape[:2]
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        
        # Create combined mask for team colors
        teal_mask = cv2.inRange(hsv, 
                                np.array(TEAM_COLORS["teal"]["lower"]),
                                np.array(TEAM_COLORS["teal"]["upper"]))
        orange_mask = _build_orange_mask(hsv)
        
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
        
        # Self-kill detection: A single-color row is valid if the colour
        # appears in TWO separate horizontal blobs (killer bg + victim bg)
        # with a visible gap in between (the weapon-icon zone).
        # Relaxed thresholds to catch spike and fall damage icons which may
        # be narrower than standard weapon icons.
        SELF_KILL_MIN_SINGLE_COLOR = 150
        SELF_KILL_MIN_GAP_PX = 10
        
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
            orange_mask = _build_orange_mask(hsv)
            
            teal_pixels = cv2.countNonZero(teal_mask)
            orange_pixels = cv2.countNonZero(orange_mask)
            
            # ---- DEBUG self-kill window ----
            _dbg = getattr(self, '_dbg_self', False)
            if _dbg:
                print(f"[DBG-SEG] row={i} teal={teal_pixels} orange={orange_pixels} total={total_pixels}")
            
            # Calculate color density (fraction of row covered by team colors)
            color_density = (teal_pixels + orange_pixels) / total_pixels
            row_color_density[i] = color_density
            
            # Normal kill: BOTH teal and orange present
            # (killer name = one color, victim name = other color)
            majority_pixels = max(teal_pixels, orange_pixels)
            minority_pixels = min(teal_pixels, orange_pixels)
            
            has_both_colors = (
                majority_pixels > MIN_COLOR_PIXELS_PRIMARY and 
                minority_pixels > MIN_COLOR_PIXELS_MINORITY
            )
            
            # Self-kill / fall-damage: Only ONE team colour, but it must
            # appear in two separate horizontal regions with a gap.
            has_single_color_two_blobs = False
            if not has_both_colors and majority_pixels >= SELF_KILL_MIN_SINGLE_COLOR:
                dominant_mask = teal_mask if teal_pixels >= orange_pixels else orange_mask
                regions = self._find_color_regions(dominant_mask)
                if _dbg:
                    print(f"[DBG-SEG] row={i} self-kill check: regions={len(regions)} bboxes={[(r[0],r[2]) for r in regions]}")
                if len(regions) >= 2:
                    regions_sorted = sorted(regions, key=lambda r: r[0])  # sort by x
                    leftmost = regions_sorted[0]
                    rightmost = regions_sorted[-1]
                    gap = rightmost[0] - (leftmost[0] + leftmost[2])  # x2_start - x1_end
                    if _dbg:
                        print(f"[DBG-SEG] row={i} gap={gap} (leftmost x={leftmost[0]} w={leftmost[2]}, rightmost x={rightmost[0]})")
                    if gap >= SELF_KILL_MIN_GAP_PX:
                        has_single_color_two_blobs = True
            
            if _dbg:
                print(f"[DBG-SEG] row={i} has_both={has_both_colors} has_single_two_blobs={has_single_color_two_blobs} -> {'CONTENT' if has_both_colors or has_single_color_two_blobs else 'EMPTY'}")
            
            if has_both_colors or has_single_color_two_blobs:
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

    def _ocr_killfeed_row(self, row_img: np.ndarray, row_width: int) -> List[Dict[str, Any]]:
        """Run OCR on a killfeed row image and return sorted name entries.
        
        Returns a list of dicts: [{name, x, x_left, x_right, conf}, ...]
        sorted by x position (left to right).
        """
        scale = 2  # All OCR preprocessing uses 2x scale

        if hasattr(self._ocr_reader, 'read_text_multipass'):
            multipass_results = self._ocr_reader.read_text_multipass(
                row_img,
                min_confidence=0.2,
                strategies=['contrast']
            )
            results = [(r.bbox, r.text, r.confidence) for r in multipass_results]
        elif hasattr(self._ocr_reader, 'read_text'):
            scaled = cv2.resize(row_img, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_LINEAR)
            ocr_results = self._ocr_reader.read_text(scaled, min_confidence=0.3)
            results = [(r.bbox, r.text, r.confidence) for r in ocr_results]
        else:
            scaled = cv2.resize(row_img, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_LINEAR)
            results = self._ocr_reader.readtext(scaled, paragraph=False)

        names = []
        for bbox, text, conf in results:
            if conf > 0.2 and len(text.strip()) >= 2:
                if isinstance(bbox, tuple) and len(bbox) == 4:
                    x_center = (bbox[0] + bbox[2] / 2) / scale
                    x_left = bbox[0] / scale
                    x_right = (bbox[0] + bbox[2]) / scale
                else:
                    x_center = (bbox[0][0] + bbox[2][0]) / 2 / scale
                    x_left = min(bbox[0][0], bbox[3][0]) / scale
                    x_right = max(bbox[1][0], bbox[2][0]) / scale
                names.append({
                    "name": text.strip(),
                    "x": x_center,
                    "x_left": x_left,
                    "x_right": x_right,
                    "conf": conf,
                })

        names.sort(key=lambda n: n["x"])
        return names

    def _parse_row(self, row_img: np.ndarray) -> Optional[Dict[str, Any]]:
        """Parse a killfeed row to extract kill information.
        
        Handles both normal kills (two different team colours) and
        self-kills / fall damage (same colour on both sides).
        """
        h, w = row_img.shape[:2]
        hsv = cv2.cvtColor(row_img, cv2.COLOR_BGR2HSV)
        
        # Detect team colors (use full orange mask including secondary hue range)
        teal_mask = cv2.inRange(hsv,
                                np.array(TEAM_COLORS["teal"]["lower"]),
                                np.array(TEAM_COLORS["teal"]["upper"]))
        orange_mask = _build_orange_mask(hsv)
        
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
        # For self-kills (fall damage), both sides are the same colour.
        killer_team = all_regions[0]["color"]
        victim_team = all_regions[-1]["color"]
        
        # Detect same-colour layout (self-kill / fall damage / spike)
        is_same_color = (killer_team == victim_team)

        # Try OCR for names
        killer_name = "Unknown"
        victim_name = "Unknown"
        names = []  # Will be populated by OCR; needed later for text boundaries
        
        self._init_ocr()
        if self._ocr_reader:
            try:
                # Same OCR strategy for all rows (including self-kills).
                # Self-kill rows look identical to normal kills in the killfeed:
                #   [killer_name] [icon] [victim_name]
                # with the same player on both sides and both backgrounds the same colour.
                names = self._ocr_killfeed_row(row_img, w)

                if len(names) >= 2:
                    killer_name = names[0]["name"]
                    victim_name = names[-1]["name"]
                elif len(names) == 1:
                    # Only 1 name found - assign based on position
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
        
        # Compute text boundaries for weapon icon extraction
        # killer_text_right = right edge of killer (leftmost) text bbox
        # victim_text_left = left edge of victim (rightmost) text bbox
        # Since names is sorted by x, names[0] is killer and names[-1] is victim.
        #
        # Filter out spurious OCR detections near row edges (agent icons can
        # produce short text fragments that shift the boundary assignment).
        # Only consider OCR results whose center is within 10%-90% of the row.
        killer_text_right = None
        victim_text_left = None
        if len(names) >= 2:
            inner_names = [n for n in names if w * 0.10 < n["x"] < w * 0.90]
            if len(inner_names) >= 2:
                killer_text_right = inner_names[0].get("x_right")
                victim_text_left = inner_names[-1].get("x_left")
                # Cap killer bbox width: PaddleOCR sometimes merges the
                # ability icon / weapon pixels into the killer name bbox,
                # inflating x_right far beyond the actual text.  Normal
                # killer name bboxes are 60-100px wide; the longest
                # name ("ENVY Eggsterr") reaches ~108px.  Cap at 100.
                ktr_xl = inner_names[0].get("x_left", 0)
                ktr_w = killer_text_right - ktr_xl
                MAX_NAME_W = 100
                if ktr_w > MAX_NAME_W:
                    killer_text_right = ktr_xl + MAX_NAME_W
            else:
                # Fall back to unfiltered if filtering removed too many
                killer_text_right = names[0].get("x_right")
                victim_text_left = names[-1].get("x_left")
            # Log all OCR boxes so we can see when PaddleOCR merges
            # weapon/icon pixels into a name bbox
            crop_num = getattr(self, '_crop_counter', 0)
            used = inner_names if len(inner_names) >= 2 else names
            for i, n in enumerate(used):
                print(f"[OCR-BOX] crop#{crop_num} box{i}: '{n['name']}' xL={n['x_left']:.0f} xR={n['x_right']:.0f} w={n['x_right']-n['x_left']:.0f} conf={n['conf']:.2f}", flush=True)
        
        return {
            "killer_name": killer_name,
            "killer_team": killer_team,  # Raw color: teal or orange
            "victim_name": victim_name,
            "victim_team": victim_team,  # Raw color: teal or orange
            "weapon": "unknown",
            # weapon_icon is a numpy image (BGR) cropped around the icon area
            # This is kept for in-process classification only and not JSON-serializable.
            # Use `set_weapon_classifier()` to attach a classifier and populate `weapon`.
            "weapon_icon": None,
            "is_headshot": False,
            "confidence": 0.7 if killer_name != "Unknown" and victim_name != "Unknown" else 0.4,
            # Text boundaries for weapon icon extraction (pixel coords in row_img)
            "killer_text_right": killer_text_right,
            "victim_text_left": victim_text_left,
        }

    def set_weapon_classifier(self, classifier: object):
        """
        Attach a weapon/ability classifier to the KillfeedDetector.

        The classifier should expose one of the following methods to perform
        inference on a BGR numpy image: `classify(img)`, `predict(img)`, or
        `infer(img)` and return a string label (e.g., 'vandal' or 'jet_ult').
        This method only stores the classifier reference; loading/initializing
        the model is the caller's responsibility.
        """
        self._weapon_classifier = classifier

    def _classify_weapon(self, icon_img: np.ndarray) -> str:
        """
        Classify a cropped weapon/ability icon using the attached classifier.
        Returns label string or 'unknown' on failure.
        """
        if icon_img is None:
            return "unknown"
        clf = getattr(self, '_weapon_classifier', None)
        if clf is None:
            return "unknown"

        # Try a few common method names to be flexible
        for method in ('classify', 'predict', 'infer'):
            fn = getattr(clf, method, None)
            if callable(fn):
                try:
                    lbl = fn(icon_img)
                    if isinstance(lbl, (list, tuple)):
                        lbl = lbl[0]
                    return str(lbl)
                except Exception:
                    continue
        # If classifier exposes a `__call__`, try that
        if callable(clf):
            try:
                lbl = clf(icon_img)
                if isinstance(lbl, (list, tuple)):
                    lbl = lbl[0]
                return str(lbl)
            except Exception:
                return "unknown"

        return "unknown"

    def _extract_weapon_icon(self, row_img: np.ndarray, killer_text_right: float = None, victim_text_left: float = None) -> Optional[np.ndarray]:
        """
        Extract the weapon/ability icon from a killfeed row image.

        Killfeed row structure:
          [Agent] [Killer Name on colored bg] [WEAPON ICON] [arrow] [Victim on colored bg] [Agent]

        Priority order:
        1. OCR + bright-pixel refinement (search zone around OCR gap)
        2. Threshold-based contour detection (center zone, bright pixels)
        3. OCR trim fallback (raw OCR gap with inward trim)
        4. Center fallback (conservative center crop)
        """
        try:
            h, w = row_img.shape[:2]
            if w < 60 or h < 10:
                return None

            # Track actual crop bounds for diagnostic overlay
            self._last_crop_bounds = None
            self._last_search_zone = None
            self._last_crop_method = None

            MIN_CROP_W = max(38, h)  # minimum crop width = row height or 38px
            crop_num = getattr(self, '_crop_counter', 0)

            # ── Strategy 1: OCR + bright-pixel refinement ──
            if killer_text_right is None or victim_text_left is None:
                ktr_s = f"{killer_text_right:.1f}" if killer_text_right is not None else "None"
                vtl_s = f"{victim_text_left:.1f}" if victim_text_left is not None else "None"
                print(f"[CROP-DBG] crop#{crop_num} SKIP Strategy1: kR={ktr_s} vL={vtl_s} (missing OCR bounds)")
            if killer_text_right is not None and victim_text_left is not None:
                left_bound = int(round(killer_text_right))
                right_bound = int(round(victim_text_left))
                gap = right_bound - left_bound

                # ── Position sanity check ──
                # Real killfeed rows sit on the right side of the
                # screen.  kR < 30% of row width indicates the OCR
                # read text from a non-killfeed element (e.g. replay
                # economy overlay, character splash screen).
                if left_bound < w * 0.30:
                    print(f"[CROP-DBG] crop#{crop_num} SKIP: kR={left_bound} < {w*0.30:.0f} (position too far left)")
                    return None

                # ── Ability icon detection ──
                # Ability icons are small squares (~h×h px).  When
                # PaddleOCR merges the icon into the killer name bbox
                # the gap shrinks to < h+15.  Use a vL-anchored crop
                # instead of the normal Strategy 1 pipeline which
                # would pick up the killer agent icon.
                if 0 < gap < h + 15:
                    icon_w = int(h * 1.05)
                    # Pull right edge inward — the ability icon sits
                    # a few px left of vL, not flush against it.
                    icon_x1 = right_bound - 4
                    icon_x0 = max(0, icon_x1 - icon_w)
                    pad_l = max(4, int(h * 0.10))
                    pad_r = 1
                    x0 = max(0, icon_x0 - pad_l)
                    x1 = min(w, icon_x1 + pad_r)
                    if (x1 - x0) < MIN_CROP_W:
                        mid = (x0 + x1) // 2
                        x0 = max(0, mid - MIN_CROP_W // 2)
                        x1 = min(w, x0 + MIN_CROP_W)
                    self._last_crop_bounds = (x0, x1)
                    self._last_crop_method = "ocr_hybrid"
                    print(f"[CROP-DBG] crop#{crop_num} ability-icon crop (gap={gap}): x0={x0} x1={x1} w={x1-x0}")
                    icon = row_img[0:h, x0:x1]
                    if icon.size > 0:
                        return icon

                # ── ktr color-boundary correction ──
                # PaddleOCR sometimes merges a small ability icon into
                # the killer name bbox, pushing ktr too far right.
                # Detect this by checking if ktr pixels are team-colored;
                # if not, scan leftward to find where team color ends.
                # Only attempt when gap is non-trivial (>= 20px).
                ktr_skip_bright = False  # Fix 11: track when weapon icon sits at kR
                if gap >= 20:
                    hsv_row = cv2.cvtColor(row_img, cv2.COLOR_BGR2HSV)
                    teal_m = cv2.inRange(hsv_row, np.array([75, 50, 80]), np.array([115, 255, 255]))
                    red_m1 = cv2.inRange(hsv_row, np.array([0, 80, 100]), np.array([15, 255, 255]))
                    red_m2 = cv2.inRange(hsv_row, np.array([165, 80, 100]), np.array([179, 255, 255]))
                    team_color_mask = cv2.bitwise_or(teal_m, cv2.bitwise_or(red_m1, red_m2))
                    strip_x0 = max(0, left_bound - 1)
                    strip_x1 = min(w, left_bound + 2)
                    strip = team_color_mask[:, strip_x0:strip_x1]
                    color_frac = np.count_nonzero(strip) / max(1, strip.size)
                    # Brightness/saturation pre-check: if kR pixels are bright
                    # and unsaturated, they belong to a weapon icon (white/gray),
                    # not a gap between name and icon.  Skip ktr correction to
                    # avoid scanning left through the entire weapon icon.
                    hsv_strip = hsv_row[:, strip_x0:strip_x1]
                    med_v = float(np.median(hsv_strip[:, :, 2]))
                    med_s = float(np.median(hsv_strip[:, :, 1]))
                    bright_unsaturated = (med_v > 170 and med_s < 60)
                    if bright_unsaturated:
                        ktr_skip_bright = True
                        print(f"[CROP-DBG] crop#{crop_num} ktr skip correction: bright weapon icon at kR (V={med_v:.0f} S={med_s:.0f})")
                    elif color_frac < 0.45:
                        # Cap correction distance: large corrections
                        # land on same-color text from different rows.
                        max_correction = max(50, int(gap * 0.40))
                        scan_limit = max(int(w * 0.10), left_bound - max_correction)
                        original_left_bound = left_bound
                        new_ktr = left_bound
                        for sx in range(left_bound - 1, scan_limit - 1, -1):
                            col_strip = team_color_mask[:, sx:sx+1]
                            if np.count_nonzero(col_strip) / max(1, col_strip.size) >= 0.35:
                                new_ktr = sx + 1
                                break
                        if new_ktr < left_bound:
                            left_bound = new_ktr
                            gap = right_bound - left_bound
                            # Guard: if correction made gap too large, the
                            # scan hit team color from a different element
                            # (e.g. agent icon, different row). Revert.
                            if gap > w * 0.38:
                                print(f"[CROP-DBG] crop#{crop_num} ktr correction {original_left_bound} -> {new_ktr} REVERTED (gap={gap} > {w*0.38:.0f})")
                                left_bound = original_left_bound
                                gap = right_bound - left_bound
                            else:
                                print(f"[CROP-DBG] crop#{crop_num} ktr corrected {original_left_bound} -> {new_ktr} (color boundary)")
                    elif gap < 50:
                        # Team color extends through the weapon/ability
                        # icon area (same bg as killer name).  Color scan
                        # cannot find a boundary, so push ktr left by a
                        # fixed amount to give Strategy 1 a wider zone.
                        original_left_bound = left_bound
                        push = max(120, int(w * 0.20))
                        left_bound = max(int(w * 0.10), left_bound - push)
                        gap = right_bound - left_bound
                        print(f"[CROP-DBG] crop#{crop_num} ktr merged-icon push {original_left_bound} -> {left_bound} (gap={gap}, team-color extends through icon)")
                else:
                    hsv_row = None

                print(f"[CROP-DBG] crop#{crop_num} OCR: kR={killer_text_right:.1f} vL={victim_text_left:.1f} gap={gap} w={w}")

                if gap >= 30 and gap <= w * 0.38:
                    # Expand search zone generously left to catch icons
                    # absorbed into the killer name bbox
                    pass  # gap in range — proceed with Strategy 1
                elif gap < 30:
                    print(f"[CROP-DBG] crop#{crop_num} SKIP Strategy1: gap={gap} < 30")
                else:
                    print(f"[CROP-DBG] crop#{crop_num} SKIP Strategy1: gap={gap} > {w*0.38:.0f} (w*0.38)")

                if gap >= 30 and gap <= w * 0.38:
                    expand_right = max(10, int(gap * 0.10))
                    # For very large gaps the gun sits fully inside
                    # kR..vL — don't search left of kR to avoid
                    # killer-name text on bright/yellow backgrounds.
                    if gap > 140:
                        search_x0 = max(int(w * 0.12), left_bound)
                    else:
                        expand_left = max(20, int(gap * 0.35))
                        search_x0 = max(int(w * 0.12), left_bound - expand_left)
                    search_x1 = min(int(w * 0.85), right_bound + expand_right)
                    self._last_search_zone = (search_x0, search_x1)

                    if hsv_row is None:
                        hsv_row = cv2.cvtColor(row_img, cv2.COLOR_BGR2HSV)

                    # Grayscale threshold within the OCR search zone.
                    # Mask out high-saturation pixels first — on yellow /
                    # team-colored backgrounds, bright background pixels
                    # pass the gray>180 threshold and create giant
                    # contours.  Icon pixels are white/gray (low sat).
                    gray = cv2.cvtColor(row_img, cv2.COLOR_BGR2GRAY)
                    _, thresh = cv2.threshold(
                        gray, 180, 255, cv2.THRESH_BINARY)
                    if hsv_row is None:
                        hsv_row = cv2.cvtColor(row_img, cv2.COLOR_BGR2HSV)
                    sat_mask = cv2.inRange(hsv_row[:, :, 1], 0, 90)
                    thresh = cv2.bitwise_and(thresh, sat_mask)
                    # Restrict to search zone
                    zone_mask = np.zeros_like(thresh)
                    zone_mask[0:h, search_x0:search_x1] = 255
                    thresh = cv2.bitwise_and(thresh, zone_mask)

                    contours_bp, _ = cv2.findContours(
                        thresh, cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE,
                    )

                    min_cnt_area = h * 2
                    bp_valid = []
                    for cnt in contours_bp:
                        area = cv2.contourArea(cnt)
                        if area >= min_cnt_area:
                            bx, by, bw, bh = cv2.boundingRect(cnt)
                            bp_valid.append((bx, by, bw, bh, area))

                    n_contours_total = len(bp_valid)
                    # Filter contours near the victim text — the headshot
                    # icon sits just left of vL and its contours must be
                    # excluded or they merge with the weapon cluster.
                    hs_margin = max(20, int(gap * 0.20))
                    if bp_valid:
                        bp_valid = [c for c in bp_valid
                                    if (c[0] + c[2] // 2) <= right_bound - hs_margin]
                    n_after_right = len(bp_valid)

                    # Also filter contours whose center is to the LEFT of
                    # left_bound (these are likely killer-name text pixels).
                    # Gap-dependent: for small gaps (<100) the gun fills
                    # most of the gap and extends left of kR, so be loose.
                    # For large gaps (>120) text leaks in, so be tight.
                    if gap <= 140:
                        left_filter_x = left_bound - max(15, int(gap * 0.20))
                    else:
                        left_filter_x = left_bound - max(5, int(gap * 0.04))
                    if bp_valid:
                        bp_valid = [c for c in bp_valid
                                    if (c[0] + c[2] // 2) >= left_filter_x]
                    n_after_left = len(bp_valid)

                    if not bp_valid:
                        print(f"[CROP-DBG] crop#{crop_num} Strategy1 brightness: contours {n_contours_total}->right_filt {n_after_right}->left_filt {n_after_left}")
                        # Fallback: saturation-based detection.
                        # On team-colored backgrounds the icon is
                        # low-saturation (white/gray) while the bg is
                        # high-saturation.  Threshold on inverted sat.
                        if hsv_row is None:
                            hsv_row = cv2.cvtColor(row_img, cv2.COLOR_BGR2HSV)
                        sat_ch = hsv_row[:, :, 1]
                        # Low saturation = text / icon pixels
                        _, sat_thresh = cv2.threshold(sat_ch, 70, 255, cv2.THRESH_BINARY_INV)
                        sat_thresh = cv2.bitwise_and(sat_thresh, zone_mask)
                        contours_sat, _ = cv2.findContours(
                            sat_thresh, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
                        for cnt in contours_sat:
                            area = cv2.contourArea(cnt)
                            if area >= min_cnt_area:
                                bx, by, bw, bh = cv2.boundingRect(cnt)
                                bp_valid.append((bx, by, bw, bh, area))
                        # Re-apply same right/left filters
                        if bp_valid:
                            bp_valid = [c for c in bp_valid
                                        if (c[0] + c[2] // 2) <= right_bound - hs_margin]
                        if bp_valid:
                            bp_valid = [c for c in bp_valid
                                        if (c[0] + c[2] // 2) >= left_filter_x]
                        if bp_valid:
                            print(f"[CROP-DBG] crop#{crop_num} saturation fallback found {len(bp_valid)} contours")
                        else:
                            print(f"[CROP-DBG] crop#{crop_num} SKIP Strategy1: no contours (brightness or saturation)")

                    if bp_valid:
                        gap_mid = (left_bound + right_bound) / 2
                        # Prefer contours near the LEFT side of the gap
                        # (weapon icon) over the right (headshot icon).
                        weapon_anchor = left_bound + gap * 0.35
                        def _score(c):
                            cx = c[0] + c[2] / 2
                            return (abs(cx - weapon_anchor), -c[4])
                        bp_valid.sort(key=_score)
                        best = bp_valid[0]
                        icon_x0 = best[0]
                        icon_x1 = best[0] + best[2]

                        cluster_gap = max(12, int(h * 0.40))
                        MAX_ICON_W = min(140, max(int(gap * 0.75), 80))
                        for c in bp_valid[1:]:
                            c_x0, c_x1 = c[0], c[0] + c[2]
                            if (c_x0 <= icon_x1 + cluster_gap and
                                    c_x1 >= icon_x0 - cluster_gap):
                                new_x0 = min(icon_x0, c_x0)
                                new_x1 = max(icon_x1, c_x1)
                                if (new_x1 - new_x0) > MAX_ICON_W:
                                    continue  # Skip: would make cluster too wide
                                icon_x0 = new_x0
                                icon_x1 = new_x1

                        # Clamp to OCR boundaries — slightly generous
                        # on the left (weapons extend left of ktr) and
                        # allow a small right overshoot (gun barrels can
                        # extend past vL).  The hs_margin filter already
                        # excluded headshot-icon contours from the cluster.
                        # Left margin: generous for small gaps (gun
                        # extends left), tight for large gaps (text leak).
                        if gap <= 140:
                            left_margin = max(15, int(gap * 0.20))
                        else:
                            left_margin = max(8, int(gap * 0.10))
                        right_margin = max(15, int(gap * 0.20))
                        icon_x0 = max(icon_x0, left_bound - left_margin)
                        icon_x1 = min(icon_x1, right_bound + right_margin)

                        # Headshot guard: prevent crop from extending
                        # into the headshot-icon zone near vL.  The
                        # hs_margin filtered contour *centers*, but a
                        # contour edge can still reach into hs territory.
                        hs_right_limit = right_bound - max(18, int(gap * 0.15))
                        if icon_x1 > hs_right_limit:
                            icon_x1 = hs_right_limit

                        # If the icon cluster is still wider than
                        # MAX_ICON_W (e.g. a single giant contour that
                        # merged weapon + headshot), trim from the right.
                        if (icon_x1 - icon_x0) > MAX_ICON_W:
                            icon_x1 = icon_x0 + MAX_ICON_W

                        pad = max(4, int(h * 0.15))
                        x0 = max(0, icon_x0 - pad)
                        x1 = min(w, icon_x1 + pad)

                        # Killer-text guard: the crop must never extend
                        # left of the ORIGINAL kR (pre-correction).
                        # The weapon icon lives between the two names;
                        # anything left of kR is killer name text.
                        # Fix 11: when ktr-skip detected a weapon icon at
                        # kR, the icon genuinely extends left of kR — use
                        # a relaxed margin (20px) instead of the strict 4px.
                        original_kR = int(round(killer_text_right))
                        guard_margin = 20 if ktr_skip_bright else 4
                        if x0 < original_kR - guard_margin:
                            print(f"[CROP-DBG] crop#{crop_num} left-clamp x0={x0} -> {original_kR - guard_margin} (killer text guard, kR={original_kR}, margin={guard_margin})")
                            x0 = original_kR - guard_margin

                        # Re-apply headshot guard after padding
                        if x1 > hs_right_limit + 2:
                            x1 = hs_right_limit + 2

                        if (x1 - x0) < MIN_CROP_W:
                            mid = (x0 + x1) // 2
                            x0 = max(0, mid - MIN_CROP_W // 2)
                            x1 = min(w, x0 + MIN_CROP_W)

                        self._last_crop_bounds = (x0, x1)
                        self._last_crop_method = "ocr_hybrid"
                        print(f"[CROP-DBG] crop#{crop_num} -> OCR+hybrid x0={x0} x1={x1} w={x1-x0} (icon {icon_x0}-{icon_x1})")
                        icon = row_img[0:h, x0:x1]
                        if icon.size > 0:
                            return icon

                    # ── Strategy 2: OCR trim fallback ──
                    inward_pct = 0.06
                    x0 = max(0, left_bound + int(gap * inward_pct))
                    x1 = min(w, right_bound - int(gap * inward_pct))
                    if (x1 - x0) < MIN_CROP_W:
                        mid = (x0 + x1) // 2
                        x0 = max(0, mid - MIN_CROP_W // 2)
                        x1 = min(w, x0 + MIN_CROP_W)
                    self._last_crop_bounds = (x0, x1)
                    self._last_crop_method = "ocr_trim"
                    print(f"[CROP-DBG] crop#{crop_num} -> OCR trim fallback x0={x0} x1={x1} w={x1-x0}")
                    icon = row_img[0:h, x0:x1]
                    if icon.size > 0:
                        return icon

            # ── Strategy 3: Threshold-based contour detection ──
            gray = cv2.cvtColor(row_img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
            contours_th, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours_th:
                center_zone_left = int(w * 0.18)
                center_zone_right = int(w * 0.82)
                min_area = h * 3

                valid_contours = []
                for cnt in contours_th:
                    bx, by, bw, bh = cv2.boundingRect(cnt)
                    cx = bx + bw // 2
                    area = cv2.contourArea(cnt)
                    if center_zone_left <= cx <= center_zone_right and area >= min_area:
                        valid_contours.append((bx, by, bw, bh, area))

                if valid_contours:
                    all_left = min(c[0] for c in valid_contours)
                    all_right = max(c[0] + c[2] for c in valid_contours)

                    pad = max(4, int(h * 0.15))
                    x0 = max(0, all_left - pad)
                    x1 = min(w, all_right + pad)

                    crop_w = x1 - x0
                    MAX_CROP = int(w * 0.30)

                    if crop_w < MIN_CROP_W:
                        mid = (x0 + x1) // 2
                        x0 = max(0, mid - MIN_CROP_W // 2)
                        x1 = min(w, x0 + MIN_CROP_W)
                    elif crop_w > MAX_CROP:
                        valid_contours.sort(key=lambda c: c[4], reverse=True)
                        bx, by, bw, bh, _ = valid_contours[0]
                        x0 = max(0, bx - pad)
                        x1 = min(w, bx + bw + pad)
                        if (x1 - x0) < MIN_CROP_W:
                            mid = (x0 + x1) // 2
                            x0 = max(0, mid - MIN_CROP_W // 2)
                            x1 = min(w, x0 + MIN_CROP_W)

                    self._last_crop_bounds = (x0, x1)
                    self._last_crop_method = "threshold"
                    print(f"[CROP-DBG] crop#{crop_num} -> threshold x0={x0} x1={x1} w={x1-x0}")
                    icon = row_img[0:h, x0:x1]
                    if icon.size > 0:
                        return icon

            # ── Strategy 4: Center fallback ──
            self._last_crop_method = "center"
            print(f"[CROP-DBG] crop#{crop_num} -> center fallback")
            return self._center_fallback_crop(row_img)

        except Exception:
            return None

    def _center_fallback_crop(self, row_img: np.ndarray) -> Optional[np.ndarray]:
        """Fallback: return a conservative center crop when gap detection fails."""
        h, w = row_img.shape[:2]
        crop_w = int(min(80, max(h, w * 0.15)))  # at least row height
        cx = w // 2
        x0 = max(0, cx - crop_w // 2)
        x1 = min(w, x0 + crop_w)
        self._last_crop_bounds = (x0, x1)
        icon = row_img[0:h, x0:x1]
        return icon if icon.size > 0 else None

    def _maybe_extract_ult_badge(
        self,
        row_img: np.ndarray,
        gap_left: float,
        gap_right: float,
    ) -> Optional[np.ndarray]:
        """Detect and extract an ultimate ability badge.

        Analyses the region between gap_left (typically the weapon icon's
        right edge) and gap_right (victim_text_left or an estimate).
        If the victim team colour covers >= 11 % of this region, an ult
        badge is present.  Crops from the first victim-colour column to
        gap_right.
        """
        self._last_ult_badge_bounds = None  # reset each call
        try:
            h, w = row_img.shape[:2]
            gap_x0 = int(round(gap_left))
            gap_x1 = int(round(gap_right))
            gap = gap_x1 - gap_x0
            if gap < 40 or h < 10:
                return None

            # Trim bottom to avoid background bleed diluting colours
            trim_bot = max(3, int(h * 0.20))
            gap_region = row_img[0:h - trim_bot, gap_x0:gap_x1]
            gh, gw = gap_region.shape[:2]
            if gh < 4 or gw < 20:
                return None

            hsv = cv2.cvtColor(gap_region, cv2.COLOR_BGR2HSV)

            # Detect teal pixels
            teal_mask = cv2.inRange(
                hsv,
                np.array([75, 50, 80]),
                np.array([115, 255, 255]),
            )
            # Detect red/orange pixels — two hue ranges
            red_mask1 = cv2.inRange(
                hsv,
                np.array([0, 120, 140]),
                np.array([10, 255, 255]),
            )
            red_mask2 = cv2.inRange(
                hsv,
                np.array([170, 120, 140]),
                np.array([179, 255, 255]),
            )
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)

            total_pixels = gh * gw
            teal_pct = cv2.countNonZero(teal_mask) / total_pixels
            red_pct = cv2.countNonZero(red_mask) / total_pixels

            # Determine killer colour (dominant) and victim colour (minority).
            # Normal crops: only the killer colour appears.  With an ult badge
            # the victim colour also shows up at >= 20 %.
            if teal_pct >= red_pct:
                killer_pct, victim_pct = teal_pct, red_pct
                victim_mask = red_mask
            else:
                killer_pct, victim_pct = red_pct, teal_pct
                victim_mask = teal_mask

            # Collect diagnostics for offline analysis
            contours_v, _ = cv2.findContours(
                victim_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            largest_area = max(cv2.contourArea(c) for c in contours_v) if contours_v else 0

            # Brightness check: real ult badges have a visible weapon icon
            # (bright/white pixels) alongside the victim color.  False
            # positives are just team-colored background with little brightness.
            gray_gap = cv2.cvtColor(gap_region, cv2.COLOR_BGR2GRAY)
            _, bright_mask = cv2.threshold(gray_gap, 180, 255, cv2.THRESH_BINARY)
            bright_pct = cv2.countNonZero(bright_mask) / total_pixels

            crop_num = getattr(self, '_crop_counter', 0)
            self._ult_diagnostics.append({
                "crop": crop_num + 1,
                "gap_x0": gap_x0, "gap_x1": gap_x1, "gap_w": gap,
                "teal_pct": round(teal_pct, 4),
                "red_pct": round(red_pct, 4),
                "killer_pct": round(killer_pct, 4),
                "victim_pct": round(victim_pct, 4),
                "largest_blob": int(largest_area),
                "bright_pct": round(bright_pct, 4),
                "detected": victim_pct >= 0.15 and killer_pct < 0.58 and largest_area >= 150 and bright_pct >= 0.20,
            })

            if victim_pct < 0.15:
                return None

            # Killer-colour ceiling: false positives from headshot icons
            # have killer_pct > 60% because the region is almost entirely
            # killer team colour.  Real ult badges split the region,
            # pulling killer_pct down to ~40-53%.
            if killer_pct >= 0.58:
                return None

            # Contiguity check — require a substantial blob, not scatter.
            if not contours_v:
                return None
            if largest_area < 150:          # reject sparse noise
                return None

            # Brightness gate: reject if gap region lacks bright weapon pixels
            if bright_pct < 0.20:
                return None

            # Find the leftmost x in the gap where victim colour appears
            # to determine where the badge region starts.
            victim_cols = np.where(victim_mask.any(axis=0))[0]
            if len(victim_cols) == 0:
                return None

            # Badge region: from the first victim-colour column to the
            # right edge of the gap, in full-height row coordinates.
            badge_x0_row = gap_x0 + int(victim_cols[0])
            badge_x1_row = gap_x1

            # Enforce minimum badge width (at least row-height square)
            badge_w = badge_x1_row - badge_x0_row
            if badge_w < h:
                badge_x0_row = max(gap_x0, badge_x1_row - h)

            badge_crop = row_img[0:h, badge_x0_row:badge_x1_row]
            if badge_crop.size == 0:
                return None

            # Store badge bounds (absolute row coordinates) for diag overlay
            self._last_ult_badge_bounds = (badge_x0_row, badge_x1_row)

            crop_num = getattr(self, '_crop_counter', 0)
            print(f"[CROP-DBG] crop#{crop_num} ULT BADGE detected (killer={killer_pct:.1%} victim={victim_pct:.1%}) -> badge x={badge_x0_row}-{badge_x1_row} w={badge_x1_row - badge_x0_row}")
            return badge_crop

        except Exception:
            return None

    def _find_color_regions(self, mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Find contiguous color regions in a mask."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions = []
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            if cw > 8 and ch > 4:
                regions.append((x, y, cw, ch))
        return regions


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
    
    MAX_REALISTIC_SCORE = 20
    HALFTIME_TOTAL_ROUNDS = 12
    
    def __init__(self, roi_name: str, target_fps: float):
        super().__init__(roi_name, target_fps)
        self._spike_planted = False
        self._confirmed_left_score = 0
        self._confirmed_right_score = 0
        self._last_score_change_ms = 0
        self._round_count = 0
        self._score_ocr_reader = None
        self._last_valid_score_ms = 0
        self._consecutive_invalid_frames = 0
        self._score_stability_start_ms = 0
        self._SCORE_STABLE_THRESHOLD_MS = 3000
        self._in_halftime = False
        self._halftime_listeners: list = []
        self._ROUND_DEBOUNCE_MS = 5000
        self._zero_zero_seen = False
    
    def has_confirmed_zero_zero(self) -> bool:
        """Returns True once a 0-0 score has been seen on screen via OCR."""
        return self._zero_zero_seen
    
    def add_halftime_listener(self, callback):
        """Add a callback to be notified of halftime state changes."""
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
        """Extract a score number (0-20) from a score ROI using EasyOCR."""
        try:
            if score_roi is None or score_roi.size == 0:
                return -1, 0.0
            ocr = self._get_score_ocr_reader()
            if ocr is None:
                return -1, 0.0
            h, w = score_roi.shape[:2]
            candidates = []
            scaled = cv2.resize(score_roi, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
            results = ocr.readtext(scaled, allowlist='0123456789')
            if results:
                candidates.append((results[0][1], results[0][2], 'scaled'))
            gray = cv2.cvtColor(score_roi, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
            thresh_scaled = cv2.resize(thresh, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
            results = ocr.readtext(thresh_scaled, allowlist='0123456789')
            if results:
                candidates.append((results[0][1], results[0][2], 'thresh'))
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(2, 2))
            enhanced = clahe.apply(gray)
            enhanced_scaled = cv2.resize(enhanced, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
            results = ocr.readtext(enhanced_scaled, allowlist='0123456789')
            if results:
                candidates.append((results[0][1], results[0][2], 'clahe'))
            best_score = -1
            best_conf = 0.0
            for text, conf, method in candidates:
                if conf >= 0.4:
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
        except Exception:
            return -1, 0.0

    def _detect(self, t_ms: float, roi_frame: np.ndarray) -> List[Event]:
        """Detect round transitions by reading scores with direct EasyOCR."""
        events = []
        h, w = roi_frame.shape[:2]
        top_hud_x, top_hud_y = 0.335, 0.005
        top_hud_w, top_hud_h = 0.330, 0.200
        # Read score ROI coordinates from ROI_CONFIG (settings.py) instead of hardcoding
        ls = ROI_CONFIG.get("top_left_score", (0.417, 0.009, 0.036, 0.055))
        rs = ROI_CONFIG.get("top_right_score", (0.555, 0.009, 0.036, 0.055))
        left_norm_x = (ls[0] - top_hud_x) / top_hud_w
        left_norm_y = (ls[1] - top_hud_y) / top_hud_h
        left_norm_w = ls[2] / top_hud_w
        left_norm_h = ls[3] / top_hud_h
        right_norm_x = (rs[0] - top_hud_x) / top_hud_w
        right_norm_y = (rs[1] - top_hud_y) / top_hud_h
        right_norm_w = rs[2] / top_hud_w
        right_norm_h = rs[3] / top_hud_h
        left_x = int(left_norm_x * w)
        left_y = int(left_norm_y * h)
        score_w = max(int(left_norm_w * w), 40)
        score_h = max(int(left_norm_h * h), 30)
        right_x = int(right_norm_x * w)
        right_y = int(right_norm_y * h)
        left_x = max(0, min(left_x, w - score_w))
        right_x = max(0, min(right_x, w - score_w))
        left_y = max(0, min(left_y, h - score_h))
        right_y = max(0, min(right_y, h - score_h))
        left_roi = roi_frame[left_y:left_y+score_h, left_x:left_x+score_w]
        right_roi = roi_frame[right_y:right_y+score_h, right_x:right_x+score_w]
        left_score, left_conf = self._extract_score(left_roi)
        right_score, right_conf = self._extract_score(right_roi)
        score_visible = left_score >= 0 and right_score >= 0 and left_conf >= 0.5 and right_conf >= 0.5
        if score_visible:
            # Track when we first see a 0-0 score (match start)
            if not self._zero_zero_seen and left_score == 0 and right_score == 0:
                self._zero_zero_seen = True
                print(f"[TopHUD] Match start detected: 0-0 score confirmed at t={t_ms/1000:.1f}s", flush=True)
            self._consecutive_invalid_frames = 0
            self._last_valid_score_ms = t_ms
            current_total = left_score + right_score
            if self._in_halftime:
                halftime_min_duration_ms = 30000
                time_in_halftime = t_ms - self._halftime_start_ms if hasattr(self, '_halftime_start_ms') else 0
                pre_halftime_left = self._confirmed_left_score
                pre_halftime_right = self._confirmed_right_score

                def is_valid_post_halftime_score(left, right, pre_left, pre_right):
                    if left < pre_left or right < pre_right:
                        return False
                    total_increase = (left + right) - (pre_left + pre_right)
                    if total_increase > 5:
                        return False
                    return True

                is_valid_score = is_valid_post_halftime_score(left_score, right_score, pre_halftime_left, pre_halftime_right)
                should_end_halftime = False
                if is_valid_score and time_in_halftime > halftime_min_duration_ms:
                    if current_total > self.HALFTIME_TOTAL_ROUNDS:
                        should_end_halftime = True
                    elif current_total == self.HALFTIME_TOTAL_ROUNDS:
                        should_end_halftime = True
                elif not is_valid_score:
                    self._score_stability_start_ms = 0
                if should_end_halftime:
                    if self._score_stability_start_ms == 0:
                        self._score_stability_start_ms = t_ms
                    elif t_ms - self._score_stability_start_ms >= self._SCORE_STABLE_THRESHOLD_MS:
                        self._in_halftime = False
                        print(f"[TopHUD] Halftime ended - stable score {left_score}-{right_score} detected at t={t_ms/1000:.1f}s")
                        for callback in self._halftime_listeners:
                            callback(False, t_ms)
        else:
            self._consecutive_invalid_frames += 1
            self._score_stability_start_ms = 0
            confirmed_total = self._confirmed_left_score + self._confirmed_right_score
            if confirmed_total == self.HALFTIME_TOTAL_ROUNDS and not self._in_halftime:
                if self._consecutive_invalid_frames >= 5:
                    self._in_halftime = True
                    self._halftime_start_ms = t_ms
                    print(f"[TopHUD] Halftime started - score {self._confirmed_left_score}-{self._confirmed_right_score} no longer visible at t={t_ms/1000:.1f}s")
                    for callback in self._halftime_listeners:
                        callback(True, t_ms)

        if left_score >= 0 and right_score >= 0 and left_conf >= 0.5 and right_conf >= 0.5:
            if left_score != self._confirmed_left_score or right_score != self._confirmed_right_score:
                total_old = self._confirmed_left_score + self._confirmed_right_score
                total_new = left_score + right_score
                rounds_added = total_new - total_old
                if rounds_added == 1:
                    left_change = left_score - self._confirmed_left_score
                    right_change = right_score - self._confirmed_right_score
                    valid_transition = (
                        (left_change == 0 and right_change == 1) or
                        (left_change == 1 and right_change == 0)
                    )
                    if valid_transition:
                        time_since_last = t_ms - self._last_score_change_ms
                        if time_since_last > self._ROUND_DEBOUNCE_MS or self._last_score_change_ms == 0:
                            print(f"[ROUND] Score: {self._confirmed_left_score}-{self._confirmed_right_score} -> {left_score}-{right_score} at t={t_ms/1000:.1f}s (conf: L={left_conf:.2f}, R={right_conf:.2f})", flush=True)
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

        # Detect spike status
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
