"""
VOD Processing Pipeline - Full Orchestration

This module integrates all components from architecture.md:
- Frame Sampler (Section 4)
- Killfeed Parser (Section 5)
- Player Panel Parser (Section 6)
- Minimap Tracker (Section 8)
- State Resolver (Section 10)
- Timeline Builder (Section 11)
"""

import os
import json
from typing import Dict, List, Optional, Any
from datetime import datetime

import cv2
import numpy as np

from app.schemas import JobStatus
from app.services.frame_sampler import FrameSampler, Frame
from app.services.vod_processor import (
    VODProcessor, 
    KillfeedDetector, 
    TopHUDDetector, 
    MinimapDetector,
    Event,
    roi_to_px,
    crop,
)
from app.services.state_resolver import StateResolver, EventType
from app.services.timeline_builder import TimelineBuilder
from app.services.database import EsportsDatabase
from app.services.player_tracker import PlayerStateTracker
from app.services.minimap_tracker import MinimapTracker as PositionTracker
from config import get_settings, ROI_CONFIG, DETECTOR_FPS


class VODPipeline:
    """
    Full VOD processing pipeline following architecture.md.
    
    Integrates:
    - Frame sampling at 10-15 FPS
    - Parallel HUD parsers (killfeed, player panel, minimap)
    - State resolution with game rules
    - Timeline building
    - Database integration for player matching
    """
    
    def __init__(self, db_connection_string: Optional[str] = None):
        self.settings = get_settings()
        self._job_manager = None
        
        # Database for player name matching
        self.db: Optional[EsportsDatabase] = None
        if db_connection_string:
            self.db = EsportsDatabase(db_connection_string)
    
    def set_job_manager(self, job_manager):
        """Set the job manager for status updates."""
        self._job_manager = job_manager
    
    def process(
        self,
        job_id: str,
        video_path: str,
        output_dir: str,
        match_id: Optional[str] = None,
        match_players: Optional[List[str]] = None,
        map_name: Optional[str] = None,
        team_a: Optional[str] = None,
        team_b: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a VOD file using the full pipeline.
        
        Args:
            job_id: Unique job identifier
            video_path: Path to the VOD file
            output_dir: Directory to save output files
            match_id: Optional match ID for database lookup
            match_players: Optional list of player names (filters OCR results)
            map_name: Optional map name
            team_a: Team A name
            team_b: Team B name
            
        Returns:
            Processing results dictionary
        """
        self._update_status(job_id, JobStatus.PROCESSING, "Initializing pipeline...")
        
        try:
            # Initialize frame sampler
            sampler = FrameSampler(
                video_path=video_path,
                base_fps=self.settings.frame_sample_fps,  # 10-15 FPS per architecture
                buffer_size=int(self.settings.frame_sample_fps * 5),  # 5 second buffer
            )
            
            if not sampler.open():
                raise ValueError(f"Could not open video: {video_path}")
            
            print(f"[{job_id}] Video: {sampler.width}x{sampler.height} @ {sampler.video_fps:.2f}fps")
            print(f"[{job_id}] Duration: {sampler.duration:.2f}s, Sampling at ~{self.settings.frame_sample_fps} fps")
            
            # Pre-compute ROI pixel coordinates
            roi_px_cache = {
                name: roi_to_px(sampler.width, sampler.height, roi_norm)
                for name, roi_norm in ROI_CONFIG.items()
            }
            
            # Initialize detectors (parallel HUD parsers)
            killfeed_detector = KillfeedDetector("killfeed", DETECTOR_FPS["killfeed"])
            top_hud_detector = TopHUDDetector("top_hud", DETECTOR_FPS["top_hud"])
            minimap_detector = MinimapDetector("minimap", DETECTOR_FPS["minimap"])
            
            # Initialize player matcher for killfeed team detection
            # Team A is typically left side (teal), Team B is right side (orange)
            from app.services.player_name_extractor import DatabasePlayerMatcher
            player_matcher = DatabasePlayerMatcher()
            
            # Parse player list (handles "NRG:p1,p2;FNC:p3,p4" format)
            parsed_players = self._parse_player_list(match_players)
            
            # Set up player matcher with known players
            # First 5 players = left team (team_a), next 5 = right team (team_b)
            if parsed_players:
                left_players = parsed_players[:5] if len(parsed_players) >= 5 else parsed_players
                right_players = parsed_players[5:10] if len(parsed_players) > 5 else []
                player_matcher.set_match_players(left_players, right_players)
                print(f"[{job_id}] Set match players: left={left_players}, right={right_players}")
            
            # Set team codes for killfeed detector to use for team color override
            print(f"[{job_id}] Setting player matcher with team_a={team_a}, team_b={team_b}")
            killfeed_detector.set_player_matcher(player_matcher, team_a, team_b)
            
            # Initialize state resolver (game engine)
            state_resolver = StateResolver()
            
            # Initialize player tracker for state management
            player_tracker = PlayerStateTracker()
            
            # Initialize minimap position tracker with Kalman filtering
            position_tracker = PositionTracker()
            
            # Load known players from database if available
            if self.db and match_id:
                known_players = self._load_match_players(match_id)
                if known_players:
                    state_resolver.initialize_players(known_players)
                    print(f"[{job_id}] Loaded {len(known_players)} players from database")
            elif parsed_players:
                # Use provided player list
                self._init_players_from_parsed_list(state_resolver, parsed_players, team_a, team_b)
            
            # Processing loop
            all_events: List[Event] = []
            frame_count = 0
            last_progress_update = 0
            
            self._update_status(job_id, JobStatus.PROCESSING, "Processing frames...")
            
            for frame in sampler.frame_stream():
                t_ms = frame.timestamp * 1000  # Convert to milliseconds
                
                # Run killfeed detector
                if "killfeed" in roi_px_cache:
                    kf_crop = crop(frame.image, roi_px_cache["killfeed"])
                    if kf_crop.size > 0:
                        kf_events = killfeed_detector.process(t_ms, kf_crop)
                        all_events.extend(kf_events)
                        
                        # Feed kills to state resolver
                        for event in kf_events:
                            if event.type == "KILL_EVENT":
                                raw_killer = event.payload.get("killer_name", "")
                                raw_victim = event.payload.get("victim_name", "")
                                killer = self._match_player_name(raw_killer, parsed_players)
                                victim = self._match_player_name(raw_victim, parsed_players)
                                
                                # Debug logging for name matching
                                if 'kaajak' in raw_killer.lower() or 'kaajak' in killer.lower():
                                    import warnings
                                    warnings.warn(f"[DEBUG KAAJAK KILL] raw_killer='{raw_killer}' -> matched='{killer}', raw_victim='{raw_victim}' -> victim='{victim}'")
                                
                                state_resolver.process_kill_event(
                                    timestamp=frame.timestamp,
                                    killer=killer,
                                    victim=victim,
                                    weapon=event.payload.get("weapon", "unknown"),
                                    confidence=event.confidence,
                                )
                
                # Run top HUD detector
                if "top_hud" in roi_px_cache:
                    hud_crop = crop(frame.image, roi_px_cache["top_hud"])
                    if hud_crop.size > 0:
                        hud_events = top_hud_detector.process(t_ms, hud_crop)
                        all_events.extend(hud_events)
                        
                        # Feed round transitions to state resolver and killfeed detector
                        for event in hud_events:
                            if event.type == "ROUND_TRANSITION":
                                # Detect round number from score/timer
                                round_num = state_resolver.current_state.round_number + 1
                                state_resolver.process_round_start(frame.timestamp, round_num)
                                # Notify killfeed detector of new round - clears per-round death tracking
                                killfeed_detector.set_round_start(t_ms)
                            elif event.type == "SPIKE_PLANTED":
                                state_resolver.process_spike_plant(frame.timestamp)
                
                # Run minimap detector
                if "minimap" in roi_px_cache:
                    mm_crop = crop(frame.image, roi_px_cache["minimap"])
                    if mm_crop.size > 0:
                        mm_events = minimap_detector.process(t_ms, mm_crop)
                        all_events.extend(mm_events)
                        
                        # Feed positions to tracker and state resolver
                        try:
                            for event in mm_events:
                                if event.type == "MINIMAP_PLAYER_CHANGE":
                                    positions = event.payload.get("positions", [])
                                    for pos in positions:
                                        # Track position using process() method
                                        position_tracker.process(t_ms, mm_crop)
                        except Exception as mm_err:
                            # Skip minimap tracking errors - focus on kill detection
                            if frame_count <= 1:
                                print(f"[Pipeline] Minimap tracking disabled: {mm_err}")
                
                frame_count += 1
                
                # Update progress periodically
                if sampler.progress - last_progress_update >= 1.0:
                    self._update_progress(
                        job_id,
                        processed_frames=frame_count,
                        total_frames=sampler.total_frames // (sampler._sample_interval or 1),
                        events_detected=len(all_events)
                    )
                    last_progress_update = sampler.progress
            
            sampler.close()
            
            # Build timeline using TimelineBuilder
            self._update_status(job_id, JobStatus.PROCESSING, "Building timeline...")
            
            timeline_builder = TimelineBuilder(state_resolver)
            timeline_builder.set_metadata(
                vod_id=job_id,
                match_id=match_id,
                map_name=map_name,
                team_a=team_a,
                team_b=team_b,
                duration=sampler.duration,
            )
            
            timeline = timeline_builder.build()
            
            # Validate consistency
            issues = state_resolver.validate_consistency()
            if issues:
                print(f"[{job_id}] Validation warnings: {len(issues)} issues found")
                for issue in issues[:5]:
                    print(f"  - {issue['type']}: {issue}")
            
            # Save outputs
            os.makedirs(output_dir, exist_ok=True)
            
            # Save raw events
            events_path = os.path.join(output_dir, f"{job_id}_events.json")
            with open(events_path, "w") as f:
                json.dump([self._event_to_dict(e) for e in all_events], f, indent=2)
            
            # Save timeline (primary output)
            timeline_path = os.path.join(output_dir, f"{job_id}_timeline.json")
            with open(timeline_path, "w") as f:
                f.write(timeline.to_json())
            
            # Save stats summary
            stats = timeline_builder.get_stats_summary()
            stats_path = os.path.join(output_dir, f"{job_id}_stats.json")
            with open(stats_path, "w") as f:
                json.dump(stats, f, indent=2)
            
            # Save validation issues
            if issues:
                issues_path = os.path.join(output_dir, f"{job_id}_issues.json")
                with open(issues_path, "w") as f:
                    json.dump(issues, f, indent=2)
            
            # Update job status
            self._update_status(
                job_id, JobStatus.COMPLETED,
                f"Processing complete. {len(all_events)} events, {len(timeline.rounds)} rounds."
            )
            self._add_output_files(job_id, [events_path, timeline_path, stats_path])
            
            return {
                "job_id": job_id,
                "status": "completed",
                "events_count": len(all_events),
                "rounds_count": len(timeline.rounds),
                "kills_count": len([e for e in all_events if e.type == "KILL_EVENT"]),
                "duration_seconds": sampler.duration,
                "confidence_stats": timeline.confidence_stats,
                "validation_issues": len(issues),
                "output_files": [events_path, timeline_path, stats_path],
            }
            
        except Exception as e:
            import traceback
            error_msg = f"Processing failed: {str(e)}"
            print(f"[{job_id}] ERROR: {error_msg}")
            traceback.print_exc()
            
            self._update_status(job_id, JobStatus.FAILED, error_msg)
            
            return {
                "job_id": job_id,
                "status": "failed",
                "error": str(e)
            }
    
    def _load_match_players(self, match_id: str) -> List[Dict[str, Any]]:
        """Load players from database for a specific match."""
        if not self.db:
            return []
        
        # This would query the esports_matches, esports_rosters, esports_players tables
        # For now, return empty - needs implementation with actual match lookup
        return []
    
    def _init_players_from_list(
        self,
        resolver: StateResolver,
        players: List[str],
        team_a: Optional[str],
        team_b: Optional[str],
    ):
        """
        Initialize player list from provided names.
        DEPRECATED: Use _parse_player_list + _init_players_from_parsed_list
        """
        parsed = self._parse_player_list(players)
        self._init_players_from_parsed_list(resolver, parsed, team_a, team_b)
    
    def _parse_player_list(self, players: Optional[List[str]]) -> List[str]:
        """
        Parse player list, handling various formats.
        
        Supports:
        - ["NRG:Ethan,Brawk,Mada,Skuba,s0m;FNC:Boaster,Alfajer,Chronicle,Kajaak,Crashies"]
        - ["Ethan", "Brawk", "Mada", "Skuba", "s0m", "Boaster", ...]
        
        Returns:
            List of individual player names
        """
        if not players:
            return []
        
        # Handle single string with semicolons (team format)
        if len(players) == 1 and ';' in players[0]:
            print(f"[Pipeline] Parsing team format: {players[0][:60]}...")
            teams_data = players[0].split(';')
            parsed_players = []
            
            for i, team_data in enumerate(teams_data[:2]):  # Only take first 2 teams
                if ':' in team_data:
                    team_name, players_str = team_data.split(':', 1)
                    team_players = [p.strip() for p in players_str.split(',') if p.strip()]
                else:
                    team_players = [p.strip() for p in team_data.split(',') if p.strip()]
                
                parsed_players.extend(team_players)
                print(f"[Pipeline] Team {i+1}: {team_players}")
            
            return parsed_players
        
        # Already a list of individual names
        return [p.strip() for p in players if p and p.strip()]
    
    def _init_players_from_parsed_list(
        self,
        resolver: StateResolver,
        players: List[str],
        team_a: Optional[str],
        team_b: Optional[str],
    ):
        """Initialize state resolver with parsed player list."""
        # Initialize player dicts (first 5 = team A/attacker, next 5 = team B/defender)
        player_dicts = []
        for i, name in enumerate(players[:10]):  # Max 10 players
            player_dicts.append({
                "player_id": name,
                "team": "attacker" if i < 5 else "defender",
                "agent": "unknown",
            })
        
        print(f"[Pipeline] Initialized {len(player_dicts)} players")
        resolver.initialize_players(player_dicts)
    
    def _match_player_name(
        self,
        ocr_name: str,
        known_players: Optional[List[str]],
    ) -> str:
        """Match OCR result to known player name using fuzzy matching.
        
        Priority order (check ALL players before moving to next tier):
        1. Exact match (case-insensitive)
        2. Substring match
        3. OCR-normalized match
        4. Character overlap match (>70%)
        """
        if not ocr_name or ocr_name == "Unknown":
            return ocr_name
        
        if not known_players:
            return ocr_name
        
        # Use database fuzzy matching if available
        if self.db:
            matched = self.db.match_player_name(ocr_name, known_players)
            if 'kaajak' in ocr_name.lower():
                import warnings
                warnings.warn(f"[DEBUG _match_player_name DB] ocr_name='{ocr_name}' -> db_matched='{matched}'")
            return matched if matched else ocr_name
        
        # Extract just the player name (remove team prefix like "NRG ")
        name_parts = ocr_name.split()
        name_only = name_parts[-1] if name_parts else ocr_name
        ocr_lower = name_only.lower().strip()
        
        # TIER 1: Check ALL players for exact match FIRST
        for player in known_players:
            player_lower = player.lower().strip()
            if ocr_lower == player_lower:
                if 'kaajak' in ocr_name.lower():
                    import warnings
                    warnings.warn(f"[DEBUG _match_player_name LOCAL] ocr_name='{ocr_name}' exact match -> '{player}'")
                return player
        
        # TIER 2: Check ALL players for substring match
        for player in known_players:
            player_lower = player.lower().strip()
            if ocr_lower in player_lower or player_lower in ocr_lower:
                if 'kaajak' in ocr_name.lower():
                    import warnings
                    warnings.warn(f"[DEBUG _match_player_name LOCAL] ocr_name='{ocr_name}' substring match -> '{player}'")
                return player
        
        # TIER 3: Check ALL players for OCR-normalized match
        ocr_norm = self._ocr_normalize(ocr_lower)
        for player in known_players:
            player_lower = player.lower().strip()
            player_norm = self._ocr_normalize(player_lower)
            if ocr_norm == player_norm:
                if 'kaajak' in ocr_name.lower():
                    import warnings
                    warnings.warn(f"[DEBUG _match_player_name LOCAL] ocr_name='{ocr_name}' ocr-norm match -> '{player}'")
                return player
        
        # TIER 4: Check ALL players for character overlap match
        # Find the BEST overlap match, not just the first one above threshold
        best_match = None
        best_ratio = 0.0
        for player in known_players:
            player_lower = player.lower().strip()
            common = sum(1 for c in ocr_lower if c in player_lower)
            ratio = common / max(len(ocr_lower), len(player_lower))
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = player
        
        if best_ratio > 0.7 and best_match:
            if 'kaajak' in ocr_name.lower():
                import warnings
                warnings.warn(f"[DEBUG _match_player_name LOCAL] ocr_name='{ocr_name}' overlap match (ratio={best_ratio:.2f}) -> '{best_match}'")
            return best_match
        
        if 'kaajak' in ocr_name.lower():
            import warnings
            warnings.warn(f"[DEBUG _match_player_name LOCAL] ocr_name='{ocr_name}' NO MATCH, returning as-is")
        return ocr_name
    
    def _ocr_normalize(self, s: str) -> str:
        """Normalize string for OCR comparison (handle common OCR confusions)."""
        import re
        s = s.lower()
        # 0/O/o/U/u confusion - OCR often reads 0 as o or u
        s = re.sub(r'[0ouv]', 'o', s)
        # 1/l/I/| confusion
        s = re.sub(r'[1il|]', 'l', s)
        # 5/s/S confusion
        s = re.sub(r'[5]', 's', s)
        return s
    
    def _event_to_dict(self, event: Event) -> Dict[str, Any]:
        """Convert Event to dictionary."""
        return {
            "t_ms": event.t_ms,
            "type": event.type,
            "roi": event.roi,
            "payload": event.payload,
            "confidence": event.confidence,
        }
    
    def _update_status(self, job_id: str, status: JobStatus, message: str):
        """Update job status if job manager is available."""
        if self._job_manager:
            self._job_manager.update_job_status(job_id, status, message)
        print(f"[{job_id}] {status.value}: {message}")
    
    def _update_progress(self, job_id: str, **kwargs):
        """Update job progress if job manager is available."""
        if self._job_manager:
            self._job_manager.update_progress(job_id, **kwargs)
    
    def _add_output_files(self, job_id: str, files: List[str]):
        """Add output files to job if job manager is available."""
        if self._job_manager:
            for f in files:
                self._job_manager.add_output_file(job_id, f)
