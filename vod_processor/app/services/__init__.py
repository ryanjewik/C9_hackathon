"""
VOD Processing Services

This package contains the core services for processing VOD files:
- frame_sampler: Generator-based frame extraction with rolling buffer
- vod_processor: Main orchestrator for frame extraction and event detection
- pipeline: Full orchestration integrating all components per architecture.md
- job_manager: Manages background processing jobs
- player_tracker: Tracks player states across frames
- minimap_tracker: Tracks player positions on the minimap
- database: Interfaces with the esports database for player/team data
- state_resolver: Fuses parser outputs and enforces game rules
- timeline_builder: Builds the final timeline output
- player_name_extractor: Extracts player names from HUD and matches OCR results
"""

from app.services.frame_sampler import FrameSampler, Frame
from app.services.job_manager import JobManager, Job
from app.services.vod_processor import VODProcessor
from app.services.pipeline import VODPipeline
from app.services.player_tracker import PlayerStateTracker, PlayerCardAnalyzer
from app.services.minimap_tracker import MinimapTracker
from app.services.database import EsportsDatabase, Player, Team, Match, PlayerGame, GameScore, Tournament
from app.services.state_resolver import StateResolver, ResolvedEvent, EventType, GameState, PlayerState
from app.services.timeline_builder import TimelineBuilder, Timeline, RoundSummary, TimelineMetadata
from app.services.player_name_extractor import PlayerNameExtractor, DatabasePlayerMatcher

__all__ = [
    "FrameSampler",
    "Frame",
    "JobManager",
    "Job",
    "VODProcessor",
    "VODPipeline",
    "PlayerStateTracker",
    "PlayerCardAnalyzer",
    "MinimapTracker",
    "EsportsDatabase",
    "Player",
    "Team",
    "Match",
    "PlayerGame",
    "GameScore",
    "Tournament",
    "StateResolver",
    "ResolvedEvent",
    "EventType",
    "GameState",
    "PlayerState",
    "TimelineBuilder",
    "Timeline",
    "RoundSummary",
    "TimelineMetadata",
    "PlayerNameExtractor",
    "DatabasePlayerMatcher",
]
