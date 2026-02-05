"""
Pydantic schemas for API requests and responses.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Processing job status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    version: str


class JobResponse(BaseModel):
    """Response for job operations."""
    job_id: str
    status: JobStatus
    message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    # Detailed progress info: processed frames, total frames, events detected
    class ProgressInfo(BaseModel):
        processed_frames: int = 0
        total_frames: int = 0
        events_detected: int = 0

    progress: Optional[ProgressInfo] = None
    error: Optional[str] = None


class Event(BaseModel):
    """A single game event."""
    t_ms: float = Field(..., description="Timestamp in milliseconds")
    type: str = Field(..., description="Event type")
    roi: Optional[str] = Field(None, description="Region of Interest that detected this event")
    payload: Dict[str, Any] = Field(default_factory=dict)
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    round_number: Optional[int] = None


class KillEvent(BaseModel):
    """Kill event details."""
    killer_name: str
    killer_team: str
    victim_name: str
    victim_team: str
    weapon: Optional[str] = None
    is_headshot: bool = False
    killer_position: Optional[List[float]] = None
    victim_position: Optional[List[float]] = None
    assisters: List[str] = Field(default_factory=list)


class PlayerState(BaseModel):
    """Player state at a given timestamp."""
    timestamp_ms: float
    player_id: str
    player_name: str
    team: str
    agent: Optional[str] = None
    alive: bool = True
    health: Optional[int] = None
    armor: Optional[int] = None
    position: Optional[List[float]] = None
    ability_1_available: Optional[bool] = None
    ability_2_available: Optional[bool] = None
    ability_3_available: Optional[bool] = None
    ultimate_ready: Optional[bool] = None
    ultimate_points: Optional[int] = None


class RoundData(BaseModel):
    """Data for a single round."""
    round_number: int
    start_time_ms: float
    end_time_ms: Optional[float] = None
    winning_team: Optional[str] = None
    win_condition: Optional[str] = None  # "elimination", "spike_detonation", "spike_defuse", "time"
    spike_planted: bool = False
    spike_plant_time_ms: Optional[float] = None
    spike_defused: bool = False
    spike_defuse_time_ms: Optional[float] = None
    kills: List[Dict[str, Any]] = Field(default_factory=list)
    events: List[Dict[str, Any]] = Field(default_factory=list)


class MatchMetadata(BaseModel):
    """Match metadata."""
    vod_id: str
    filename: str
    duration_ms: float
    resolution: Optional[List[int]] = None
    fps: Optional[float] = None
    map_name: Optional[str] = None
    teams: List[str] = Field(default_factory=list)
    players: List[Dict[str, Any]] = Field(default_factory=list)
    total_rounds: Optional[int] = None
    final_score: Optional[List[int]] = None


class TimelineResponse(BaseModel):
    """Full timeline response."""
    metadata: MatchMetadata
    events: List[Event] = Field(default_factory=list)
    rounds_with_kills: List[Dict[str, Any]] = Field(default_factory=list)  # Round-organized kills
    player_states: List[PlayerState] = Field(default_factory=list)
    kill_summary: Optional[Dict[str, Any]] = None
    kill_timeline: List[Dict[str, Any]] = Field(default_factory=list)  # Flat timeline


class EventsResponse(BaseModel):
    """Events list response."""
    job_id: str
    total_events: int
    filtered_events: int
    events: List[Dict[str, Any]]


class RoundEventsResponse(BaseModel):
    """Events for a specific round."""
    job_id: str
    round_number: int
    start_time_ms: float
    end_time_ms: Optional[float] = None
    winning_team: Optional[str] = None
    win_condition: Optional[str] = None
    spike_planted: bool = False
    spike_plant_time_ms: Optional[float] = None
    kills: List[Dict[str, Any]] = Field(default_factory=list)
    events: List[Dict[str, Any]] = Field(default_factory=list)
