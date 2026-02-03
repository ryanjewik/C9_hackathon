"""
Timeline Builder - Aggregates resolved events into the final timeline output.
Based on architecture.md Section 11.
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime

from app.services.state_resolver import StateResolver, ResolvedEvent, EventType, GameState


@dataclass
class TimelineMetadata:
    """Metadata about the processed VOD."""
    vod_id: str
    match_id: Optional[str] = None
    map_name: Optional[str] = None
    tournament: Optional[str] = None
    team_a: Optional[str] = None
    team_b: Optional[str] = None
    date: Optional[str] = None
    duration_seconds: float = 0
    total_rounds: int = 0
    final_score: Optional[Dict[str, int]] = None
    processing_timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    processor_version: str = "1.0.0"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "vod_id": self.vod_id,
            "match_id": self.match_id,
            "map_name": self.map_name,
            "tournament": self.tournament,
            "teams": {
                "team_a": self.team_a,
                "team_b": self.team_b,
            },
            "date": self.date,
            "duration_seconds": self.duration_seconds,
            "total_rounds": self.total_rounds,
            "final_score": self.final_score,
            "processing": {
                "timestamp": self.processing_timestamp,
                "version": self.processor_version,
            },
        }


@dataclass
class RoundSummary:
    """Summary of a single round."""
    round_number: int
    start_time: float
    end_time: float
    winning_team: Optional[str] = None
    win_reason: Optional[str] = None
    kills: List[Dict] = field(default_factory=list)
    spike_planted: bool = False
    spike_plant_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_number": self.round_number,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "winning_team": self.winning_team,
            "win_reason": self.win_reason,
            "kills": self.kills,
            "spike": {
                "planted": self.spike_planted,
                "plant_time": self.spike_plant_time,
            } if self.spike_planted else None,
        }


@dataclass
class Timeline:
    """Complete timeline output."""
    metadata: TimelineMetadata
    rounds: List[RoundSummary]
    events: List[Dict]
    player_roster: Dict[str, Dict]
    confidence_stats: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "player_roster": self.player_roster,
            "rounds": [r.to_dict() for r in self.rounds],
            "events": self.events,
            "confidence_stats": self.confidence_stats,
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class TimelineBuilder:
    """
    Builds the final timeline output from resolved events.
    
    Responsibilities:
    - Aggregate resolved events
    - Organize by round
    - Compute statistics
    - Format for output
    """
    
    def __init__(self, state_resolver: StateResolver):
        """
        Initialize the timeline builder.
        
        Args:
            state_resolver: The state resolver containing processed events
        """
        self.resolver = state_resolver
        self.metadata = TimelineMetadata(vod_id="")
        self.rounds: List[RoundSummary] = []
        self.confidence_scores: List[float] = []
    
    def set_metadata(
        self,
        vod_id: str,
        match_id: Optional[str] = None,
        map_name: Optional[str] = None,
        tournament: Optional[str] = None,
        team_a: Optional[str] = None,
        team_b: Optional[str] = None,
        date: Optional[str] = None,
        duration: float = 0,
    ):
        """Set timeline metadata."""
        self.metadata = TimelineMetadata(
            vod_id=vod_id,
            match_id=match_id,
            map_name=map_name,
            tournament=tournament,
            team_a=team_a,
            team_b=team_b,
            date=date,
            duration_seconds=duration,
        )
    
    def build(self) -> Timeline:
        """
        Build the complete timeline from the state resolver.
        
        Returns:
            Complete Timeline object
        """
        # Organize events by round
        self._build_rounds()
        
        # Get all events formatted
        events = self._format_events()
        
        # Build player roster
        roster = self._build_roster()
        
        # Calculate confidence statistics
        confidence_stats = self._calculate_confidence_stats()
        
        # Update metadata
        self.metadata.total_rounds = len(self.rounds)
        if self.rounds:
            self.metadata.final_score = {
                "attacker": self.resolver.current_state.attacker_score,
                "defender": self.resolver.current_state.defender_score,
            }
        
        return Timeline(
            metadata=self.metadata,
            rounds=self.rounds,
            events=events,
            player_roster=roster,
            confidence_stats=confidence_stats,
        )
    
    def _build_rounds(self):
        """Build round summaries from events."""
        self.rounds = []
        
        # Get round start/end events
        round_starts = [
            e for e in self.resolver.event_history
            if e.event_type == EventType.ROUND_START
        ]
        round_ends = [
            e for e in self.resolver.event_history
            if e.event_type == EventType.ROUND_END
        ]
        
        for start_event in round_starts:
            round_num = start_event.payload["round_number"]
            start_time = start_event.timestamp
            
            # Find matching end event
            end_event = next(
                (e for e in round_ends if e.payload["round_number"] == round_num),
                None
            )
            end_time = end_event.timestamp if end_event else start_time + 120
            
            # Get kills in this round
            round_kills = [
                e for e in self.resolver.event_history
                if e.event_type == EventType.KILL
                and start_time <= e.timestamp <= end_time
            ]
            
            # Get spike plant in this round
            spike_event = next(
                (e for e in self.resolver.event_history
                 if e.event_type == EventType.SPIKE_PLANT
                 and start_time <= e.timestamp <= end_time),
                None
            )
            
            round_summary = RoundSummary(
                round_number=round_num,
                start_time=start_time,
                end_time=end_time,
                winning_team=end_event.payload.get("winning_team") if end_event else None,
                win_reason=end_event.payload.get("reason") if end_event else None,
                kills=[self._format_kill_event(k) for k in round_kills],
                spike_planted=spike_event is not None,
                spike_plant_time=spike_event.timestamp if spike_event else None,
            )
            
            self.rounds.append(round_summary)
    
    def _format_events(self) -> List[Dict]:
        """Format all events for output."""
        events = []
        
        for event in sorted(self.resolver.event_history, key=lambda e: e.timestamp):
            formatted = event.to_dict()
            events.append(formatted)
            self.confidence_scores.append(event.confidence)
        
        return events
    
    def _format_kill_event(self, event: ResolvedEvent) -> Dict:
        """Format a kill event for the round summary."""
        return {
            "time": event.timestamp,
            "killer": event.payload["killer"],
            "victim": event.payload["victim"],
            "weapon": event.payload["weapon"],
            "positions": {
                "killer": event.payload.get("killer_position"),
                "victim": event.payload.get("victim_position"),
            },
            "confidence": event.confidence,
        }
    
    def _build_roster(self) -> Dict[str, Dict]:
        """Build player roster from state resolver."""
        roster = {}
        
        for player_id, player in self.resolver.current_state.players.items():
            roster[player_id] = {
                "team": player.team,
                "agent": player.agent,
            }
        
        # Add any additional info from match_players
        for player_id, info in self.resolver.match_players.items():
            if player_id in roster:
                roster[player_id].update({
                    k: v for k, v in info.items()
                    if k not in ["player_id"]
                })
        
        return roster
    
    def _calculate_confidence_stats(self) -> Dict[str, float]:
        """Calculate confidence statistics."""
        if not self.confidence_scores:
            return {"mean": 0, "min": 0, "max": 0}
        
        return {
            "mean": sum(self.confidence_scores) / len(self.confidence_scores),
            "min": min(self.confidence_scores),
            "max": max(self.confidence_scores),
            "total_events": len(self.confidence_scores),
        }
    
    def get_round_timeline(self, round_number: int) -> Optional[RoundSummary]:
        """Get the timeline for a specific round."""
        return next(
            (r for r in self.rounds if r.round_number == round_number),
            None
        )
    
    def get_events_by_type(self, event_type: str) -> List[Dict]:
        """Get all events of a specific type."""
        return [
            e.to_dict() for e in self.resolver.event_history
            if e.event_type.value == event_type
        ]
    
    def get_player_timeline(self, player_id: str) -> List[Dict]:
        """Get all events involving a specific player."""
        events = []
        
        for event in self.resolver.event_history:
            if event.event_type == EventType.KILL:
                if (event.payload.get("killer") == player_id or
                    event.payload.get("victim") == player_id):
                    events.append({
                        **event.to_dict(),
                        "player_role": "killer" if event.payload.get("killer") == player_id else "victim"
                    })
        
        return sorted(events, key=lambda e: e["timestamp"])
    
    def get_stats_summary(self) -> Dict[str, Any]:
        """Generate match statistics summary."""
        kills_by_player: Dict[str, int] = {}
        deaths_by_player: Dict[str, int] = {}
        weapons_used: Dict[str, int] = {}
        
        for event in self.resolver.event_history:
            if event.event_type == EventType.KILL:
                killer = event.payload.get("killer")
                victim = event.payload.get("victim")
                weapon = event.payload.get("weapon", "unknown")
                
                if killer:
                    kills_by_player[killer] = kills_by_player.get(killer, 0) + 1
                if victim:
                    deaths_by_player[victim] = deaths_by_player.get(victim, 0) + 1
                
                weapons_used[weapon] = weapons_used.get(weapon, 0) + 1
        
        return {
            "kills_by_player": kills_by_player,
            "deaths_by_player": deaths_by_player,
            "kd_ratios": {
                p: kills_by_player.get(p, 0) / max(1, deaths_by_player.get(p, 0))
                for p in set(kills_by_player.keys()) | set(deaths_by_player.keys())
            },
            "weapons_used": weapons_used,
            "total_kills": sum(kills_by_player.values()),
            "rounds_played": len(self.rounds),
            "spike_plants": sum(1 for r in self.rounds if r.spike_planted),
        }
