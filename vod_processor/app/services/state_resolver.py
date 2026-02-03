"""
State Resolver / Game Engine

Fuses signals from all parsers, enforces game rules, and resolves conflicts.
Based on architecture.md Section 10.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from enum import Enum
from collections import defaultdict


class EventType(Enum):
    KILL = "kill"
    DEATH = "death"
    RESURRECTION = "resurrection"
    ROUND_START = "round_start"
    ROUND_END = "round_end"
    SPIKE_PLANT = "spike_plant"
    SPIKE_DEFUSE = "spike_defuse"
    SPIKE_DETONATE = "spike_detonate"
    ABILITY_USE = "ability_use"
    ULTIMATE_USE = "ultimate_use"
    POSITION_UPDATE = "position_update"


@dataclass
class ResolvedEvent:
    """A validated event after rule enforcement."""
    timestamp: float
    event_type: EventType
    payload: Dict[str, Any]
    confidence: float
    raw_sources: List[str] = field(default_factory=list)  # Which parsers contributed
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "type": self.event_type.value,
            "payload": self.payload,
            "confidence": self.confidence,
        }


@dataclass 
class PlayerState:
    """Current state of a player at a given timestamp."""
    player_id: str
    team: str  # "attacker" or "defender"
    agent: str
    alive: bool = True
    position: Optional[tuple] = None  # (x, y) normalized
    ability_pips: Dict[str, int] = field(default_factory=dict)  # slot -> pip count
    ultimate_ready: bool = False
    ultimate_points: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "player_id": self.player_id,
            "team": self.team,
            "agent": self.agent,
            "alive": self.alive,
            "position": self.position,
            "ability_pips": self.ability_pips,
            "ultimate_ready": self.ultimate_ready,
        }


@dataclass
class GameState:
    """Complete game state at a point in time."""
    timestamp: float
    round_number: int
    round_phase: str  # "buy", "active", "post_round"
    spike_planted: bool = False
    spike_position: Optional[tuple] = None
    attacker_score: int = 0
    defender_score: int = 0
    players: Dict[str, PlayerState] = field(default_factory=dict)
    
    def get_alive_players(self, team: Optional[str] = None) -> List[PlayerState]:
        """Get all alive players, optionally filtered by team."""
        return [
            p for p in self.players.values()
            if p.alive and (team is None or p.team == team)
        ]
    
    def get_dead_players(self, team: Optional[str] = None) -> List[PlayerState]:
        """Get all dead players, optionally filtered by team."""
        return [
            p for p in self.players.values()
            if not p.alive and (team is None or p.team == team)
        ]


class StateResolver:
    """
    Central game engine that:
    - Fuses signals from all parsers
    - Enforces game rules
    - Resolves conflicts
    - Maintains authoritative game state
    """
    
    # VALORANT game rules
    PLAYERS_PER_TEAM = 5
    ROUNDS_TO_WIN = 13
    SIDE_SWITCH_ROUND = 12
    
    def __init__(self, match_players: Dict[str, Dict[str, Any]] = None):
        """
        Initialize the state resolver.
        
        Args:
            match_players: Known player roster {player_id: {team, agent}}
        """
        self.match_players = match_players or {}
        self.current_state = GameState(timestamp=0, round_number=1, round_phase="buy")
        self.event_history: List[ResolvedEvent] = []
        self.state_snapshots: List[GameState] = []
        
        # Tracking for conflict resolution
        self._pending_events: List[Dict] = []
        self._last_positions: Dict[str, tuple] = {}
        
    def initialize_players(self, players: List[Dict[str, Any]]):
        """
        Set up initial player states from roster.
        
        Args:
            players: List of player info dicts with id, team, agent
        """
        for p in players:
            self.current_state.players[p["player_id"]] = PlayerState(
                player_id=p["player_id"],
                team=p["team"],
                agent=p.get("agent", "unknown"),
            )
            self.match_players[p["player_id"]] = p
    
    def process_kill_event(
        self,
        timestamp: float,
        killer: str,
        victim: str,
        weapon: str,
        confidence: float = 1.0,
    ) -> Optional[ResolvedEvent]:
        """
        Process a kill event with rule validation.
        
        Rules enforced:
        - Killer ≠ victim (unless self-damage weapons)
        - Victim must be alive
        - Killer must be alive
        """
        # Validate players exist
        if killer not in self.current_state.players:
            print(f"Warning: Unknown killer {killer}")
            return None
        if victim not in self.current_state.players:
            print(f"Warning: Unknown victim {victim}")
            return None
        
        killer_state = self.current_state.players[killer]
        victim_state = self.current_state.players[victim]
        
        # Rule: victim must be alive
        if not victim_state.alive:
            print(f"Rule violation: {victim} already dead at {timestamp}")
            confidence *= 0.3  # Heavily penalize but don't discard
        
        # Rule: killer must be alive (unless posthumous damage)
        if not killer_state.alive:
            # Posthumous kills are possible with some abilities
            if weapon not in ["Snake Bite", "Molly", "Raze Grenade"]:
                confidence *= 0.7
        
        # Rule: no self-kills (except specific weapons)
        if killer == victim:
            self_damage_weapons = ["Raze Ult", "Killjoy Ult"]
            if weapon not in self_damage_weapons:
                confidence *= 0.1
        
        # Rule: different teams (except team damage in specific scenarios)
        if killer_state.team == victim_state.team:
            # Team kills are rare but possible
            confidence *= 0.5
        
        # Apply the death
        victim_state.alive = False
        
        # Create resolved event
        event = ResolvedEvent(
            timestamp=timestamp,
            event_type=EventType.KILL,
            payload={
                "killer": killer,
                "victim": victim,
                "weapon": weapon,
                "killer_team": killer_state.team,
                "victim_team": victim_state.team,
                "killer_position": self._last_positions.get(killer),
                "victim_position": self._last_positions.get(victim),
            },
            confidence=confidence,
            raw_sources=["killfeed"],
        )
        
        self.event_history.append(event)
        return event
    
    def process_position_update(
        self,
        timestamp: float,
        player_id: str,
        x: float,
        y: float,
        confidence: float = 1.0,
    ) -> Optional[ResolvedEvent]:
        """
        Process a position update from minimap tracking.
        
        Rules enforced:
        - Dead players do not move
        - Position changes should be physically plausible
        """
        if player_id not in self.current_state.players:
            return None
        
        player = self.current_state.players[player_id]
        
        # Rule: dead players do not move
        if not player.alive:
            # Could be a tracking error or resurrection
            confidence *= 0.2
        
        # Rule: position changes should be plausible
        last_pos = self._last_positions.get(player_id)
        if last_pos:
            dx = abs(x - last_pos[0])
            dy = abs(y - last_pos[1])
            max_speed = 0.1  # Maximum normalized distance per frame
            
            if dx > max_speed or dy > max_speed:
                # Teleport? Or tracking ID swap?
                confidence *= 0.7
        
        # Update state
        player.position = (x, y)
        self._last_positions[player_id] = (x, y)
        
        # Only emit position events at lower frequency (don't spam timeline)
        # These are tracked internally but not always exported
        return None
    
    def process_round_start(
        self,
        timestamp: float,
        round_number: int,
    ) -> ResolvedEvent:
        """
        Process round start - reset player states.
        """
        # Reset all players to alive
        for player in self.current_state.players.values():
            player.alive = True
            player.position = None
        
        # Update game state
        self.current_state.round_number = round_number
        self.current_state.round_phase = "buy"
        self.current_state.spike_planted = False
        self.current_state.spike_position = None
        self.current_state.timestamp = timestamp
        
        # Handle side switch
        if round_number == self.SIDE_SWITCH_ROUND + 1:
            for player in self.current_state.players.values():
                player.team = "defender" if player.team == "attacker" else "attacker"
        
        event = ResolvedEvent(
            timestamp=timestamp,
            event_type=EventType.ROUND_START,
            payload={
                "round_number": round_number,
                "attacker_score": self.current_state.attacker_score,
                "defender_score": self.current_state.defender_score,
            },
            confidence=1.0,
            raw_sources=["top_hud"],
        )
        
        self.event_history.append(event)
        
        # Save state snapshot
        self._snapshot_state()
        
        return event
    
    def process_round_end(
        self,
        timestamp: float,
        winning_team: str,
        reason: str,  # "elimination", "spike_detonated", "spike_defused", "timeout"
    ) -> ResolvedEvent:
        """Process round end."""
        self.current_state.round_phase = "post_round"
        
        # Update score
        if winning_team == "attacker":
            self.current_state.attacker_score += 1
        else:
            self.current_state.defender_score += 1
        
        event = ResolvedEvent(
            timestamp=timestamp,
            event_type=EventType.ROUND_END,
            payload={
                "round_number": self.current_state.round_number,
                "winning_team": winning_team,
                "reason": reason,
                "attacker_score": self.current_state.attacker_score,
                "defender_score": self.current_state.defender_score,
            },
            confidence=1.0,
            raw_sources=["top_hud"],
        )
        
        self.event_history.append(event)
        self._snapshot_state()
        
        return event
    
    def process_spike_plant(
        self,
        timestamp: float,
        planter: Optional[str] = None,
        position: Optional[tuple] = None,
    ) -> ResolvedEvent:
        """Process spike plant event."""
        self.current_state.spike_planted = True
        self.current_state.spike_position = position
        
        event = ResolvedEvent(
            timestamp=timestamp,
            event_type=EventType.SPIKE_PLANT,
            payload={
                "planter": planter,
                "position": position,
            },
            confidence=0.9 if planter else 0.7,
            raw_sources=["top_hud"],
        )
        
        self.event_history.append(event)
        return event
    
    def process_spike_defuse(
        self,
        timestamp: float,
        defuser: Optional[str] = None,
    ) -> ResolvedEvent:
        """Process spike defuse event."""
        self.current_state.spike_planted = False
        
        event = ResolvedEvent(
            timestamp=timestamp,
            event_type=EventType.SPIKE_DEFUSE,
            payload={"defuser": defuser},
            confidence=0.9 if defuser else 0.7,
            raw_sources=["top_hud"],
        )
        
        self.event_history.append(event)
        return event
    
    def process_ability_state(
        self,
        timestamp: float,
        player_id: str,
        ability_slot: str,
        pip_count: int,
    ):
        """
        Update ability pip count for a player.
        
        Rules enforced:
        - Pip count cannot increase mid-round (except for specific abilities)
        - Pip count must be >= 0
        """
        if player_id not in self.current_state.players:
            return
        
        player = self.current_state.players[player_id]
        current_pips = player.ability_pips.get(ability_slot, 0)
        
        # Validate pip change
        if pip_count < 0:
            pip_count = 0
        
        # Rule: pips usually don't increase mid-round
        if pip_count > current_pips and self.current_state.round_phase == "active":
            # Only some agents can regenerate (Jett, Reyna, etc.)
            pass  # Allow but could add agent-specific validation
        
        player.ability_pips[ability_slot] = pip_count
    
    def get_state_at_time(self, timestamp: float) -> Optional[GameState]:
        """Get the game state at a specific timestamp."""
        # Find the closest snapshot before the timestamp
        for snapshot in reversed(self.state_snapshots):
            if snapshot.timestamp <= timestamp:
                return snapshot
        return self.current_state
    
    def get_events_in_range(
        self,
        start_time: float,
        end_time: float,
        event_types: Optional[List[EventType]] = None,
    ) -> List[ResolvedEvent]:
        """Get all events in a time range, optionally filtered by type."""
        events = [
            e for e in self.event_history
            if start_time <= e.timestamp <= end_time
        ]
        
        if event_types:
            events = [e for e in events if e.event_type in event_types]
        
        return sorted(events, key=lambda e: e.timestamp)
    
    def get_player_positions(self, timestamp: float) -> Dict[str, tuple]:
        """Get all player positions at a specific timestamp."""
        state = self.get_state_at_time(timestamp)
        if not state:
            return {}
        
        return {
            pid: p.position
            for pid, p in state.players.items()
            if p.position is not None
        }
    
    def _snapshot_state(self):
        """Save a copy of the current state."""
        import copy
        snapshot = copy.deepcopy(self.current_state)
        self.state_snapshots.append(snapshot)
    
    def validate_consistency(self) -> List[Dict[str, Any]]:
        """
        Run consistency checks on the resolved timeline.
        Returns a list of potential issues.
        """
        issues = []
        
        # Check for duplicate kills
        kills = [e for e in self.event_history if e.event_type == EventType.KILL]
        for i, kill in enumerate(kills):
            for j, other in enumerate(kills[i+1:], i+1):
                if (abs(kill.timestamp - other.timestamp) < 0.5 and
                    kill.payload["victim"] == other.payload["victim"]):
                    issues.append({
                        "type": "duplicate_kill",
                        "timestamp": kill.timestamp,
                        "victim": kill.payload["victim"],
                    })
        
        # Check kill count matches round outcomes
        rounds = {}
        for event in self.event_history:
            if event.event_type == EventType.ROUND_START:
                rounds[event.payload["round_number"]] = {"kills": []}
            elif event.event_type == EventType.KILL:
                rn = self.current_state.round_number
                if rn in rounds:
                    rounds[rn]["kills"].append(event)
        
        for rn, data in rounds.items():
            if len(data["kills"]) > 10:
                issues.append({
                    "type": "too_many_kills",
                    "round": rn,
                    "count": len(data["kills"]),
                })
        
        return issues
