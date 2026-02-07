"""
Player State Tracker - Tracks player identities, states, and abilities.
"""

import cv2
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from config import (
    ROI_CONFIG,
    PLAYER_CARD_SUBREGIONS,
    TEAM_COLORS,
)


@dataclass
class PlayerIdentity:
    """Represents a player's identity."""
    player_id: str
    player_name: str
    team: str  # "left" or "right"
    team_tag: str  # "FNC", "NRG", etc.
    agent: Optional[str] = None
    roster_index: int = 0  # 0-4


@dataclass
class PlayerState:
    """Represents a player's state at a given timestamp."""
    timestamp_ms: float
    player_id: str
    alive: bool = True
    health: int = 100
    armor: int = 0
    ability_1_available: int = 0
    ability_2_available: int = 0
    ability_3_available: int = 0
    ultimate_ready: bool = False
    ultimate_points: int = 0
    position: Optional[Tuple[float, float]] = None


class PlayerCardAnalyzer:
    """
    Analyzes individual player cards from the HUD.
    Extracts health, abilities, and alive status.
    """
    
    def __init__(self, team: str, slot: int):
        """
        Args:
            team: "left" or "right"
            slot: 1-5 (player slot on team side)
        """
        self.team = team
        self.slot = slot
        self.player_id = f"{team}_player_{slot}"
        self._last_alive = True
        self._last_health = 100
    
    def analyze(self, card_image: np.ndarray) -> PlayerState:
        """
        Analyze a player card image and extract state.
        
        Args:
            card_image: BGR image of the player card ROI
            
        Returns:
            PlayerState with extracted information
        """
        h, w = card_image.shape[:2]
        
        # Determine if player is alive based on grayscale/color
        alive, health = self._estimate_health(card_image)
        
        # Count available abilities
        abilities = self._count_abilities(card_image)
        
        # Check ultimate status
        ult_ready, ult_points = self._check_ultimate(card_image)
        
        state = PlayerState(
            timestamp_ms=0,  # Will be set by caller
            player_id=self.player_id,
            alive=alive,
            health=health,
            ability_1_available=abilities.get(1, 0),
            ability_2_available=abilities.get(2, 0),
            ability_3_available=abilities.get(3, 0),
            ultimate_ready=ult_ready,
            ultimate_points=ult_points,
        )
        
        self._last_alive = alive
        self._last_health = health
        
        return state
    
    def _get_subregion(self, card_image: np.ndarray, subregion_name: str) -> Optional[np.ndarray]:
        """Extract a subregion from the player card, handling mirroring for right-side cards."""
        sub = PLAYER_CARD_SUBREGIONS.get(subregion_name)
        if sub is None:
            return None
        
        h, w = card_image.shape[:2]
        sx, sy, sw, sh = sub
        
        # Mirror horizontally for right-side cards
        if self.team == "right":
            sx = 1.0 - sx - sw
        
        px = int(sx * w)
        py = int(sy * h)
        pw = int(sw * w)
        ph = int(sh * h)
        
        return card_image[py:py+ph, px:px+pw]
    
    def _estimate_health(self, card_image: np.ndarray) -> Tuple[bool, int]:
        """
        Estimate if player is alive and their health percentage.
        Dead players have grayscale cards; alive players have colored elements.
        """
        hsv = cv2.cvtColor(card_image, cv2.COLOR_BGR2HSV)
        h, w = card_image.shape[:2]
        
        # Check for green health bar colors
        green_mask = cv2.inRange(hsv, np.array([35, 80, 80]), np.array([85, 255, 255]))
        green_pixels = cv2.countNonZero(green_mask)
        
        # Check for grayscale (dead player indicator)
        gray_mask = cv2.inRange(hsv, np.array([0, 0, 40]), np.array([180, 50, 150]))
        gray_pixels = cv2.countNonZero(gray_mask)
        
        total_pixels = h * w
        
        # If mostly grayscale, player is dead
        is_alive = green_pixels > (total_pixels * 0.01) or gray_pixels < (total_pixels * 0.3)
        
        # Estimate health from health bar region
        health_region = self._get_subregion(card_image, "health_shield")
        health_pct = 0
        
        if health_region is not None and health_region.size > 0:
            hsv_health = cv2.cvtColor(health_region, cv2.COLOR_BGR2HSV)
            health_green = cv2.inRange(hsv_health, np.array([35, 80, 80]), np.array([85, 255, 255]))
            health_pct = min(100, int((cv2.countNonZero(health_green) / (health_region.size / 3 + 1)) * 100))
        
        if not is_alive:
            health_pct = 0
        elif health_pct == 0 and is_alive:
            health_pct = 100  # Default if can't detect
        
        return is_alive, health_pct
    
    def _count_abilities(self, card_image: np.ndarray) -> Dict[int, int]:
        """
        Count available ability pips for each ability slot.
        
        Returns:
            Dict mapping slot number (1-3) to pip count
        """
        abilities = {}
        
        for slot in [1, 2, 3]:
            ability_region = self._get_subregion(card_image, f"ability_{slot}")
            if ability_region is None or ability_region.size == 0:
                abilities[slot] = 0
                continue
            
            # Count bright pixels (available pips are bright)
            gray = cv2.cvtColor(ability_region, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            
            # Count contours (each pip is a separate bright region)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter by size (pips are small)
            pip_count = 0
            for c in contours:
                area = cv2.contourArea(c)
                if 10 < area < 500:  # Reasonable pip size
                    pip_count += 1
            
            abilities[slot] = min(pip_count, 4)  # Max 4 pips per ability
        
        return abilities
    
    def _check_ultimate(self, card_image: np.ndarray) -> Tuple[bool, int]:
        """
        Check if ultimate is ready and how many points toward it.
        
        Returns:
            Tuple of (is_ready, points)
        """
        ult_region = self._get_subregion(card_image, "ult_charge")
        if ult_region is None or ult_region.size == 0:
            return False, 0
        
        # Ultimate ready indicator is typically bright/glowing
        hsv = cv2.cvtColor(ult_region, cv2.COLOR_BGR2HSV)
        
        # Check for bright golden/yellow glow (ready indicator)
        yellow_mask = cv2.inRange(hsv, np.array([15, 100, 150]), np.array([35, 255, 255]))
        yellow_pixels = cv2.countNonZero(yellow_mask)
        
        # Check for white/bright (also ready indicator)
        white_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 50, 255]))
        white_pixels = cv2.countNonZero(white_mask)
        
        total = ult_region.shape[0] * ult_region.shape[1]
        
        is_ready = (yellow_pixels + white_pixels) > (total * 0.15)
        
        # Points estimation would need OCR or more sophisticated analysis
        points = 8 if is_ready else 0  # Simplified: either ready (8) or not (0)
        
        return is_ready, points


class PlayerStateTracker:
    """
    Tracks all player states across both teams throughout the match.
    """
    
    def __init__(self):
        # Create analyzers for all 10 players
        self.analyzers: Dict[str, PlayerCardAnalyzer] = {}
        
        for team in ["left", "right"]:
            for slot in range(1, 6):
                player_id = f"{team}_player_{slot}"
                self.analyzers[player_id] = PlayerCardAnalyzer(team, slot)
        
        # Track state history
        self.state_history: List[Dict[str, PlayerState]] = []
        
        # Track known player identities
        self.identities: Dict[str, PlayerIdentity] = {}
    
    def update_identity(self, player_id: str, identity: PlayerIdentity):
        """Update a player's identity (name, agent, etc.)."""
        self.identities[player_id] = identity
    
    def process_frame(
        self,
        timestamp_ms: float,
        player_cards: Dict[str, np.ndarray]
    ) -> Dict[str, PlayerState]:
        """
        Process all player cards for a single frame.
        
        Args:
            timestamp_ms: Current timestamp
            player_cards: Dict mapping player_id to card image
            
        Returns:
            Dict mapping player_id to PlayerState
        """
        states = {}
        
        for player_id, card_image in player_cards.items():
            if player_id not in self.analyzers:
                continue
            
            analyzer = self.analyzers[player_id]
            state = analyzer.analyze(card_image)
            state.timestamp_ms = timestamp_ms
            states[player_id] = state
        
        # Store in history
        self.state_history.append(states)
        
        return states
    
    def get_alive_players(self, timestamp_ms: float) -> List[str]:
        """Get list of alive players at a timestamp."""
        # Find closest state snapshot
        if not self.state_history:
            return []
        
        # For simplicity, use the last known state
        last_states = self.state_history[-1]
        return [pid for pid, state in last_states.items() if state.alive]
    
    def apply_kill_event(
        self,
        timestamp_ms: float,
        victim_name: str,
        victim_team: str
    ):
        """
        Update state based on a detected kill event.
        
        This provides a secondary source of truth for alive/dead state.
        """
        # Find the player by name/team
        team_side = "left" if victim_team == "teal" else "right"
        
        # Would need name matching logic here
        # For now, just log
        pass
    
    def export_timeline(self) -> List[Dict[str, Any]]:
        """Export all state changes as timeline events."""
        events = []
        
        prev_states: Dict[str, PlayerState] = {}
        
        for snapshot in self.state_history:
            for player_id, state in snapshot.items():
                prev = prev_states.get(player_id)
                
                # Detect state changes
                if prev is None:
                    # First observation
                    events.append({
                        "timestamp_ms": state.timestamp_ms,
                        "type": "PLAYER_STATE_INIT",
                        "player_id": player_id,
                        "state": {
                            "alive": state.alive,
                            "health": state.health,
                        }
                    })
                else:
                    # Check for changes
                    if prev.alive and not state.alive:
                        events.append({
                            "timestamp_ms": state.timestamp_ms,
                            "type": "PLAYER_DIED",
                            "player_id": player_id,
                        })
                    elif not prev.alive and state.alive:
                        events.append({
                            "timestamp_ms": state.timestamp_ms,
                            "type": "PLAYER_REVIVED",
                            "player_id": player_id,
                        })
                    
                    if prev.ultimate_ready != state.ultimate_ready and state.ultimate_ready:
                        events.append({
                            "timestamp_ms": state.timestamp_ms,
                            "type": "ULTIMATE_READY",
                            "player_id": player_id,
                        })
                
                prev_states[player_id] = state
        
        return events
