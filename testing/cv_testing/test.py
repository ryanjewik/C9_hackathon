import cv2
import numpy as np
import json
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

# -----------------------------
# Config: ROIs (normalized coords)
# -----------------------------
# Use normalized coordinates (x, y, w, h) in [0..1] so it scales with resolution.
# VALORANT broadcast HUD layout based on Champions Paris Grand Final
# Calibrated for 1920x1080 resolution

# Individual player card dimensions (approximate)
# Cards are ~235px wide, ~65px tall at 1920x1080
PLAYER_CARD_WIDTH = 0.16   # ~235px
PLAYER_CARD_HEIGHT = 0.09  # ~65px

LEFT_TEAM_X = 0.010         # ~27px from left edge
RIGHT_TEAM_X = 0.828        # ~1651px from left edge (cards end around x=1890)

# Y positions for each player slot (top to bottom)
# Chronicle starts at y~365, spacing ~75px between card tops
LEFT_PLAYER_Y = [0.505, 0.605, 0.705, 0.805, 0.905]   # FNC: Chronicle, Kaajak, Boaster, Alfajer, Crashies
RIGHT_PLAYER_Y = [0.505, 0.605, 0.705, 0.805, 0.905]  # NRG: skuba, brawk, Ethan, s0m, mada

ROI_NORM: Dict[str, Tuple[float, float, float, float]] = {
    # Minimap - top left corner showing map and player positions
    # Map area ends around y=340px
    "minimap": (0.016, 0.032, 0.250, 0.385),
    
    # Top HUD - center top showing round score, timer, spike status
    # Captures: team logos, scores (2-2), round timer (0:16), round number
    "top_hud": (0.335, 0.005, 0.330, 0.200),
    # Split top HUD into focused subregions for easier OCR and detection
    "top_left_score":  (0.335, 0.005, 0.110, 0.060),
    "top_center_timer":(0.445, 0.005, 0.110, 0.060),
    "top_right_score": (0.555, 0.005, 0.110, 0.060),
    "top_spike_icon":  (0.485, 0.065, 0.035, 0.058),
    "top_plant_text":  (0.43, 0.127, 0.14, 0.070),
    
    # Kill feed - top right corner showing recent kills
    # Below the sponsor logo, captures kill entries
    "killfeed": (0.690, 0.056, 0.300, 0.280),
    
    # Bottom HUD - spectated player info (health, armor, ammo, abilities)
    # At very bottom: health/armor (50/100), ability bar, ammo counter (19)
    "bottom_hud": (0.215, 0.870, 0.570, 0.125),
    
    # Left team player cards (Team 1 - e.g., FNC: Chronicle, Kaajak, Boaster, Alfajer, Crashies)
    "left_player_1": (LEFT_TEAM_X, LEFT_PLAYER_Y[0], PLAYER_CARD_WIDTH, PLAYER_CARD_HEIGHT),
    "left_player_2": (LEFT_TEAM_X, LEFT_PLAYER_Y[1], PLAYER_CARD_WIDTH, PLAYER_CARD_HEIGHT),
    "left_player_3": (LEFT_TEAM_X, LEFT_PLAYER_Y[2], PLAYER_CARD_WIDTH, PLAYER_CARD_HEIGHT),
    "left_player_4": (LEFT_TEAM_X, LEFT_PLAYER_Y[3], PLAYER_CARD_WIDTH, PLAYER_CARD_HEIGHT),
    "left_player_5": (LEFT_TEAM_X, LEFT_PLAYER_Y[4], PLAYER_CARD_WIDTH, PLAYER_CARD_HEIGHT),
    
    # Right team player cards (Team 2 - e.g., NRG: skuba, brawk, Ethan, s0m, mada)
    "right_player_1": (RIGHT_TEAM_X, RIGHT_PLAYER_Y[0], PLAYER_CARD_WIDTH, PLAYER_CARD_HEIGHT),
    "right_player_2": (RIGHT_TEAM_X, RIGHT_PLAYER_Y[1], PLAYER_CARD_WIDTH, PLAYER_CARD_HEIGHT),
    "right_player_3": (RIGHT_TEAM_X, RIGHT_PLAYER_Y[2], PLAYER_CARD_WIDTH, PLAYER_CARD_HEIGHT),
    "right_player_4": (RIGHT_TEAM_X, RIGHT_PLAYER_Y[3], PLAYER_CARD_WIDTH, PLAYER_CARD_HEIGHT),
    "right_player_5": (RIGHT_TEAM_X, RIGHT_PLAYER_Y[4], PLAYER_CARD_WIDTH, PLAYER_CARD_HEIGHT),
}

# Sub-regions within player cards (relative to player card ROI)
# These help extract specific data from each player's HUD element
PLAYER_CARD_SUBREGIONS = {
    # Tuned subregions (x,y,w,h) relative to player card box
    # Layout (left cards):
    # Row1: agent_icon | player_name | health_shield
    # Row2: ult_charge | abilities   | weapon | money
    "agent_icon":   (0.00, 0.04, 0.16, 0.46),    # Agent portrait (top-left)
    "player_name":  (0.17, 0.02, 0.44, 0.36),    # Player name area (top-middle)
    "health_shield":(0.62, 0.02, 0.36, 0.36),    # Health / shield numbers (top-right)
    "ult_charge":   (0.00, 0.52, 0.14, 0.44),    # Ultimate charge (below icon)
    "abilities":    (0.16, 0.50, 0.40, 0.44),    # Abilities row (lower-left/mid)
    "weapon":       (0.58, 0.48, 0.28, 0.44),    # Weapon icon and ammo (lower-right)
    # Individual ability boxes (left-to-right). These will be mirrored for right-side cards.
    # Reduced to three narrow ability boxes (left-to-right). Mirrored for right-side cards.
    "ability_1":    (0.18, 0.52, 0.10, 0.40),    # ability slot 1 (left-most)
    "ability_2":    (0.32, 0.52, 0.10, 0.40),    # ability slot 2 (center)
    "ability_3":    (0.46, 0.52, 0.10, 0.40),    # ability slot 3 (right-most)
    # Move money left to overlap weapon slightly and fully cover value
    "money":        (0.70, 0.52, 0.28, 0.44),    # Credits / economy (overlaps weapon area)
}

# Bottom HUD sub-regions (for spectated player)
BOTTOM_HUD_SUBREGIONS = {
    # Tuned bottom HUD subregions (relative to bottom_hud ROI)
    "health":     (0.02, 0.30, 0.14, 0.38),      # Health number (left side)
    "armor":      (0.16, 0.30, 0.12, 0.38),      # Armor number right next to health
    "abilities":  (0.28, 0.28, 0.44, 0.48),      # Ability icons row (center)
    "ammo":       (0.74, 0.28, 0.24, 0.46),      # Ammo counter (right side)
    "ult_points": (0.44, 0.60, 0.12, 0.30),      # Ultimate meter near center-bottom
}

# Per-detector "effective FPS" (not video FPS)
DET_FPS: Dict[str, float] = {
    "killfeed":     20.0,   # Fast - kills happen quickly
    "top_hud":      10.0,   # Slower - score changes rarely
    "bottom_hud":   15.0,   # Medium - tracks spectated player
    "minimap":      12.0,   # Medium - player positions change
    "player_card":  12.0,   # Medium - health/abilities update during fights
}

# Debug output controls
WRITE_DEBUG_VIDEO = True
DEBUG_VIDEO_PATH = "debug_cv.mp4"

# Event output
EVENTS_PATH = "events.json"


# -----------------------------
# Utility helpers
# -----------------------------
def roi_to_px(frame_w: int, frame_h: int, roi_norm: Tuple[float, float, float, float]) -> Tuple[int, int, int, int]:
    x, y, w, h = roi_norm
    px = int(x * frame_w)
    py = int(y * frame_h)
    pw = int(w * frame_w)
    ph = int(h * frame_h)
    return px, py, pw, ph


def crop(frame, roi_px: Tuple[int, int, int, int]):
    x, y, w, h = roi_px
    return frame[y:y+h, x:x+w]


def mse_gray(a, b) -> float:
    # Mean squared error between two grayscale images
    if a is None or b is None:
        return 1e9
    if a.shape != b.shape:
        return 1e9
    diff = (a.astype("float32") - b.astype("float32"))
    return float((diff * diff).mean())


def _find_contours_gray(img_gray, blur=5, thresh=30):
    g = cv2.GaussianBlur(img_gray, (blur, blur), 0)
    _, th = cv2.threshold(g, thresh, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def count_roundish_contours(roi_bgr) -> int:
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    contours = _find_contours_gray(gray, blur=5, thresh=50)
    count = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < 30 or area > 2000:
            continue
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / float(h + 1e-6)
        if 0.6 <= aspect <= 1.6:
            count += 1
    return count


def count_color_blobs(roi_bgr, min_area=50) -> int:
    # heuristically count colored blobs on minimap (player icons)
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    # broad mask for bright colored player icons (red/blue/green/yellow)
    masks = []
    # red range (two ranges)
    masks.append(cv2.inRange(hsv, (0, 100, 100), (10, 255, 255)))
    masks.append(cv2.inRange(hsv, (160, 100, 100), (179, 255, 255)))
    # blue
    masks.append(cv2.inRange(hsv, (90, 80, 80), (140, 255, 255)))
    # green/yellow
    masks.append(cv2.inRange(hsv, (20, 80, 80), (60, 255, 255)))

    m = np.zeros_like(masks[0])
    for mm in masks:
        m = cv2.bitwise_or(m, mm)

    # cleanup
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = 0
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            cnt += 1
    return cnt


@dataclass
class Event:
    t_ms: float
    type: str
    roi: str
    payload: Dict[str, Any]


class RateGate:
    """
    Run a detector at a target effective FPS, using timestamps from the video.
    """
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


# -----------------------------
# Detector base + skeleton detectors
# -----------------------------
class Detector:
    name: str

    def __init__(self, name: str, target_fps: float, change_mse_threshold: Optional[float] = None):
        self.name = name
        self.gate = RateGate(target_fps)
        self.change_mse_threshold = change_mse_threshold
        self._prev_gray = None  # for change gating
        self._last_fire_ms = -1e9

    def should_run(self, t_ms: float, roi_bgr) -> bool:
        if not self.gate.due(t_ms):
            return False

        if self.change_mse_threshold is None:
            return True

        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        score = mse_gray(gray, self._prev_gray)
        self._prev_gray = gray
        return score >= self.change_mse_threshold

    def process(self, t_ms: float, roi_bgr) -> List[Event]:
        raise NotImplementedError


class KillfeedDetector(Detector):
    """Detects kill events in the top-right killfeed area."""
    def __init__(self, target_fps: float):
        super().__init__("killfeed", target_fps, change_mse_threshold=60.0)
        self.last_row_count = 0

    def process(self, t_ms: float, roi_bgr) -> List[Event]:
        # Detect killfeed entries by looking for horizontal text/icon rows
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        
        # Edge detection to find text/icon boundaries
        edges = cv2.Canny(gray, 50, 150)
        
        # Find horizontal structures (killfeed entries are horizontal bars)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (roi_bgr.shape[1] // 4, 3))
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Count valid killfeed rows (wide rectangles)
        row_count = 0
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            aspect = w / (h + 1e-6)
            if aspect > 3.0 and w > roi_bgr.shape[1] * 0.4:
                row_count += 1

        evs: List[Event] = []
        
        # Detect new kill entries
        if row_count > self.last_row_count:
            new_kills = row_count - self.last_row_count
            evs.append(Event(
                t_ms=t_ms,
                type="KILL_EVENT",
                roi=self.name,
                payload={"new_kills": new_kills, "total_visible": row_count}
            ))
        
        self.last_row_count = row_count
        return evs


class PlayerCardDetector(Detector):
    """Detects changes in a player's HUD card (health, abilities, alive status)."""
    def __init__(self, name: str, target_fps: float, team: str, slot: int):
        super().__init__(name, target_fps, change_mse_threshold=35.0)
        self.team = team  # "left" or "right"
        self.slot = slot  # 1-5
        self.last_alive = True
        self.last_health_estimate = 100
        self.last_ability_count = 0

    def _estimate_health(self, roi_bgr) -> Tuple[int, bool]:
        """Estimate health percentage and alive status from player card."""
        h, w = roi_bgr.shape[:2]

        # Helper to extract a named subregion and mirror for right players
        def subroi(name_key: str):
            sub = PLAYER_CARD_SUBREGIONS.get(name_key)
            if sub is None:
                return None
            sx = int(sub[0] * w)
            sw = int(sub[2] * w)
            # mirror horizontally for right-side cards
            if self.team == "right":
                sx = int((1.0 - sub[0] - sub[2]) * w)
            sy = int(sub[1] * h)
            sh = int(sub[3] * h)
            return roi_bgr[sy:sy+sh, sx:sx+sw]

        # Use the health_shield subregion for estimating health
        health_region = subroi("health_shield")
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

        # Fallback full-roi masks for alive detection
        green_mask_full = cv2.inRange(hsv, (35, 80, 80), (85, 255, 255))
        green_pixels = cv2.countNonZero(green_mask_full)
        gray_mask = cv2.inRange(hsv, (0, 0, 40), (180, 50, 150))
        gray_pixels = cv2.countNonZero(gray_mask)

        total_pixels = h * w
        is_alive = green_pixels > (total_pixels * 0.01) or gray_pixels < (total_pixels * 0.3)

        if health_region is not None and health_region.size > 0:
            hsv_health = cv2.cvtColor(health_region, cv2.COLOR_BGR2HSV)
            health_green = cv2.inRange(hsv_health, (35, 80, 80), (85, 255, 255))
            health_pct = min(100, int((cv2.countNonZero(health_green) / (health_region.size/3 + 1)) * 100))
        else:
            health_pct = 0 if not is_alive else 100

        return health_pct, is_alive

    def _count_abilities(self, roi_bgr) -> int:
        """Count available ability indicators."""
        h, w = roi_bgr.shape[:2]
        count = 0

        # Prefer individual ability boxes if defined
        for i in range(1, 4):
            key = f"ability_{i}"
            sub = PLAYER_CARD_SUBREGIONS.get(key)
            if sub is None:
                continue

            sx = int(sub[0] * w)
            sw = int(sub[2] * w)
            # mirror for right-side cards
            if self.team == "right":
                sx = int((1.0 - sub[0] - sub[2]) * w)
            sy = int(sub[1] * h)
            sh = int(sub[3] * h)

            subroi = roi_bgr[sy:sy+sh, sx:sx+sw]
            if subroi.size == 0:
                continue

            gray = cv2.cvtColor(subroi, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
            prop = cv2.countNonZero(thresh) / (subroi.size/3 + 1)
            # if a reasonable fraction of the box is non-dark, treat it as present
            if prop > 0.02:
                count += 1

        # Fallback: if no individual boxes, fall back to previous coarse method
        if count == 0 and "abilities" in PLAYER_CARD_SUBREGIONS:
            sub = PLAYER_CARD_SUBREGIONS.get("abilities")
            sx = int(sub[0] * w)
            sw = int(sub[2] * w)
            if self.team == "right":
                sx = int((1.0 - sub[0] - sub[2]) * w)
            sy = int(sub[1] * h)
            sh = int(sub[3] * h)
            ability_region = roi_bgr[sy:sy+sh, sx:sx+sw]
            if ability_region.size > 0:
                gray = cv2.cvtColor(ability_region, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in contours:
                    area = cv2.contourArea(c)
                    if 50 < area < 2000:
                        x, y, ww, hh = cv2.boundingRect(c)
                        aspect = ww / (hh + 1e-6)
                        if 0.5 <= aspect <= 2.0:
                            count += 1

        return min(count, 4)

    def process(self, t_ms: float, roi_bgr) -> List[Event]:
        evs: List[Event] = []
        
        health, is_alive = self._estimate_health(roi_bgr)
        ability_count = self._count_abilities(roi_bgr)
        
        # Detect death event
        if self.last_alive and not is_alive:
            evs.append(Event(
                t_ms=t_ms,
                type="PLAYER_DEATH",
                roi=self.name,
                payload={"team": self.team, "slot": self.slot}
            ))
        
        # Detect respawn/round reset
        if not self.last_alive and is_alive:
            evs.append(Event(
                t_ms=t_ms,
                type="PLAYER_RESPAWN",
                roi=self.name,
                payload={"team": self.team, "slot": self.slot}
            ))
        
        # Detect significant health change
        health_delta = abs(health - self.last_health_estimate)
        if health_delta > 20 and is_alive:
            evs.append(Event(
                t_ms=t_ms,
                type="HEALTH_CHANGE",
                roi=self.name,
                payload={
                    "team": self.team, 
                    "slot": self.slot,
                    "prev_health": self.last_health_estimate,
                    "curr_health": health,
                    "delta": health - self.last_health_estimate
                }
            ))
        
        # Detect ability usage
        if ability_count < self.last_ability_count:
            evs.append(Event(
                t_ms=t_ms,
                type="ABILITY_USED",
                roi=self.name,
                payload={
                    "team": self.team,
                    "slot": self.slot,
                    "abilities_remaining": ability_count
                }
            ))
        
        self.last_alive = is_alive
        self.last_health_estimate = health
        self.last_ability_count = ability_count
        
        return evs


class TopHudDetector(Detector):
    """Detects round score changes and timer events in top center HUD."""
    def __init__(self, target_fps: float):
        super().__init__("top_hud", target_fps, change_mse_threshold=50.0)
        self.last_brightness = 0
        self.spike_planted = False

    def process(self, t_ms: float, roi_bgr) -> List[Event]:
        evs: List[Event] = []
        
        # Split into left score, center (timer/spike), right score
        h, w = roi_bgr.shape[:2]
        center_region = roi_bgr[:, int(w*0.35):int(w*0.65)]
        
        # Detect spike plant indicator (orange/red glow in center)
        hsv = cv2.cvtColor(center_region, cv2.COLOR_BGR2HSV)
        orange_mask = cv2.inRange(hsv, (5, 100, 100), (25, 255, 255))
        orange_pct = cv2.countNonZero(orange_mask) / (center_region.size / 3 + 1)
        
        spike_now = orange_pct > 0.05
        
        if spike_now and not self.spike_planted:
            evs.append(Event(
                t_ms=t_ms,
                type="SPIKE_PLANTED",
                roi=self.name,
                payload={}
            ))
        elif not spike_now and self.spike_planted:
            evs.append(Event(
                t_ms=t_ms,
                type="SPIKE_RESOLVED",
                roi=self.name,
                payload={}
            ))
        
        self.spike_planted = spike_now
        
        # Detect brightness flash (round end/start)
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        
        if abs(brightness - self.last_brightness) > 30:
            evs.append(Event(
                t_ms=t_ms,
                type="ROUND_TRANSITION",
                roi=self.name,
                payload={"brightness_delta": brightness - self.last_brightness}
            ))
        
        self.last_brightness = brightness
        return evs


class BottomHudDetector(Detector):
    """Detects spectated player's health, abilities, and weapon changes."""
    def __init__(self, target_fps: float):
        super().__init__("bottom_hud", target_fps, change_mse_threshold=40.0)
        self.last_ult_ready = False
        self._prev_gray = None

    def process(self, t_ms: float, roi_bgr) -> List[Event]:
        evs: List[Event] = []
        h, w = roi_bgr.shape[:2]
        
        # Detect ultimate ready (bright X icon or filled ult meter)
        ult_region = roi_bgr[int(h*0.6):, int(w*0.42):int(w*0.58)]
        if ult_region.size > 0:
            hsv = cv2.cvtColor(ult_region, cv2.COLOR_BGR2HSV)
            # Ultimate ready usually has bright yellow/white glow
            bright_mask = cv2.inRange(hsv, (0, 0, 200), (180, 60, 255))
            yellow_mask = cv2.inRange(hsv, (20, 100, 150), (40, 255, 255))
            combined = cv2.bitwise_or(bright_mask, yellow_mask)
            ult_pct = cv2.countNonZero(combined) / (ult_region.size / 3 + 1)
            
            ult_ready = ult_pct > 0.15
            
            if ult_ready and not self.last_ult_ready:
                evs.append(Event(
                    t_ms=t_ms,
                    type="ULT_READY",
                    roi=self.name,
                    payload={}
                ))
            
            self.last_ult_ready = ult_ready
        
        # Detect significant HUD changes (weapon swap, damage taken)
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        if self._prev_gray is not None and self._prev_gray.shape == gray.shape:
            mse = mse_gray(gray, self._prev_gray)
            if mse > 300:
                evs.append(Event(
                    t_ms=t_ms,
                    type="SPECTATED_PLAYER_CHANGE",
                    roi=self.name,
                    payload={"change_magnitude": mse}
                ))
        
        self._prev_gray = gray.copy()
        return evs


class MinimapDetector(Detector):
    """Detects player position changes and rotations on the minimap."""
    def __init__(self, target_fps: float):
        super().__init__("minimap", target_fps, change_mse_threshold=25.0)
        self.last_player_count = 0
        self.last_positions: List[Tuple[int, int]] = []

    def _find_player_blips(self, roi_bgr) -> List[Tuple[int, int, str]]:
        """Find player icons on minimap, returns (x, y, color_team)."""
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        blips = []
        
        # Green team indicators (defending team / allies)
        green_mask = cv2.inRange(hsv, (35, 80, 100), (85, 255, 255))
        # Blue/cyan team indicators
        blue_mask = cv2.inRange(hsv, (85, 80, 100), (130, 255, 255))
        # Red/orange enemy indicators
        red_mask1 = cv2.inRange(hsv, (0, 100, 100), (15, 255, 255))
        red_mask2 = cv2.inRange(hsv, (165, 100, 100), (180, 255, 255))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        # Yellow indicators
        yellow_mask = cv2.inRange(hsv, (15, 100, 100), (35, 255, 255))
        
        for mask, team in [(green_mask, "green"), (blue_mask, "blue"), 
                           (red_mask, "red"), (yellow_mask, "yellow")]:
            # Clean up mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for c in contours:
                area = cv2.contourArea(c)
                if 30 < area < 3000:  # Filter noise
                    M = cv2.moments(c)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        blips.append((cx, cy, team))
        
        return blips

    def process(self, t_ms: float, roi_bgr) -> List[Event]:
        evs: List[Event] = []
        
        blips = self._find_player_blips(roi_bgr)
        player_count = len(blips)
        
        # Detect player count change (death visible on minimap)
        if player_count != self.last_player_count:
            evs.append(Event(
                t_ms=t_ms,
                type="MINIMAP_PLAYER_CHANGE",
                roi=self.name,
                payload={
                    "prev_count": self.last_player_count,
                    "curr_count": player_count,
                    "blip_colors": [b[2] for b in blips]
                }
            ))
        
        # Detect significant position changes (rotation)
        if len(blips) >= 3 and len(self.last_positions) >= 3:
            total_movement = 0
            for (x, y, _) in blips[:5]:
                min_dist = min(
                    ((x - px)**2 + (y - py)**2)**0.5 
                    for px, py in self.last_positions[:5]
                ) if self.last_positions else 0
                total_movement += min_dist
            
            avg_movement = total_movement / len(blips)
            if avg_movement > 20:
                evs.append(Event(
                    t_ms=t_ms,
                    type="MINIMAP_ROTATION",
                    roi=self.name,
                    payload={"avg_movement": avg_movement}
                ))
        
        self.last_player_count = player_count
        self.last_positions = [(x, y) for x, y, _ in blips]
        
        return evs


# -----------------------------
# Main processing loop
# -----------------------------
def build_detectors() -> List[Detector]:
    """Build all detectors for the VALORANT HUD."""
    detectors: List[Detector] = []
    
    # Core HUD detectors
    detectors.append(KillfeedDetector(DET_FPS["killfeed"]))
    detectors.append(TopHudDetector(DET_FPS["top_hud"]))
    detectors.append(BottomHudDetector(DET_FPS["bottom_hud"]))
    detectors.append(MinimapDetector(DET_FPS["minimap"]))
    
    # Left team player cards (5 players)
    for i in range(1, 6):
        detectors.append(PlayerCardDetector(
            name=f"left_player_{i}",
            target_fps=DET_FPS["player_card"],
            team="left",
            slot=i
        ))
    
    # Right team player cards (5 players)
    for i in range(1, 6):
        detectors.append(PlayerCardDetector(
            name=f"right_player_{i}",
            target_fps=DET_FPS["player_card"],
            team="right",
            slot=i
        ))
    
    return detectors


def draw_debug_overlay(frame, roi_norm_dict: Dict[str, Tuple[float, float, float, float]], 
                       w: int, h: int) -> np.ndarray:
    """Draw ROI rectangles and labels on frame for debugging."""
    dbg = frame.copy()
    
    # Color coding by ROI type
    colors = {
        "minimap": (255, 255, 0),      # Cyan
        "top_hud": (0, 255, 255),       # Yellow
        "killfeed": (0, 0, 255),        # Red
        "bottom_hud": (255, 0, 255),    # Magenta
        "left_player": (0, 255, 0),     # Green
        "right_player": (255, 128, 0),  # Orange
    }
    # Specific top-subregion colors
    top_colors = {
        "top_left_score": (200, 220, 0),
        "top_center_timer": (0, 220, 220),
        "top_right_score": (0, 180, 255),
        "top_spike_icon": (0, 0, 255),
        "top_plant_text": (180, 0, 180),
    }
    
    for name, roi_norm in roi_norm_dict.items():
        x, y, rw, rh = roi_to_px(w, h, roi_norm)
        
        # Determine color
        color = (0, 255, 0)  # default green
        for key, c in colors.items():
            if name.startswith(key):
                color = c
                break
        # Override for explicit top-HUD subregions
        if name in top_colors:
            color = top_colors[name]
        
        cv2.rectangle(dbg, (x, y), (x + rw, y + rh), color, 2)
        
        # Smaller font for player cards
        font_scale = 0.4 if "player" in name else 0.6
        cv2.putText(dbg, name, (x + 3, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)
        # Draw subregions for player cards so you can fine-tune placement
        if name.startswith("left_player") or name.startswith("right_player"):
            # small colors for subregions
            sub_colors = {
                "agent_icon": (200, 200, 0),
                "health_bar": (0, 200, 0),
                "abilities":  (0, 150, 255),
                "economy":    (180, 0, 180),
                "weapon":     (50, 200, 50),
            }
            for subname, subnorm in PLAYER_CARD_SUBREGIONS.items():
                # Mirror horizontally for right-side player cards
                if name.startswith("right_player"):
                    rel_x = (1.0 - subnorm[0] - subnorm[2])
                else:
                    rel_x = subnorm[0]
                sx = x + int(rel_x * rw)
                sy = y + int(subnorm[1] * rh)
                sw = int(subnorm[2] * rw)
                sh = int(subnorm[3] * rh)
                sc = sub_colors.get(subname, (255, 255, 255))
                cv2.rectangle(dbg, (sx, sy), (sx + sw, sy + sh), sc, 1)
                # label small; avoid overflow
                lbl_x = sx + 2
                lbl_y = sy + 12 if sh >= 14 else sy + sh
                cv2.putText(dbg, subname, (lbl_x, lbl_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, sc, 1)

        # Draw bottom-hud subregions when present
        if name == "bottom_hud":
            bh_colors = {
                "health": (0, 200, 0),
                "armor": (180, 180, 0),
                "abilities": (0, 150, 255),
                "ammo": (50, 200, 50),
                "ult_points": (0, 200, 200),
            }
            for subname, subnorm in BOTTOM_HUD_SUBREGIONS.items():
                sx = x + int(subnorm[0] * rw)
                sy = y + int(subnorm[1] * rh)
                sw = int(subnorm[2] * rw)
                sh = int(subnorm[3] * rh)
                sc = bh_colors.get(subname, (255, 255, 255))
                cv2.rectangle(dbg, (sx, sy), (sx + sw, sy + sh), sc, 1)
                lbl_x = sx + 2
                lbl_y = sy + 12 if sh >= 14 else sy + sh
                cv2.putText(dbg, subname, (lbl_x, lbl_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, sc, 1)

    return dbg


def main():
    video_path = "test.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {video_path} ({w}x{h}) fps={fps:.2f}")

    # Build detectors
    detectors = build_detectors()
    print(f"Initialized {len(detectors)} detectors:")
    for det in detectors:
        print(f"  - {det.name}")

    # Debug video writer
    writer = None
    if WRITE_DEBUG_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(DEBUG_VIDEO_PATH, fourcc, fps, (w, h))
        if not writer.isOpened():
            print("WARN: Could not open debug video writer; disabling.")
            writer = None

    events: List[Event] = []
    event_counts: Dict[str, int] = {}

    frame_idx = 0
    t_start = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

        # For each detector: crop ROI and optionally run
        for det in detectors:
            if det.name not in ROI_NORM:
                continue
                
            roi_norm = ROI_NORM[det.name]
            roi_px = roi_to_px(w, h, roi_norm)
            roi_img = crop(frame, roi_px)

            if roi_img.size == 0:
                continue

            if det.should_run(t_ms, roi_img):
                evs = det.process(t_ms, roi_img)
                for ev in evs:
                    events.append(ev)
                    event_counts[ev.type] = event_counts.get(ev.type, 0) + 1

        # Debug overlay: draw ROIs
        if writer is not None:
            dbg = draw_debug_overlay(frame, ROI_NORM, w, h)
            writer.write(dbg)

        frame_idx += 1
        if frame_idx % int(max(1, fps * 2)) == 0:
            elapsed = time.time() - t_start
            print(f"Processed {frame_idx} frames in {elapsed:.1f}s "
                  f"(x{(frame_idx/fps)/elapsed:.2f} realtime) | "
                  f"Events: {len(events)}")

    cap.release()
    if writer is not None:
        writer.release()

    # Save events
    out = [
        {"t_ms": e.t_ms, "type": e.type, "roi": e.roi, "payload": e.payload}
        for e in events
    ]
    with open(EVENTS_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # Print summary
    print(f"\n{'='*50}")
    print(f"Processing complete!")
    print(f"{'='*50}")
    print(f"Total events: {len(events)}")
    print(f"\nEvents by type:")
    for event_type, count in sorted(event_counts.items(), key=lambda x: -x[1]):
        print(f"  {event_type}: {count}")
    print(f"\nWrote events to {EVENTS_PATH}")
    if WRITE_DEBUG_VIDEO:
        print(f"Wrote debug video to {DEBUG_VIDEO_PATH}")


if __name__ == "__main__":
    main()
