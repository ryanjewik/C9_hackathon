import cv2
import numpy as np
import json
import time
import re
import shutil
import subprocess
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set

# PostgreSQL support for dynamic player name lookup
try:
    import psycopg2
    HAS_POSTGRES = True
except ImportError:
    HAS_POSTGRES = False
    print("Warning: psycopg2 not installed. Database player lookup disabled.")

# Try to import OCR libraries (optional, will fall back to color-based detection)
try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False
    print("Warning: pytesseract not installed. Text detection disabled.")

# Do NOT import easyocr at module import time. Importing easyocr triggers
# a torch import which can fail on some Windows systems (DLL init error).
# We'll attempt to import and initialize EasyOCR lazily inside
# KillfeedTextReader._init_reader().
HAS_EASYOCR = False


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
    "killfeed": (0.690, 0.08, 0.300, 0.280),
    
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
    "killfeed":     8.0,    # Slower - kills visible for ~3-5 seconds, reduce duplicates
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
KILL_SUMMARY_PATH = "kill_summary.json"


# -----------------------------
# Database Player Name Lookup
# -----------------------------
class PlayerNameDatabase:
    """
    Dynamic player name lookup using PostgreSQL database.
    Caches player names for performance and provides fuzzy matching.
    """
    _instance = None
    _player_names: Set[str] = set()
    _last_refresh: float = 0
    _refresh_interval: float = 300  # Refresh every 5 minutes
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._match_filter = None  # Initialize match filter
        self._connection_params = {
            'host': os.environ.get('POSTGRES_HOST', 'localhost'),
            'port': os.environ.get('POSTGRES_PORT', '5432'),
            'database': os.environ.get('POSTGRES_DB', 'esports'),
            'user': os.environ.get('POSTGRES_USER', 'postgres'),
            'password': os.environ.get('POSTGRES_PASSWORD', ''),
        }
        self._refresh_player_names()
        
        # Check for MATCH_PLAYERS environment variable
        # Format: comma-separated list of player names
        match_players_env = os.environ.get('MATCH_PLAYERS', '')
        if match_players_env:
            player_list = [p.strip() for p in match_players_env.split(',') if p.strip()]
            if player_list:
                self.set_match_player_filter(player_list)
    
    def _refresh_player_names(self) -> None:
        """Fetch player names from database."""
        if not HAS_POSTGRES:
            print("PostgreSQL not available - using fallback player list")
            # Minimal fallback - just use the known players for this match
            self._player_names = set()
            return
            
        try:
            conn = psycopg2.connect(**self._connection_params)
            cursor = conn.cursor()
            cursor.execute("SELECT nickname FROM esports_players;")
            rows = cursor.fetchall()
            self._player_names = {row[0] for row in rows if row[0]}
            cursor.close()
            conn.close()
            self._last_refresh = time.time()
            print(f"Loaded {len(self._player_names)} player names from database")
        except Exception as e:
            print(f"Database connection failed: {e}")
            self._player_names = set()
    
    def get_player_names(self) -> Set[str]:
        """Get cached player names, refreshing if needed."""
        if time.time() - self._last_refresh > self._refresh_interval:
            self._refresh_player_names()
        return self._player_names
    
    def set_match_player_filter(self, player_names: List[str]) -> None:
        """
        Set a filter to only match against specific players (e.g., players in current match).
        This significantly improves accuracy by reducing false positives from the 25k+ player database.
        
        Args:
            player_names: List of player nicknames expected in this match
        """
        self._match_filter = set(name.lower() for name in player_names)
        print(f"Match filter set: {len(self._match_filter)} players")
    
    def clear_match_filter(self) -> None:
        """Clear the match player filter."""
        self._match_filter = None
    
    def get_filtered_players(self) -> Set[str]:
        """Get player names, filtered by match if filter is set."""
        all_names = self.get_player_names()
        if hasattr(self, '_match_filter') and self._match_filter:
            # Return only players that match the filter (case-insensitive)
            return {name for name in all_names if name.lower() in self._match_filter}
        return all_names
    
    def fuzzy_match(self, ocr_name: str, threshold: float = 0.6) -> Optional[str]:
        """
        Fuzzy match OCR output against database player names.
        Returns best match if above threshold, else None.
        
        Note: Higher default threshold (0.6) to reduce false positives with 25k+ players.
        Use set_match_player_filter() to narrow the search space for better accuracy.
        """
        if not ocr_name or len(ocr_name) < 2:
            return None
        
        # Use filtered players if match filter is set
        names = self.get_filtered_players()
        if not names:
            return None
        
        ocr_clean = ocr_name.lower().strip()
        # Remove common team tag prefixes
        ocr_clean = re.sub(r'^(nrg|fnc|100t|sen|c9|eg|loud|drx|prx|fut|lev|kru|g2|th|geng|t1|edg|fpx|blg)[\s_]?', '', ocr_clean, flags=re.IGNORECASE)
        
        best_score = 0.0
        best_match = None
        
        for name in names:
            name_lower = name.lower()
            
            # Exact match (case insensitive)
            if ocr_clean == name_lower:
                return name
            
            # Substring containment - only accept if length is significant
            # Avoid short substrings like "er" matching "Boaster"
            if len(name_lower) >= 4 and len(ocr_clean) >= 4:
                if name_lower in ocr_clean or ocr_clean in name_lower:
                    return name
            
            # Calculate similarity
            score = calculate_similarity(ocr_clean, name_lower)
            
            # Bonus for matching prefix
            if len(ocr_clean) >= 2 and len(name_lower) >= 2:
                if ocr_clean[:2] == name_lower[:2]:
                    score += 0.1
            
            if score > best_score:
                best_score = score
                best_match = name
        
        return best_match if best_score >= threshold else None


# Global database instance
_player_db: Optional[PlayerNameDatabase] = None

def get_player_database() -> PlayerNameDatabase:
    """Get or create the global player database instance."""
    global _player_db
    if _player_db is None:
        _player_db = PlayerNameDatabase()
    return _player_db


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


def _apply_ocr_corrections(text: str) -> str:
    """
    Apply GENERIC OCR corrections for gaming fonts.
    These are character-level fixes that apply universally, NOT player-specific fixes.
    Player-specific matching is done via database fuzzy matching.
    """
    result = text
    
    # GENERIC character-level OCR fixes (not player-specific)
    # These fix common font rendering issues that affect ALL text
    generic_fixes = [
        # I/l/1/| confusion - very common in gaming fonts
        ('|', 'l'),             # Pipe to lowercase L
        ('!', 'i'),             # Exclamation to i (when in middle of word)
        # Common ligature/kerning issues  
        ('rn', 'm'),            # rn often looks like m
        ('vv', 'w'),            # vv often looks like w
        ('cl', 'd'),            # cl can look like d
        ('cI', 'd'),            # cI can look like d
        # Double character fixes (OCR artifacts)
        ('ll', 'l'),            # Double l often wrong (apply carefully)
        # Common symbol substitutions
        ('$', 's'),             # Dollar sign to s
        ('@', 'a'),             # At sign to a
        # Spacing artifacts
        ('  ', ' '),            # Double space to single
    ]
    
    for wrong, right in generic_fixes:
        if wrong in result:
            result = result.replace(wrong, right)
    
    # More aggressive normalization for comparison (doesn't change display)
    # Strip any remaining non-alphanumeric from interior
    result = re.sub(r'(?<=[a-zA-Z])[^a-zA-Z0-9\s]+(?=[a-zA-Z])', '', result)
    
    return result


def clean_player_name(raw_name: str, team_color: Optional[str] = None) -> str:
    """
    Clean up OCR output to extract proper player name.
    Applies gaming font corrections and strips team tags.
    
    Args:
        raw_name: The raw OCR'd name
        team_color: Optional team color ("teal" or "orange") - used for team tag
    """
    if not raw_name or raw_name == "Unknown":
        return "Unknown"
    
    name = raw_name.strip()
    
    # Remove common OCR artifacts
    name = re.sub(r'^[\[\(\{\|\]@\'\"#\-\d~;:]+', '', name)  # Leading brackets/symbols/numbers
    name = re.sub(r'[\]\)\}\|;:,\.\'\"_\-\d~]+$', '', name)  # Trailing punctuation/numbers
    name = re.sub(r'[\"\']', '', name)  # Quotes
    
    # If name is too short after cleanup, return Unknown
    if len(name) < 2:
        return "Unknown"
    
    # Apply gaming font OCR corrections early
    name = _apply_ocr_corrections(name)
    
    # Team tags sometimes bleed into OCR
    # Strip them aggressively - these are OCR'd versions with errors
    team_tag_variants = [
        # NRG variants (with optional space/separator after)
        r'^[NnMmWwIi][RrPp][GgCcOo][\s_~\-]*',  # NRG with various OCR errors
        r'^[Nn][Rr][Gg][\s_~\-]*',  # Clean NRG
        r'^[Nn][Rr][Cc][\s_~\-]*',  # NRC misread
        r'^[Mm][Rr][Gg][\s_~\-]*',  # MRG misread  
        r'^[Nn][Gg][\s_~\-]+',      # NG partial (require space to avoid stripping valid name starts)
        # FNC variants
        r'^[FfEeGg][NnMm][Cc][\s_~\-]*',  # FNC with errors
        r'^[Ff][Nn][Cc][\s_~\-]*',  # Clean FNC
        r'^[Ff][Cc][\s_~\-]*',      # FC partial
        r'^[Ff][Dd][\s_~\-]*',      # Fd partial
        r'^[Ff][;:~][\s_~\-]*',      # F; OCR artifact
        # Generic: 2-3 uppercase followed by space and another uppercase
        r'^[A-Z]{2,3}[\s_~\-]+(?=[A-Z])',  # ABC followed by whitespace and capital
    ]
    
    # Run multiple passes to strip stacked tags like "NRG NRG kubri"
    for _ in range(3):
        stripped = False
        for pattern in team_tag_variants:
            match = re.match(pattern, name, re.IGNORECASE)
            if match:
                remainder = name[match.end():].strip()
                # Accept if remainder looks like a name (2+ chars, starts with letter)
                if len(remainder) >= 2 and remainder[0].isalpha():
                    name = remainder
                    stripped = True
                    break
        if not stripped:
            break
    
    # Clean up remaining artifacts
    name = re.sub(r'^[\W\d_]+', '', name)  # Leading non-letters
    name = re.sub(r'[\W_]+$', '', name)    # Trailing non-alphanumeric
    
    # Remove internal OCR junk like "~" or ";" that got embedded
    name = re.sub(r'[~;:]', '', name)
    
    # Apply corrections again after stripping tags
    name = _apply_ocr_corrections(name)
    
    if not name or len(name) < 2:
        return "Unknown"
    
    # Capitalize first letter, keep rest as-is
    name = name[0].upper() + name[1:] if len(name) > 1 else name.upper()
    
    # Try fuzzy matching against known roster to correct OCR errors
    matched_name = fuzzy_match_player(name, team_color, threshold=0.4)
    if matched_name:
        name = matched_name
    
    # Add team tag based on color detection (the authoritative source)
    if team_color and team_color in ("teal", "orange"):
        team_tag = "FNC" if team_color == "teal" else "NRG"
        return f"{team_tag} {name}"
    
    return name


def fuzzy_match_player(ocr_name: str, team_color: Optional[str] = None, threshold: float = 0.55) -> Optional[str]:
    """
    Fuzzy match OCR output against player names from database.
    Returns the best match if similarity is above threshold, else None.
    
    Args:
        ocr_name: The OCR'd player name
        team_color: Optional team color (currently unused - database doesn't have team info)
        threshold: Minimum similarity score to accept a match (default 0.55 to reduce false positives)
    """
    if not ocr_name or len(ocr_name) < 2:
        return None
    
    # Get player database - use filtered players if match filter is set
    db = get_player_database()
    player_names = db.get_filtered_players()
    
    # If no database players, return None (can't match)
    if not player_names:
        print("Warning: No player names in database for fuzzy matching")
        return None
    
    ocr_lower = ocr_name.lower().strip()
    
    # Remove any remaining team tag artifacts (common tags)
    ocr_lower = re.sub(r'^(nrg|fnc|100t|sen|c9|eg|loud|drx|prx|fut|lev|kru|g2|th|geng|t1|edg|fpx|blg|wrg|inc|enc)[\s_]?', '', ocr_lower, flags=re.IGNORECASE)
    
    # OCR-aware normalization for comparison
    def ocr_normalize(s: str) -> str:
        """Normalize string accounting for common OCR confusions."""
        s = s.lower()
        s = re.sub(r'[0ou]', 'o', s)      # 0/O/u confusion (u looks like 0 in some fonts)
        s = re.sub(r'[1il|!]', 'i', s)    # 1/l/I/|/! confusion
        s = re.sub(r'[5s$]', 's', s)      # 5/S/$ confusion
        s = re.sub(r'[8b]', 'b', s)       # 8/B confusion
        s = re.sub(r'[gs]', 's', s)       # g/s confusion (gaming fonts)
        s = re.sub(r'rn', 'm', s)         # rn looks like m
        s = re.sub(r'vv', 'w', s)         # vv looks like w
        s = re.sub(r'cl', 'd', s)         # cl looks like d
        return s
    
    ocr_normalized = ocr_normalize(ocr_lower)
    
    best_score = 0.0
    best_match = None
    
    for name in player_names:
        name_lower = name.lower()
        name_normalized = ocr_normalize(name_lower)
        
        # Exact match (case insensitive)
        if ocr_lower == name_lower:
            return name
        
        # Exact match after normalization
        if ocr_normalized == name_normalized:
            return name
        
        # Direct substring match - require minimum length to avoid false positives
        if len(name_lower) >= 4 and len(ocr_lower) >= 4:
            if name_lower in ocr_lower or ocr_lower in name_lower:
                return name
            if name_normalized in ocr_normalized or ocr_normalized in name_normalized:
                return name
        
        # Check prefix/suffix matches (OCR often drops chars at edges)
        if len(name_lower) >= 3 and len(ocr_lower) >= 3:
            # Starting chars match
            if ocr_lower.startswith(name_lower[:3]) or name_lower.startswith(ocr_lower[:3]):
                score = calculate_similarity(ocr_lower, name_lower)
                if score > 0.4:
                    return name
            # Ending chars match
            if ocr_lower.endswith(name_lower[-3:]) or name_lower.endswith(ocr_lower[-3:]):
                score = calculate_similarity(ocr_lower, name_lower)
                if score > 0.4:
                    return name
        
        # Calculate similarity score using both raw and normalized
        score = calculate_similarity(ocr_lower, name_lower)
        score_norm = calculate_similarity(ocr_normalized, name_normalized)
        score = max(score, score_norm)
        
        # Bonus for matching first 2 characters
        if len(ocr_lower) >= 2 and len(name_lower) >= 2:
            if ocr_lower[:2] == name_lower[:2] or ocr_normalized[:2] == name_normalized[:2]:
                score += 0.15
        
        if score > best_score:
            best_score = score
            best_match = name
    
    return best_match if best_score >= threshold else None


def calculate_similarity(s1: str, s2: str) -> float:
    """
    Calculate similarity ratio between two strings using multiple methods.
    More tolerant of OCR errors by considering:
    - Longest common subsequence
    - Character frequency overlap with counts
    - Common OCR substitutions (l/i/1, o/0, t/f, m/n, etc.)
    - Substring containment
    """
    if not s1 or not s2:
        return 0.0
    
    # Normalize for comparison
    s1 = s1.lower().strip()
    s2 = s2.lower().strip()
    
    # If identical after normalization
    if s1 == s2:
        return 1.0
    
    len1, len2 = len(s1), len(s2)
    
    # Apply OCR-aware normalization (common substitutions)
    def ocr_normalize(s):
        # Common OCR confusions for gaming fonts
        s = s.replace('0', 'o').replace('1', 'l').replace('|', 'l').replace('!', 'i')
        s = s.replace('$', 's').replace('8', 'b').replace('5', 's')
        s = s.replace('rn', 'm').replace('cl', 'd').replace('vv', 'w')
        # Gaming fonts: t/f, i/j, n/m often confused
        # Keep both options by not normalizing these
        return s
    
    s1_norm = ocr_normalize(s1)
    s2_norm = ocr_normalize(s2)
    
    # Substring containment boost - if one string contains the other, high match
    shorter, longer = (s1_norm, s2_norm) if len1 <= len2 else (s2_norm, s1_norm)
    if len(shorter) >= 4 and shorter in longer:
        return 0.85 + 0.15 * (len(shorter) / len(longer))
    
    # Check if one is a suffix/prefix of the other (OCR often adds/drops chars at start)
    if len(shorter) >= 4:
        if longer.endswith(shorter) or longer.startswith(shorter):
            return 0.8 + 0.2 * (len(shorter) / len(longer))
        # Also check for suffix of at least 5 chars
        if len(shorter) >= 5 and shorter[-5:] == longer[-5:]:
            return 0.75
    
    # Quick check for very different lengths (but more lenient)
    if abs(len1 - len2) > max(len1, len2) * 0.8:
        return 0.0
    
    # Method 1: LCS ratio (longest common subsequence)
    matches = 0
    j = 0
    for c in s1_norm:
        while j < len(s2_norm):
            if s2_norm[j] == c:
                matches += 1
                j += 1
                break
            j += 1
    lcs_score = (2.0 * matches) / (len(s1_norm) + len(s2_norm))
    
    # Method 2: Character frequency overlap (handles reordering/OCR scramble)
    from collections import Counter
    c1 = Counter(s1_norm)
    c2 = Counter(s2_norm)
    common = sum((c1 & c2).values())  # Intersection with counts
    total = sum((c1 | c2).values())   # Union with counts
    freq_score = common / total if total > 0 else 0
    
    # Method 3: N-gram overlap (2-char and 3-char sequences)
    def get_ngrams(s, n):
        return set(s[i:i+n] for i in range(max(0, len(s)-n+1)))
    
    bigrams1 = get_ngrams(s1_norm, 2)
    bigrams2 = get_ngrams(s2_norm, 2)
    trigrams1 = get_ngrams(s1_norm, 3)
    trigrams2 = get_ngrams(s2_norm, 3)
    
    bigram_score = len(bigrams1 & bigrams2) / len(bigrams1 | bigrams2) if bigrams1 and bigrams2 else 0
    trigram_score = len(trigrams1 & trigrams2) / len(trigrams1 | trigrams2) if trigrams1 and trigrams2 else 0
    
    # Method 4: Edit distance based score (Levenshtein-like)
    # Simple approximation: penalize differences
    def edit_sim(a, b):
        if len(a) > len(b):
            a, b = b, a
        if len(a) == 0:
            return 0.0
        # Count matching chars at each position (with some shift tolerance)
        matches = 0
        for i, c in enumerate(a):
            if i < len(b) and b[i] == c:
                matches += 1
            elif i > 0 and i-1 < len(b) and b[i-1] == c:
                matches += 0.5
            elif i+1 < len(b) and b[i+1] == c:
                matches += 0.5
        return matches / len(b)
    
    edit_score = edit_sim(s1_norm, s2_norm)
    
    # Combine all scores - take the best approach
    combined = 0.3 * lcs_score + 0.25 * freq_score + 0.2 * bigram_score + 0.15 * trigram_score + 0.1 * edit_score
    
    return max(lcs_score, freq_score, bigram_score * 1.1, combined)


def strip_team_prefix(name: str) -> str:
    """Strip team prefix from name for comparison."""
    if not name:
        return name
    # Strip common team tag patterns (2-4 uppercase/mixed chars followed by space)
    import re
    # Match patterns like "NRG ", "FNC ", "TNC ", etc.
    match = re.match(r'^[A-Za-z0-9]{2,4}\s+', name)
    if match:
        remainder = name[match.end():]
        if len(remainder) >= 2:
            return remainder
    return name


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


@dataclass
class KillfeedEntry:
    """Represents a single parsed killfeed entry."""
    killer_name: str = ""
    killer_team: str = ""  # "left" or "right" based on color
    victim_name: str = ""
    victim_team: str = ""
    weapon: str = ""
    assisters: List[str] = field(default_factory=list)
    is_headshot: bool = False
    row_idx: int = 0
    confidence: float = 0.0


# Team color ranges in HSV for killfeed detection
# Calibrated for VCT broadcast: FNC uses teal/cyan, NRG uses orange
TEAM_COLORS = {
    "teal": {  # FNC / Defenders - cyan/teal team tags
        "lower": np.array([75, 80, 120]),
        "upper": np.array([105, 255, 255])
    },
    "orange": {  # NRG / Attackers - orange team tags
        "lower": np.array([5, 120, 150]),
        "upper": np.array([25, 255, 255])
    },
    "red": {  # Alternative red range for some broadcasts
        "lower1": np.array([0, 100, 120]),
        "upper1": np.array([8, 255, 255]),
        "lower2": np.array([165, 100, 120]),
        "upper2": np.array([179, 255, 255])
    },
    "white": {  # Weapon icons and player name text
        "lower": np.array([0, 0, 180]),
        "upper": np.array([180, 40, 255])
    }
}

# Killfeed row detection parameters
# Based on VCT broadcast screenshots: rows are ~28-38px tall at 1080p
KILLFEED_ROW_HEIGHT_RANGE = (20, 50)  # pixels at 1080p
KILLFEED_ROW_MIN_WIDTH_RATIO = 0.4    # min width as ratio of ROI width
KILLFEED_MAX_ROWS = 6                  # max visible killfeed entries

# Known team tags for OCR post-processing
# Player names are now fetched dynamically from the PostgreSQL database
# via: SELECT nickname FROM esports_players;
KNOWN_TEAM_TAGS = ["NRG", "FNC"]

# Time-based deduplication window (ms) - kills stay visible in killfeed for several seconds
# Use a moderate window - too short = duplicates, too long = misses separate kills
KILL_DEDUP_WINDOW_MS = 6000  # Extended to catch same-victim duplicates within 6 seconds


class KillfeedRowSegmenter:
    """Segments the killfeed ROI into individual row entries."""
    
    def __init__(self, roi_height: int):
        self.roi_height = roi_height
        self.estimated_row_height = roi_height // KILLFEED_MAX_ROWS
    
    def segment_rows(self, roi_bgr) -> List[Tuple[int, int, np.ndarray]]:
        """
        Returns list of (y_start, y_end, row_image) for each detected row.
        Rows are returned top-to-bottom (most recent kill first).
        Uses color-based detection to find killfeed entries (looking for team tag colors).
        """
        h, w = roi_bgr.shape[:2]
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        
        # Create combined mask for team colors (teal + orange)
        teal_mask = cv2.inRange(hsv, TEAM_COLORS["teal"]["lower"], 
                                TEAM_COLORS["teal"]["upper"])
        orange_mask = cv2.inRange(hsv, TEAM_COLORS["orange"]["lower"], 
                                  TEAM_COLORS["orange"]["upper"])
        red_mask1 = cv2.inRange(hsv, TEAM_COLORS["red"]["lower1"], 
                                TEAM_COLORS["red"]["upper1"])
        red_mask2 = cv2.inRange(hsv, TEAM_COLORS["red"]["lower2"], 
                                TEAM_COLORS["red"]["upper2"])
        
        # Combine all team color masks
        color_mask = cv2.bitwise_or(teal_mask, orange_mask)
        color_mask = cv2.bitwise_or(color_mask, red_mask1)
        color_mask = cv2.bitwise_or(color_mask, red_mask2)
        
        # Also detect white text (player names)
        white_mask = cv2.inRange(hsv, TEAM_COLORS["white"]["lower"], 
                                 TEAM_COLORS["white"]["upper"])
        combined_mask = cv2.bitwise_or(color_mask, white_mask)
        
        # Dilate to connect nearby regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        dilated = cv2.dilate(combined_mask, kernel, iterations=1)
        
        # Project horizontally to find rows with content
        h_proj = np.sum(dilated, axis=1).astype(np.float32)
        
        # Smooth the projection
        kernel_size = max(3, self.estimated_row_height // 4)
        if kernel_size % 2 == 0:
            kernel_size += 1
        h_proj_smooth = cv2.GaussianBlur(h_proj.reshape(-1, 1), 
                                          (1, kernel_size), 0).flatten()
        
        # Find row boundaries by looking for regions with content
        # Use a lower threshold to catch more rows
        max_val = np.max(h_proj_smooth)
        threshold = max_val * 0.15 if max_val > 0 else 0
        
        rows = []
        in_row = False
        row_start = 0
        
        for y in range(h):
            if h_proj_smooth[y] > threshold:
                if not in_row:
                    row_start = y
                    in_row = True
            else:
                if in_row:
                    row_end = y
                    row_height = row_end - row_start
                    if KILLFEED_ROW_HEIGHT_RANGE[0] <= row_height <= KILLFEED_ROW_HEIGHT_RANGE[1]:
                        rows.append((row_start, row_end, roi_bgr[row_start:row_end, :]))
                    elif row_height > KILLFEED_ROW_HEIGHT_RANGE[1]:
                        # Row too tall - might be multiple merged rows, try to split
                        num_splits = row_height // 35  # ~35px per row
                        split_h = row_height // max(1, num_splits)
                        for i in range(num_splits):
                            y1 = row_start + i * split_h
                            y2 = row_start + (i + 1) * split_h
                            if KILLFEED_ROW_HEIGHT_RANGE[0] <= (y2 - y1) <= KILLFEED_ROW_HEIGHT_RANGE[1]:
                                rows.append((y1, y2, roi_bgr[y1:y2, :]))
                    in_row = False
        
        # Handle case where last row extends to bottom
        if in_row:
            row_height = h - row_start
            if KILLFEED_ROW_HEIGHT_RANGE[0] <= row_height <= KILLFEED_ROW_HEIGHT_RANGE[1]:
                rows.append((row_start, h, roi_bgr[row_start:h, :]))
        
        return rows[:KILLFEED_MAX_ROWS]


class KillfeedTextReader:
    """Reads text from killfeed entries using OCR."""
    
    def __init__(self):
        self.reader = None
        self._init_reader()
    
    def _init_reader(self):
        """Initialize the best available OCR reader."""
        # Try to import and initialize EasyOCR lazily. Importing EasyOCR
        # will import torch; on some Windows systems that can raise an
        # OSError (DLL init failure). Catch all exceptions and fall back
        # to pytesseract when possible.
        try:
            import easyocr as _easyocr
            try:
                self.reader = _easyocr.Reader(['en'], gpu=False, verbose=False)
                self.backend = "easyocr"
                print("Using EasyOCR for killfeed text detection")
                return
            except Exception as e:
                print(f"EasyOCR Reader init failed: {e}")
        except Exception as e:
            # Could be ImportError or OSError from torch DLL load
            print(f"EasyOCR import/init failed: {e}")

        # Verify Tesseract engine is available in PATH before selecting pytesseract
        if HAS_TESSERACT:
            try:
                # Prefer shutil.which but also tolerate checking via subprocess
                if shutil.which("tesseract") is not None:
                    self.backend = "tesseract"
                    print("Using Tesseract for killfeed text detection")
                    return
                # Fallback: try calling tesseract --version
                proc = subprocess.run(["tesseract", "--version"], capture_output=True, text=True)
                if proc.returncode == 0:
                    self.backend = "tesseract"
                    print("Using Tesseract for killfeed text detection")
                    return
                else:
                    print("Tesseract binary not found in PATH (subprocess returned non-zero)")
            except Exception as e:
                print(f"Tesseract detection failed: {e}")

        self.backend = "none"
        print("No OCR available - using color-based detection only")
    
    def preprocess_for_ocr(self, row_img: np.ndarray, white_only: bool = True) -> np.ndarray:
        """Preprocess image for better OCR results.
        
        Args:
            row_img: BGR image of killfeed row
            white_only: If True, isolate only white/bright text (player names)
                       If False, process all text (including colored team tags)
        """
        # Upscale first for better quality
        scale = 2
        img = cv2.resize(row_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        if white_only:
            # Isolate WHITE TEXT ONLY (player names are white, team tags are colored)
            # Convert to HSV to detect white/bright pixels
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # White text has: low saturation, high value
            # Balance between catching all text vs excluding colored tags
            # Teal (FNC): H=75-105, S=80-255 (high saturation)
            # Orange (NRG): H=5-25, S=120-255 (high saturation)
            # White text: any H, S<50, V>160
            lower_white = np.array([0, 0, 160])    # Any hue, low sat, bright
            upper_white = np.array([180, 50, 255]) # S<50 excludes most colored text
            
            # Create mask for white pixels only
            white_mask = cv2.inRange(hsv, lower_white, upper_white)
            
            # Also catch anti-aliased edges
            lower_aa = np.array([0, 0, 130])
            upper_aa = np.array([180, 45, 180])
            aa_mask = cv2.inRange(hsv, lower_aa, upper_aa)
            
            # Combine masks
            text_mask = cv2.bitwise_or(white_mask, aa_mask)
            
            # Dilate slightly to connect broken characters
            kernel = np.ones((2, 2), np.uint8)
            text_mask = cv2.dilate(text_mask, kernel, iterations=1)
            
            # Create black background with white text
            result = np.zeros_like(img)
            result[text_mask > 0] = [255, 255, 255]
            
            # Convert to grayscale
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        else:
            # Original approach - convert all to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Adaptive thresholding for text
            gray = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
        
        # Denoise
        gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        return gray
    
    def read_text(self, row_img: np.ndarray) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        """
        Read text from a killfeed row image.
        Returns list of (text, confidence, bbox) tuples.
        """
        if self.backend == "none":
            return []
        
        # Enhanced preprocessing for better OCR
        scale = 4  # Higher scale for better character recognition
        upscaled = cv2.resize(row_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # Apply CLAHE for better contrast (helps with game text)
        lab = cv2.cvtColor(upscaled, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Sharpen the image for crisper text edges
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # Blend original enhanced with sharpened (less aggressive sharpening)
        final = cv2.addWeighted(enhanced, 0.5, sharpened, 0.5, 0)
        
        results_list = []
        
        if self.backend == "easyocr" and self.reader:
            try:
                # Use enhanced image for OCR
                results = self.reader.readtext(final, paragraph=False, detail=1)
                for bbox, text, conf in results:
                    if conf > 0.35 and len(text.strip()) >= 2:
                        # Apply gaming font corrections
                        corrected = self._correct_gaming_ocr(text)
                        results_list.append((corrected, conf, tuple(map(int, [
                            bbox[0][0]/scale, bbox[0][1]/scale,
                            (bbox[2][0]-bbox[0][0])/scale, (bbox[2][1]-bbox[0][1])/scale
                        ]))))
                
                return results_list
            except Exception as e:
                print(f"EasyOCR error: {e}")
                return []
        
        return []
    
    def _correct_gaming_ocr(self, text: str) -> str:
        """
        Apply common OCR error corrections for gaming fonts.
        Gaming fonts often cause specific character confusions.
        """
        result = text
        
        # Common OCR substitutions in gaming fonts (order matters)
        corrections = [
            # Lowercase L / uppercase I / digit 1 confusion
            ('lcIe', 'icle'),      # ChronlcIe -> Chronicle
            ('lcle', 'icle'),      # Chronlcle -> Chronicle
            ('lc1e', 'icle'),
            ('Icle', 'icle'),
            # Digit 0 vs letter O
            ('s0m', 'som'),        # Player name s0m
            ('0m', 'om'),
            # Common letter swaps
            ('Ihan', 'than'),      # Elhan -> Ethan
            ('lhan', 'than'),
            ('rash', 'rash'),      # Keep Crashies
            ('Grash', 'Crash'),    # Grashies -> Crashies
            ('grash', 'crash'),
            # Double letters
            ('aaj', 'aj'),         # Kaajak -> Kajak (but Kajaak is correct)
            ('jjer', 'jer'),       # Alfajjer -> Alfajer
            # Prefix issues
            ('Alf', 'Alf'),        # Keep Alfajer
        ]
        
        for wrong, right in corrections:
            if wrong in result:
                result = result.replace(wrong, right)
        
        return result


class KillfeedColorAnalyzer:
    """Analyzes colors in killfeed to determine teams and structure."""
    
    @staticmethod
    def detect_team_colors(row_img: np.ndarray) -> Dict[str, List[Tuple[int, int, int, int]]]:
        """
        Detect team-colored regions (team tags like NRG, FNC) in a killfeed row.
        Returns dict with 'teal' and 'orange' keys containing bbox lists.
        Killfeed format: [Agent] [TeamTag colored] [Name white] [Weapon] [TeamTag colored] [Name white] [Agent]
        """
        hsv = cv2.cvtColor(row_img, cv2.COLOR_BGR2HSV)
        results = {"teal": [], "orange": []}
        
        # Teal/cyan mask (FNC team tag)
        teal_mask = cv2.inRange(hsv, TEAM_COLORS["teal"]["lower"], 
                                TEAM_COLORS["teal"]["upper"])
        
        # Orange mask (NRG team tag)
        orange_mask = cv2.inRange(hsv, TEAM_COLORS["orange"]["lower"], 
                                  TEAM_COLORS["orange"]["upper"])
        
        # Also check red range as fallback
        red_mask1 = cv2.inRange(hsv, TEAM_COLORS["red"]["lower1"], 
                                TEAM_COLORS["red"]["upper1"])
        red_mask2 = cv2.inRange(hsv, TEAM_COLORS["red"]["lower2"], 
                                TEAM_COLORS["red"]["upper2"])
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        # Combine orange and red into one "attacker" mask
        orange_mask = cv2.bitwise_or(orange_mask, red_mask)
        
        # Clean up masks
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        
        for color_name, mask in [("teal", teal_mask), ("orange", orange_mask)]:
            cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, 
                                            cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                # Team tags are small colored text regions
                if w > 8 and h > 4:
                    results[color_name].append((x, y, w, h))
        
        return results
    
    @staticmethod
    def detect_weapon_icon(row_img: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect the weapon icon (usually white/gray in the middle).
        Returns bbox of weapon icon region or None.
        """
        hsv = cv2.cvtColor(row_img, cv2.COLOR_BGR2HSV)
        h, w = row_img.shape[:2]
        
        # White/bright icon detection
        white_mask = cv2.inRange(hsv, TEAM_COLORS["white"]["lower"], 
                                  TEAM_COLORS["white"]["upper"])
        
        # Look for weapon icon in the middle portion of the row
        middle_x = w // 4
        middle_w = w // 2
        middle_region = white_mask[:, middle_x:middle_x + middle_w]
        
        contours, _ = cv2.findContours(middle_region, cv2.RETR_EXTERNAL, 
                                        cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the largest white blob in the middle (weapon icon)
        best_bbox = None
        best_area = 0
        for c in contours:
            area = cv2.contourArea(c)
            if area > best_area and area > 50:
                x, y, w_c, h_c = cv2.boundingRect(c)
                best_bbox = (x + middle_x, y, w_c, h_c)
                best_area = area
        
        return best_bbox
    
    @staticmethod
    def detect_headshot_icon(row_img: np.ndarray) -> bool:
        """Detect if there's a headshot indicator (skull icon or special marker)."""
        # Headshot kills often have a distinctive red/orange tint or skull icon
        # This is a simplified check - may need refinement
        hsv = cv2.cvtColor(row_img, cv2.COLOR_BGR2HSV)
        
        # Look for orange/yellow headshot marker
        hs_lower = np.array([10, 100, 150])
        hs_upper = np.array([25, 255, 255])
        hs_mask = cv2.inRange(hsv, hs_lower, hs_upper)
        
        return cv2.countNonZero(hs_mask) > 20


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
    """
    Enhanced killfeed detector that parses individual kill entries.
    
    Killfeed behavior:
    - New kills appear at the BOTTOM of the visible stack
    - Old kills are at the TOP and disappear after ~5 seconds
    - When top kill expires, everything shifts UP
    
    Detection strategy:
    - Only process when row count INCREASES (new kill appeared)
    - The new kill is always the BOTTOM-most row
    - Use row count changes to know when to look for new kills
    """
    def __init__(self, target_fps: float):
        super().__init__("killfeed", target_fps, change_mse_threshold=60.0)
        self.last_row_count = 0
        self.row_segmenter: Optional[KillfeedRowSegmenter] = None
        self.text_reader: Optional[KillfeedTextReader] = None
        self.color_analyzer = KillfeedColorAnalyzer()
        
        # Track the last time we emitted a kill event  
        self.last_kill_emit_ms = 0.0
        
        # Minimum time between kill events for the same row position
        self.min_kill_interval_ms = 500.0
        
        # Track emitted kills by signature (time, killer_team, victim_team, killer_name, victim_name)
        # This prevents duplicate emissions while allowing for OCR variations
        self.recent_kill_signatures: List[Tuple[float, str, str, str, str]] = []
        
        # Initialize OCR reader lazily
        self._ocr_initialized = False
    
    def _cleanup_signatures(self, t_ms: float):
        """Remove old kill signatures."""
        cutoff = t_ms - KILL_DEDUP_WINDOW_MS
        self.recent_kill_signatures = [
            (t, kt, vt, kn, vn) for (t, kt, vt, kn, vn) in self.recent_kill_signatures 
            if t > cutoff
        ]
    
    def _has_recent_kill(self, t_ms: float, killer_team: str, victim_team: str, 
                         killer_name: str, victim_name: str) -> bool:
        """
        Check if we recently emitted a kill that looks like this one.
        Uses name normalization and similarity to detect duplicates even with OCR variations.
        Also detects swapped killer/victim which indicates duplicate with detection errors.
        """
        # Strip team prefixes for comparison
        killer_base = strip_team_prefix(killer_name)
        victim_base = strip_team_prefix(victim_name)
        
        for (sig_t, sig_kt, sig_vt, sig_kn, sig_vn) in self.recent_kill_signatures:
            time_diff = t_ms - sig_t
            
            # Skip if too old
            if time_diff > KILL_DEDUP_WINDOW_MS:
                continue
            
            # Strip team prefixes from stored names too
            sig_killer_base = strip_team_prefix(sig_kn)
            sig_victim_base = strip_team_prefix(sig_vn)
            
            # Calculate name similarities using base names (team prefix stripped)
            killer_sim = calculate_similarity(killer_base, sig_killer_base) if killer_base and sig_killer_base else 0
            victim_sim = calculate_similarity(victim_base, sig_victim_base) if victim_base and sig_victim_base else 0
            
            # Debug output for similarity checks
            if victim_sim > 0.4 or killer_sim > 0.4:
                print(f"  [DEDUP] {killer_name}->{victim_name} vs {sig_kn}->{sig_vn}: "
                      f"killer_sim={killer_sim:.2f}, victim_sim={victim_sim:.2f}, time_diff={time_diff:.0f}ms")
            
            # Also check for SWAPPED names (killer<->victim confusion)
            killer_as_victim_sim = calculate_similarity(killer_base, sig_victim_base) if killer_base and sig_victim_base else 0
            victim_as_killer_sim = calculate_similarity(victim_base, sig_killer_base) if victim_base and sig_killer_base else 0
            
            # CRITICAL: If the VICTIM is the same (high similarity), it's likely a duplicate
            # A player can only die once per round, so victim matching is strong signal
            if victim_sim > 0.65 and sig_vt == victim_team:
                # Same victim on same team within 6 seconds = definitely duplicate
                if time_diff < 6000:
                    print(f"  [DEDUP] BLOCKED: Same victim ({victim_base} ~ {sig_victim_base})")
                    return True
            
            # Check for exact same teams and similar names
            if sig_kt == killer_team and sig_vt == victim_team:
                # Very high similarity on BOTH = definitely same kill
                if killer_sim > 0.7 and victim_sim > 0.7:
                    if time_diff < 6000:
                        return True
                # Good similarity on BOTH = likely same kill
                elif killer_sim > 0.5 and victim_sim > 0.5:
                    if time_diff < 5000:
                        return True
                # Moderate similarity on BOTH = might be same kill
                elif killer_sim > 0.35 and victim_sim > 0.35:
                    if time_diff < 4000:
                        return True
                # One name is Unknown - require the other to be very similar
                elif (killer_name == "Unknown" and victim_sim > 0.6) or \
                     (victim_name == "Unknown" and killer_sim > 0.6):
                    if time_diff < 2000:
                        return True
                # Very close in time with exact same detected names
                elif killer_sim > 0.85 and time_diff < 1000:
                    return True
                elif victim_sim > 0.85 and time_diff < 1000:
                    return True
            
            # Check for SWAPPED detection (same players, different roles)
            # This catches detection errors where killer/victim are confused
            elif killer_as_victim_sim > 0.6 and victim_as_killer_sim > 0.6:
                # Same two players but roles swapped - likely a detection error
                if time_diff < 5000:
                    return True
            elif (killer_as_victim_sim > 0.5 and victim_as_killer_sim > 0.5):
                # Probably same kill with swapped detection
                if time_diff < 4000:
                    return True
            
            # Check for TEAM SWAP: same names, different teams
            # This catches when OCR correctly gets names but team colors are confused
            # e.g., Kill 2: "FNC Chronicle -> NRG Ethan" vs Kill 4: "NRG Chronicle -> FNC Ethan"
            elif sig_kt != killer_team and sig_vt != victim_team:
                # Teams are different - check if names are the same
                if killer_sim > 0.6 and victim_sim > 0.6:
                    # Same names, teams swapped = same event, team detection error
                    if time_diff < 5000:
                        return True
                elif killer_sim > 0.5 and victim_sim > 0.5:
                    if time_diff < 3000:
                        return True
        
        return False
    
    def _init_ocr(self):
        """Lazily initialize OCR to avoid slow startup."""
        if not self._ocr_initialized:
            self.text_reader = KillfeedTextReader()
            self._ocr_initialized = True
    
    def _parse_row(self, row_idx: int, row_img: np.ndarray, debug: bool = True) -> Optional[KillfeedEntry]:
        """
        Parse a single killfeed row to extract kill information.
        
        VALORANT killfeed format:
        [Killer Team Tag] [Killer Name] [Weapon Icon] [Victim Team Tag] [Victim Name]
        
        Strategy:
        1. Detect all team color regions (team tags show in teal or orange)
        2. Detect all names via OCR
        3. Associate each name with its NEAREST team color region
        4. Use positions: killer is on left side, victim is on right side
        """
        entry = KillfeedEntry(row_idx=row_idx)
        h, w = row_img.shape[:2]
        
        # Analyze colors to determine killer/victim teams
        team_colors = self.color_analyzer.detect_team_colors(row_img)
        
        teal_regions = team_colors["teal"]
        orange_regions = team_colors["orange"]
        
        if debug:
            print(f"  [Row {row_idx}] Teal regions: {len(teal_regions)}, Orange regions: {len(orange_regions)}")
        
        # Detect weapon icon as center divider
        weapon_bbox = self.color_analyzer.detect_weapon_icon(row_img)
        if weapon_bbox:
            wx, wy, ww, wh = weapon_bbox
            weapon_center_x = wx + ww // 2
            entry.weapon = "detected"
            if debug:
                print(f"  [Row {row_idx}] Weapon at x={weapon_center_x}")
        else:
            # Fallback: assume weapon is around 45% from left
            weapon_center_x = int(w * 0.45)
        
        # Collect ALL colored regions with full position info
        all_color_regions = []
        for x, y, rw, rh in teal_regions:
            region_center = x + rw // 2
            all_color_regions.append({
                "color": "teal",
                "x": x,
                "center_x": region_center,
                "right_edge": x + rw,
                "width": rw
            })
        for x, y, rw, rh in orange_regions:
            region_center = x + rw // 2
            all_color_regions.append({
                "color": "orange",
                "x": x,
                "center_x": region_center,
                "right_edge": x + rw,
                "width": rw
            })
        
        # Sort by x position
        all_color_regions.sort(key=lambda r: r["x"])
        
        if debug and all_color_regions:
            print(f"  [Row {row_idx}] Color regions (L->R): {[(r['color'], r['x'], r['right_edge']) for r in all_color_regions]}")
        
        # Detect headshot
        entry.is_headshot = self.color_analyzer.detect_headshot_icon(row_img)
        
        # Try OCR for player names FIRST
        self._init_ocr()
        name_regions = []  # Will store: {"name": str, "x": int, "center_x": int, "raw_name": str}
        
        if self.text_reader and self.text_reader.backend != "none":
            try:
                texts = self.text_reader.read_text(row_img)
                
                # Filter out pure team tags and short noise from OCR results
                team_tag_like = {'nrg', 'fnc', 'fng', 'nrc', 'mrg', 'enc', 'fc', 'ng', 'nr', 
                                 'wrg', 'inc', 'fnd', 'nrb', 'mrc', 'f;', 'f~'}
                
                def strip_team_prefix(text: str) -> Tuple[str, int]:
                    """
                    Strip team tag prefix and return (cleaned_name, offset_px).
                    Offset estimates how many pixels the team tag took up.
                    """
                    txt = text.strip()
                    # Approximate character width in the killfeed font at our scale
                    char_width = 8  # ~8 pixels per character at 4x scale then down
                    
                    # Check common team tag patterns
                    tag_patterns = [
                        (r'^[NnMmWw][Rr][GgCcOo]\s*', 3),   # NRG variants
                        (r'^[FfEeGg][NnMm][Cc]\s*', 3),     # FNC variants
                        (r'^[Ff][Cc]\s*', 2),               # FC
                        (r'^[Nn][Rr]\s*', 2),               # NR
                    ]
                    
                    for pattern, tag_len in tag_patterns:
                        match = re.match(pattern, txt, re.IGNORECASE)
                        if match:
                            cleaned = txt[match.end():]
                            if len(cleaned) >= 2:  # Must have remaining name
                                offset = (tag_len + 1) * char_width  # +1 for space
                                return cleaned, offset
                    
                    return txt, 0
                
                def is_likely_name(text: str) -> bool:
                    """Check if text looks like a player name, not just a team tag."""
                    txt = text.strip().lower()
                    if len(txt) < 3:
                        return False
                    if txt in team_tag_like:
                        return False
                    return True
                
                for text, conf, bbox in texts:
                    if not is_likely_name(text):
                        continue
                    x, y, bw, bh = bbox
                    
                    # Strip team tag and adjust position
                    cleaned_name, offset = strip_team_prefix(text)
                    adjusted_x = x + offset
                    
                    name_regions.append({
                        "name": cleaned_name,
                        "raw_name": text,
                        "x": adjusted_x,  # Position of actual name (after tag)
                        "raw_x": x,       # Original position
                        "center_x": adjusted_x + (bw - offset) // 2,
                        "conf": conf
                    })
                
            except Exception as e:
                pass  # OCR failed
        
        # Sort names by x position
        name_regions.sort(key=lambda n: n["x"])
        
        if debug and name_regions:
            print(f"  [Row {row_idx}] Names detected: {[(n['name'], n['x']) for n in name_regions]}")
        
        # Now associate names with team colors based on proximity
        # Strategy: Find the closest team color region to the LEFT of each name
        # (team tags appear just before player names)
        
        def find_team_for_name(name_region: dict) -> Optional[str]:
            """Find the team color for a name by finding the closest color region that starts near/before it."""
            name_x = name_region["x"]
            best_color = None
            best_distance = float('inf')
            
            for cr in all_color_regions:
                # The team tag's START position should be near (but before or at) the name
                # Team tags like "NRG" or "FNC" appear right before the player name
                color_start = cr["x"]
                
                # The team tag should start before or very close to where the name starts
                # Allow the color region to start up to 50px before the name (team tags ~30px wide)
                # or up to 10px after (in case of OCR bbox overlap)
                distance = name_x - color_start
                
                # Team tag should be within reasonable distance to the left of name
                if -10 <= distance <= 100:  # Tag can be slightly after or up to 100px before
                    if abs(distance) < best_distance:
                        best_distance = abs(distance)
                        best_color = cr["color"]
                
                # Also check if the name falls WITHIN the color region
                # (color detection might span the entire team tag + name area)
                color_end = cr["right_edge"]
                if color_start <= name_x <= color_end:
                    # Name is within color region - this is a strong match
                    if best_distance > 5:  # Override only if we don't have a closer match
                        best_distance = 0
                        best_color = cr["color"]
            
            return best_color
        
        # Process names
        if len(name_regions) >= 2:
            # Two or more names found
            killer_name_region = name_regions[0]   # Leftmost
            victim_name_region = name_regions[-1]  # Rightmost
            
            entry.killer_name = killer_name_region["name"]
            entry.victim_name = victim_name_region["name"]
            
            # Find team for each name based on nearest color region
            killer_team = find_team_for_name(killer_name_region)
            victim_team = find_team_for_name(victim_name_region)
            
            if killer_team and victim_team:
                entry.killer_team = killer_team
                entry.victim_team = victim_team
                # Only high confidence if teams are DIFFERENT (valid cross-team kill)
                if killer_team != victim_team:
                    entry.confidence = 0.85
                else:
                    # Same team detected for both - likely detection error, low confidence
                    entry.confidence = 0.4
                if debug:
                    print(f"  [Row {row_idx}] Team assignment: {killer_team}->{entry.killer_name} kills {victim_team}->{entry.victim_name}")
            elif killer_team:
                entry.killer_team = killer_team
                # Infer victim team as opposite
                entry.victim_team = "orange" if killer_team == "teal" else "teal"
                entry.confidence = 0.75
                if debug:
                    print(f"  [Row {row_idx}] Inferred: {killer_team}->{entry.killer_name} kills {entry.victim_team}->{entry.victim_name}")
            elif victim_team:
                entry.victim_team = victim_team
                # Infer killer team as opposite
                entry.killer_team = "orange" if victim_team == "teal" else "teal"
                entry.confidence = 0.75
                if debug:
                    print(f"  [Row {row_idx}] Inferred: {entry.killer_team}->{entry.killer_name} kills {victim_team}->{entry.victim_name}")
            else:
                # Fallback to leftmost/rightmost color region method
                if len(all_color_regions) >= 2:
                    entry.killer_team = all_color_regions[0]["color"]
                    entry.victim_team = all_color_regions[-1]["color"]
                    entry.confidence = 0.6
                    if debug:
                        print(f"  [Row {row_idx}] Fallback L/R: {entry.killer_team} -> {entry.victim_team}")
                elif len(all_color_regions) == 1:
                    # Only one color - can't determine teams reliably
                    entry.confidence = 0.3
                else:
                    entry.confidence = 0.2
                    
        elif len(name_regions) == 1:
            # Only one name - determine if killer or victim by position
            name_region = name_regions[0]
            name_x = name_region["center_x"]
            
            # Use weapon position as divider
            if name_x < weapon_center_x:
                entry.killer_name = name_region["name"]
                killer_team = find_team_for_name(name_region)
                if killer_team:
                    entry.killer_team = killer_team
                    entry.victim_team = "orange" if killer_team == "teal" else "teal"
                    entry.confidence = 0.6
                    if debug:
                        print(f"  [Row {row_idx}] Single name (left): {killer_team}->{entry.killer_name}")
                else:
                    entry.confidence = 0.3
            else:
                entry.victim_name = name_region["name"]
                victim_team = find_team_for_name(name_region)
                if victim_team:
                    entry.victim_team = victim_team
                    entry.killer_team = "orange" if victim_team == "teal" else "teal"
                    entry.confidence = 0.6
                    if debug:
                        print(f"  [Row {row_idx}] Single name (right): ?->{entry.victim_name} ({victim_team})")
                else:
                    entry.confidence = 0.3
        else:
            # No names found - use color positions only
            if len(all_color_regions) >= 2:
                entry.killer_team = all_color_regions[0]["color"]
                entry.victim_team = all_color_regions[-1]["color"]
                entry.confidence = 0.4
            elif len(all_color_regions) == 1:
                entry.killer_team = all_color_regions[0]["color"]
                entry.confidence = 0.2
            else:
                entry.confidence = 0.1

        # If we have names or reasonable confidence, return entry
        if (entry.killer_name or entry.victim_name) or entry.confidence >= 0.3:
            return entry

        return None

    def process(self, t_ms: float, roi_bgr) -> List[Event]:
        """
        Process killfeed ROI and extract kill events.
        
        Strategy:
        - Process ALL visible rows on each frame
        - Use name-based deduplication with fuzzy matching
        - Track recent kills by (team combo + name similarity) to avoid duplicates
        - Allow new kills to be detected even if row count doesn't change
        """
        h, w = roi_bgr.shape[:2]
        
        # Initialize row segmenter if needed
        if self.row_segmenter is None:
            self.row_segmenter = KillfeedRowSegmenter(h)
        
        # Cleanup old signatures
        self._cleanup_signatures(t_ms)
        
        # Segment into individual rows
        rows = self.row_segmenter.segment_rows(roi_bgr)
        row_count = len(rows)
        
        evs: List[Event] = []
        
        # Process ALL visible rows, using deduplication to avoid repeats
        for i, (y_start, y_end, row_img) in enumerate(rows):
            # Parse the row
            entry = self._parse_row(i, row_img)
            if not entry or entry.confidence < 0.7:  # Require at least 0.7 confidence
                continue
            
            # Get team info (colors) for team-aware name cleaning
            killer_team = entry.killer_team or "unknown"
            victim_team = entry.victim_team or "unknown"
            
            # Require different teams for a valid kill (cross-team kills only)
            if killer_team == victim_team:
                continue
            
            # Clean up player names with team color info for better matching
            killer_clean = clean_player_name(entry.killer_name, killer_team) if entry.killer_name else "Unknown"
            victim_clean = clean_player_name(entry.victim_name, victim_team) if entry.victim_name else "Unknown"
            
            # Skip if BOTH names are Unknown (we need at least one known name to validate)
            if killer_clean == "Unknown" and victim_clean == "Unknown":
                continue
            
            # Check for recent duplicate using names + team combo
            if self._has_recent_kill(t_ms, killer_team, victim_team, killer_clean, victim_clean):
                continue
            
            # Store cleaned names
            entry.killer_name = killer_clean
            entry.victim_name = victim_clean
            
            # Record this kill signature (with names for better dedup)
            self.recent_kill_signatures.append(
                (t_ms, killer_team, victim_team, killer_clean, victim_clean)
            )
            
            # Main kill event
            evs.append(Event(
                t_ms=t_ms,
                type="KILL_EVENT",
                roi=self.name,
                payload={
                    "killer_name": entry.killer_name or "Unknown",
                    "killer_team": entry.killer_team,
                    "victim_name": entry.victim_name or "Unknown",
                    "victim_team": entry.victim_team,
                    "weapon": entry.weapon,
                    "is_headshot": entry.is_headshot,
                    "confidence": entry.confidence,
                    "assisters": entry.assisters
                }
            ))
            
            # Also emit a death event for the victim
            evs.append(Event(
                t_ms=t_ms,
                type="DEATH_EVENT",
                roi=self.name,
                payload={
                    "player_name": entry.victim_name or "Unknown",
                    "player_team": entry.victim_team,
                    "killed_by": entry.killer_name or "Unknown",
                    "weapon": entry.weapon,
                    "was_headshot": entry.is_headshot
                }
            ))
        
        # Also emit summary if row count changed
        if row_count != self.last_row_count:
            evs.append(Event(
                t_ms=t_ms,
                type="KILLFEED_UPDATE",
                roi=self.name,
                payload={
                    "visible_rows": row_count,
                    "previous_rows": self.last_row_count,
                    "delta": row_count - self.last_row_count
                }
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
# Round Tracking & Timeline Generation
# -----------------------------
@dataclass
class RoundInfo:
    """Information about a single round."""
    round_number: int
    start_time_ms: float
    end_time_ms: Optional[float] = None
    left_score: int = 0
    right_score: int = 0
    winner: Optional[str] = None  # "left" or "right"
    spike_planted: bool = False
    spike_plant_time_ms: Optional[float] = None
    spike_defused: bool = False
    events: List[Event] = field(default_factory=list)


class RoundTracker:
    """
    Tracks round progression and organizes events by round.
    Uses PLAYER_RESPAWN events (all 10 players respawn) to detect round starts.
    Uses score changes or all-dead scenarios to detect round ends.
    """
    
    def __init__(self):
        self.current_round: Optional[RoundInfo] = None
        self.completed_rounds: List[RoundInfo] = []
        self.round_number = 0
        
        # Track player alive states
        self.players_alive = {
            "left": set(range(1, 6)),   # {1, 2, 3, 4, 5}
            "right": set(range(1, 6))
        }
        self.last_respawn_time = -1e9
        # small debounce state to avoid spurious end/start churn
        self._last_round_change_ms = -1e9
        
    def process_event(self, event: Event) -> Optional[Event]:
        """
        Process an event and track round state.
        May emit ROUND_START or ROUND_END events.
        Returns any generated round-related event.
        """
        # Attach current round to event
        if self.current_round:
            event.payload["round_number"] = self.current_round.round_number
            self.current_round.events.append(event)
        
        # Handle player deaths
        if event.type == "PLAYER_DEATH":
            team = event.payload.get("team")
            slot = event.payload.get("slot")
            if team and slot:
                self.players_alive[team].discard(slot)
                # If no round active, do not end a round based on noisy death reports
                if self.current_round is None:
                    return None

                # Avoid ending the round immediately after it started due to noisy detections
                if (event.t_ms - (self.current_round.start_time_ms or 0)) < 500:
                    return None

                # Check for team wipe (only end when a round is active and passes debounce)
                if len(self.players_alive["left"]) == 0:
                    self._last_round_change_ms = event.t_ms
                    return self._end_round(event.t_ms, "right")
                elif len(self.players_alive["right"]) == 0:
                    self._last_round_change_ms = event.t_ms
                    return self._end_round(event.t_ms, "left")
        
        # Handle player respawns (round start indicator)
        elif event.type == "PLAYER_RESPAWN":
            team = event.payload.get("team")
            slot = event.payload.get("slot")
            if team and slot:
                self.players_alive[team].add(slot)
            
            # If most players are alive again, it's a new round
            total_alive = len(self.players_alive["left"]) + len(self.players_alive["right"])
            
            # New round detection: 10 players alive within short time window
            # Only start a round if one is not already active
            if self.current_round is None and total_alive >= 9:
                time_since_last = event.t_ms - self.last_respawn_time
                if time_since_last > 5000:  # More than 5 seconds since last mass respawn
                    self.last_respawn_time = event.t_ms
                    # Also guard against immediate rapid churn
                    if (event.t_ms - self._last_round_change_ms) > 500:
                        self._last_round_change_ms = event.t_ms
                        return self._start_round(event.t_ms)

            # Handle explicit round transition signals from top HUD (brightness flash)
            elif event.type == "ROUND_TRANSITION":
                # If no round is active, treat this as a likely round start
                if self.current_round is None:
                    # debounce frequent transitions
                    if (event.t_ms - self._last_round_change_ms) > 500:
                        self._last_round_change_ms = event.t_ms
                        return self._start_round(event.t_ms)
                else:
                    # If a round is active and the transition is large, consider it an end
                    # Only end if the round has been active for a reasonable minimum time
                    rt = event.payload.get("brightness_delta", 0)
                    if (event.t_ms - (self.current_round.start_time_ms or 0)) > 1000 and abs(rt) > 40:
                        self._last_round_change_ms = event.t_ms
                        # Winner unknown here; leave winner=None and let other logic fill it
                        return self._end_round(event.t_ms, "unknown")
        
        # Handle spike plant
        elif event.type == "SPIKE_PLANTED":
            if self.current_round:
                self.current_round.spike_planted = True
                self.current_round.spike_plant_time_ms = event.t_ms
        
        # Handle spike defuse
        elif event.type == "SPIKE_DEFUSED":
            if self.current_round:
                self.current_round.spike_defused = True
                return self._end_round(event.t_ms, "defense")  # Defenders win
        
        return None
    
    def _start_round(self, t_ms: float) -> Event:
        """Start a new round."""
        # End previous round if any
        if self.current_round and self.current_round.end_time_ms is None:
            self.current_round.end_time_ms = t_ms
            self.completed_rounds.append(self.current_round)
        
        self.round_number += 1
        self.current_round = RoundInfo(
            round_number=self.round_number,
            start_time_ms=t_ms
        )
        
        # Reset alive tracking
        self.players_alive = {
            "left": set(range(1, 6)),
            "right": set(range(1, 6))
        }
        
        return Event(
            t_ms=t_ms,
            type="ROUND_START",
            roi="round_tracker",
            payload={"round_number": self.round_number}
        )
    
    def _end_round(self, t_ms: float, winner: str) -> Event:
        """End the current round."""
        if self.current_round:
            self.current_round.end_time_ms = t_ms
            self.current_round.winner = winner
            self.completed_rounds.append(self.current_round)
            
            round_num = self.current_round.round_number
            self.current_round = None
            
            return Event(
                t_ms=t_ms,
                type="ROUND_END",
                roi="round_tracker",
                payload={
                    "round_number": round_num,
                    "winner": winner
                }
            )
        return None
    
    def get_timeline(self) -> List[Dict[str, Any]]:
        """Generate a timeline summary of all rounds."""
        timeline = []
        
        for rnd in self.completed_rounds:
            round_summary = {
                "round_number": rnd.round_number,
                "start_time_ms": rnd.start_time_ms,
                "end_time_ms": rnd.end_time_ms,
                "duration_ms": (rnd.end_time_ms - rnd.start_time_ms) if rnd.end_time_ms else None,
                "winner": rnd.winner,
                "spike_planted": rnd.spike_planted,
                "spike_plant_time_ms": rnd.spike_plant_time_ms,
                "events": []
            }
            
            # Summarize events by type
            event_summary: Dict[str, List] = {}
            for ev in rnd.events:
                if ev.type not in event_summary:
                    event_summary[ev.type] = []
                event_summary[ev.type].append({
                    "t_ms": ev.t_ms,
                    "payload": ev.payload
                })
            
            round_summary["events_by_type"] = event_summary
            
            # Count kills per team (map color to team side)
            kills = {"teal": 0, "orange": 0}
            for ev in rnd.events:
                if ev.type == "KILL_EVENT":
                    killer_team = ev.payload.get("killer_team", "")
                    if killer_team in kills:
                        kills[killer_team] += 1
            
            round_summary["kills_per_team"] = kills
            
            timeline.append(round_summary)
        
        return timeline


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


def extract_match_players_from_frame(frame: np.ndarray, text_reader) -> List[str]:
    """
    Extract player names from the player card boxes on both sides of the screen.
    This reads the 'player_name' subregion from each of the 10 player cards.
    
    Note: This function has limited accuracy due to complex backgrounds on player cards.
    Consider using collect_players_from_killfeed() instead for more reliable results.
    
    Args:
        frame: A video frame (ideally from start of match when all players visible)
        text_reader: Initialized KillfeedTextReader for OCR
        
    Returns:
        List of extracted player names (up to 10)
    """
    h, w = frame.shape[:2]
    extracted_names = []
    
    # Process all 10 player cards (5 left, 5 right)
    player_cards = [f"left_player_{i}" for i in range(1, 6)] + \
                   [f"right_player_{i}" for i in range(1, 6)]
    
    for card_name in player_cards:
        if card_name not in ROI_NORM:
            continue
            
        # Get the player card ROI
        card_roi = ROI_NORM[card_name]
        card_x, card_y, card_w, card_h = roi_to_px(w, h, card_roi)
        card_img = frame[card_y:card_y+card_h, card_x:card_x+card_w]
        
        if card_img.size == 0:
            continue
        
        # Get the player_name subregion within the card
        name_subroi = PLAYER_CARD_SUBREGIONS.get("player_name")
        if not name_subroi:
            continue
        
        # For right-side cards, mirror the x-coordinate
        if "right" in card_name:
            # Mirror: new_x = 1 - (x + w)
            sub_x = 1.0 - (name_subroi[0] + name_subroi[2])
        else:
            sub_x = name_subroi[0]
        
        sub_y = name_subroi[1]
        sub_w = name_subroi[2]
        sub_h = name_subroi[3]
        
        # Convert to pixels within card
        px = int(sub_x * card_w)
        py = int(sub_y * card_h)
        pw = int(sub_w * card_w)
        ph = int(sub_h * card_h)
        
        # Extract the name region
        name_img = card_img[py:py+ph, px:px+pw]
        
        if name_img.size == 0:
            continue
        
        # Preprocess for player card OCR - use the standard read_text method
        try:
            results = text_reader.read_text(name_img)
            if results:
                # Take the text with highest confidence
                best_result = max(results, key=lambda x: x[1])
                raw_name = best_result[0].strip()
                conf = best_result[1]
                
                # Clean up: remove team tags
                cleaned = re.sub(r'^(NRG|FNC|100T|SEN|C9|EG|LOUD|DRX|PRX|FUT|LEV|KRU|G2|TH|GENG|T1|EDG|FPX|BLG)[\s_]?', 
                                '', raw_name, flags=re.IGNORECASE).strip()
                
                # Basic validation - must look like a player name
                if cleaned and len(cleaned) >= 2 and cleaned[0].isalpha():
                    extracted_names.append(cleaned)
                    print(f"  {card_name}: '{raw_name}' -> '{cleaned}' (conf={conf:.2f})")
                    
        except Exception as e:
            print(f"  {card_name}: OCR error - {e}")
    
    return extracted_names


def collect_players_from_killfeed_scan(cap: cv2.VideoCapture, duration_sec: float = 30.0) -> List[str]:
    """
    Collect player names by scanning the killfeed over a portion of the video.
    This is more reliable than reading player cards since killfeed text is cleaner.
    
    Args:
        cap: OpenCV VideoCapture object
        duration_sec: How many seconds of video to scan
        
    Returns:
        List of unique player names found in killfeed
    """
    print("\n" + "="*50)
    print("Collecting player names from killfeed scan...")
    print("="*50)
    
    # Initialize components
    text_reader = KillfeedTextReader()
    
    # Save original position
    original_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Scan the entire video (or up to duration_sec, whichever is shorter)
    start_frame = 0
    max_frames = int(fps * duration_sec) if duration_sec > 0 else total_frames
    end_frame = min(total_frames, max_frames)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Sample every N frames
    sample_interval = int(fps / 2)  # ~2 samples per second for more coverage
    
    all_names = set()
    frame_count = 0
    
    while cap.get(cv2.CAP_PROP_POS_FRAMES) < end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        
        frame_count += 1
        if frame_count % sample_interval != 0:
            continue
        
        # Get killfeed ROI
        if "killfeed" not in ROI_NORM:
            continue
            
        kf_roi = ROI_NORM["killfeed"]
        kf_x, kf_y, kf_w, kf_h = roi_to_px(w, h, kf_roi)
        kf_img = frame[kf_y:kf_y+kf_h, kf_x:kf_x+kf_w]
        
        if kf_img.size == 0:
            continue
        
        # Run OCR on killfeed
        try:
            results = text_reader.read_text(kf_img)
            for text, conf, _ in results:
                if conf < 0.3:
                    continue
                    
                # Extract potential player names
                # Remove team tags and clean up
                cleaned = re.sub(r'^(NRG|FNC|100T|SEN|C9|EG|LOUD|DRX|PRX|FUT|LEV|KRU|G2|TH|GENG|T1|EDG|FPX|BLG)[\s_]?', 
                                '', text.strip(), flags=re.IGNORECASE).strip()
                
                # Basic validation
                if cleaned and 2 <= len(cleaned) <= 15 and cleaned[0].isalpha():
                    # Normalize: remove trailing numbers/symbols
                    cleaned = re.sub(r'[\d\W]+$', '', cleaned)
                    if len(cleaned) >= 2:
                        all_names.add(cleaned)
                        
        except Exception:
            pass
    
    # Restore position
    cap.set(cv2.CAP_PROP_POS_FRAMES, original_pos)
    
    # Filter to likely player names (fuzzy match against database)
    db = get_player_database()
    db_names = db.get_player_names()
    
    # OCR-aware normalization for comparison (same as fuzzy_match_player)
    def ocr_normalize(s: str) -> str:
        """Normalize string accounting for common OCR confusions."""
        s = s.lower()
        s = re.sub(r'[0ou]', 'o', s)      # 0/O/u confusion
        s = re.sub(r'[1il|!]', 'i', s)    # 1/l/I/|/! confusion
        s = re.sub(r'[5s$]', 's', s)      # 5/S/$ confusion
        s = re.sub(r'[8b]', 'b', s)       # 8/B confusion
        s = re.sub(r'[gs]', 's', s)       # g/s confusion (gaming fonts)
        s = re.sub(r'rn', 'm', s)         # rn looks like m
        s = re.sub(r'vv', 'w', s)         # vv looks like w
        s = re.sub(r'cl', 'd', s)         # cl looks like d
        return s
    
    matched_players = []
    for name in all_names:
        # Check if it's in the database (exact or fuzzy)
        name_lower = name.lower().strip()
        name_normalized = ocr_normalize(name_lower)
        
        # Skip very short names (likely noise)
        if len(name_lower) < 3:
            continue
        
        best_match = None
        best_score = 0.0
        
        for db_name in db_names:
            db_lower = db_name.lower()
            db_normalized = ocr_normalize(db_lower)
            
            # Exact match (case insensitive) - highest priority
            if name_lower == db_lower:
                best_match = db_name
                best_score = 1.0
                break
            
            # Exact match after OCR normalization - also high priority
            if name_normalized == db_normalized:
                best_match = db_name
                best_score = 0.99
                break
            
            # Calculate similarity scores
            score = calculate_similarity(name_lower, db_lower)
            score_norm = calculate_similarity(name_normalized, db_normalized)
            current_score = max(score, score_norm)
            
            # Bonus for similar length (to avoid short matches in long names)
            len_diff = abs(len(name_lower) - len(db_lower))
            if len_diff <= 2:
                current_score += 0.1
            elif len_diff <= 4:
                current_score += 0.05
            
            if current_score > best_score:
                best_score = current_score
                best_match = db_name
        
        # Only accept match if score is high enough (0.75 for better accuracy)
        if best_match and best_score >= 0.75:
            matched_players.append(best_match)
            print(f"  MATCH: '{name}' -> '{best_match}' (score={best_score:.2f})")
    
    # Deduplicate
    unique_players = list(set(matched_players))
    
    print(f"\nRaw names collected from killfeed: {len(all_names)}")
    for name in sorted(all_names)[:20]:  # Show first 20
        print(f"  RAW: '{name}'")
    if len(all_names) > 20:
        print(f"  ... and {len(all_names) - 20} more")
    
    print(f"\nMatched {len(unique_players)} to database:")
    for name in unique_players:
        print(f"  - {name}")
    print("="*50 + "\n")
    
    return unique_players


def auto_detect_match_players(cap: cv2.VideoCapture, num_samples: int = 5) -> List[str]:
    """
    Auto-detect player names by scanning the killfeed.
    Falls back to player card reading if killfeed doesn't yield enough names.
    
    Args:
        cap: OpenCV VideoCapture object (will seek to start)
        num_samples: Number of frames to sample (for player card fallback)
        
    Returns:
        List of unique player names detected
    """
    # Primary method: scan killfeed for player names
    # Scan up to 120 seconds or entire video to find all player names
    players = collect_players_from_killfeed_scan(cap, duration_sec=120.0)
    
    if len(players) >= 6:  # Got enough players from killfeed
        return players
    
    # Fallback: try reading player cards
    print("Killfeed scan didn't find enough players, trying player cards...")
    
    text_reader = KillfeedTextReader()
    original_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    
    # Seek to a few seconds in
    start_frame = int(fps * 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    all_names = []
    frame_interval = int(fps)
    
    for i in range(num_samples):
        ok, frame = cap.read()
        if not ok:
            break
            
        print(f"\nSampling frame {i+1}/{num_samples}...")
        names = extract_match_players_from_frame(frame, text_reader)
        all_names.extend(names)
        
        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame + frame_interval)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, original_pos)
    
    # Combine and deduplicate
    all_players = list(set(players + all_names))
    
    # Normalize case
    seen = {}
    for name in all_players:
        name_lower = name.lower()
        if name_lower not in seen:
            seen[name_lower] = name
    
    unique_names = list(seen.values())
    
    print(f"\n" + "="*50)
    print(f"Detected {len(unique_names)} unique players:")
    for name in unique_names:
        print(f"  - {name}")
    print("="*50 + "\n")
    
    return unique_names


def main():
    import sys
    
    # Get video path from command line or use default
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = "test.mp4"
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {video_path} ({w}x{h}) fps={fps:.2f}")
    
    # Auto-detect match players from HUD if not already set via environment
    db = get_player_database()
    if not os.environ.get('MATCH_PLAYERS'):
        detected_players = auto_detect_match_players(cap, num_samples=5)
        if detected_players:
            db.set_match_player_filter(detected_players)
            print(f"Auto-set match filter with {len(detected_players)} detected players")
        else:
            print("Warning: Could not auto-detect players. Using full database.")
    
    # Reset video to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Build detectors
    detectors = build_detectors()
    print(f"Initialized {len(detectors)} detectors:")
    for det in detectors:
        print(f"  - {det.name}")

    # Initialize round tracker
    round_tracker = RoundTracker()
    print("Round tracker initialized")

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
                    # Process through round tracker to group by round
                    round_ev = round_tracker.process_event(ev)
                    if round_ev:
                        events.append(round_ev)
                        event_counts[round_ev.type] = event_counts.get(round_ev.type, 0) + 1
                    
                    events.append(ev)
                    event_counts[ev.type] = event_counts.get(ev.type, 0) + 1

        # Debug overlay: draw ROIs
        if writer is not None:
            dbg = draw_debug_overlay(frame, ROI_NORM, w, h)
            writer.write(dbg)

        frame_idx += 1
        if frame_idx % int(max(1, fps * 2)) == 0:
            elapsed = time.time() - t_start
            current_round = round_tracker.round_number or 0
            print(f"Processed {frame_idx} frames in {elapsed:.1f}s "
                  f"(x{(frame_idx/fps)/elapsed:.2f} realtime) | "
                  f"Events: {len(events)} | Round: {current_round}")

    cap.release()
    if writer is not None:
        writer.release()

    # Save raw events
    raw_events = [
        {"t_ms": e.t_ms, "type": e.type, "roi": e.roi, "payload": e.payload}
        for e in events
    ]
    with open(EVENTS_PATH, "w", encoding="utf-8") as f:
        json.dump(raw_events, f, indent=2)

    # Generate and save round timeline
    timeline = round_tracker.get_timeline()
    timeline_path = EVENTS_PATH.replace(".json", "_timeline.json")
    with open(timeline_path, "w", encoding="utf-8") as f:
        json.dump({
            "video": video_path,
            "resolution": f"{w}x{h}",
            "fps": fps,
            "total_rounds": len(timeline),
            "rounds": timeline
        }, f, indent=2)

    # Generate a simplified kill-summary: round, timestamp, killer -> victim
    kill_summary = []
    for ev in events:
        if ev.type == "KILL_EVENT":
            ks = {
                "t_ms": ev.t_ms,
                "round": ev.payload.get("round_number"),
                "killer": ev.payload.get("killer_name", "Unknown"),
                "victim": ev.payload.get("victim_name", "Unknown"),
                "killer_team": ev.payload.get("killer_team"),
                "victim_team": ev.payload.get("victim_team"),
                "confidence": ev.payload.get("confidence")
            }
            kill_summary.append(ks)

    with open(KILL_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "video": video_path,
            "kill_count": len(kill_summary),
            "kills": kill_summary
        }, f, indent=2)

    # Print summary
    print(f"\n{'='*50}")
    print(f"Processing complete!")
    print(f"{'='*50}")
    print(f"Total events: {len(events)}")
    print(f"Rounds detected: {len(timeline)}")
    print(f"\nEvents by type:")
    for event_type, count in sorted(event_counts.items(), key=lambda x: -x[1]):
        print(f"  {event_type}: {count}")
    
    # Print round summary
    if timeline:
        print(f"\nRound Summary:")
        for rnd in timeline:
            duration = rnd.get("duration_ms")
            duration_str = f"{duration/1000:.1f}s" if duration else "?"
            winner = rnd.get("winner", "?")
            spike = "💣" if rnd.get("spike_planted") else ""
            print(f"  R{rnd['round_number']}: {duration_str} - Winner: {winner} {spike}")
    
    print(f"\nWrote events to {EVENTS_PATH}")
    print(f"Wrote timeline to {timeline_path}")
    if WRITE_DEBUG_VIDEO:
        print(f"Wrote debug video to {DEBUG_VIDEO_PATH}")


if __name__ == "__main__":
    main()
