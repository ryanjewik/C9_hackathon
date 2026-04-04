"""
VALORANT VOD Timeline Processor - Configuration
"""

from typing import Dict, Tuple, List
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables.
    
    All values are loaded from environment variables or .env file.
    Defaults are provided for non-sensitive settings only.
    """

    # Database - loaded from environment/docker-compose
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "cloud9"
    postgres_user: str = "postgres"
    postgres_password: str  # Required - no default, must be set in .env

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # File Storage
    upload_dir: str = "/app/uploads"
    output_dir: str = "/app/outputs"
    max_upload_size_mb: int = 5000

    # Processing
    worker_concurrency: int = 2
    frame_sample_fps: float = 15.0  # Increased for better killfeed OCR coverage

    # Speed optimization
    fast_mode: bool = False
    killfeed_fps: float = 15.0  # Increased for better detection of fast multi-kills
    top_hud_fps: float = 2.0

    # Optional player filter
    match_players: str = ""

    class Config:
        env_file = "../.env"  # .env is in vod_processor/, config is in vod_processor/config/
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()


# ==============================================
# ROI Configuration (normalized coordinates)
# ==============================================

# Individual player card dimensions
PLAYER_CARD_WIDTH = 0.175
PLAYER_CARD_HEIGHT = 0.09

# X positions for player cards
LEFT_TEAM_X = 0.005
RIGHT_TEAM_X = 0.820

# Y positions for each player slot
LEFT_PLAYER_Y = [0.505, 0.605, 0.705, 0.805, 0.905]
RIGHT_PLAYER_Y = [0.505, 0.605, 0.705, 0.805, 0.905]

ROI_CONFIG: Dict[str, Tuple[float, float, float, float]] = {
    # Map indicator - series scoreboard in top-left showing "CURRENT: <MAP>"
    # Shows: "LOTUS 13-6 | CURRENT: ABYSS | NEXT: ASCENT"
    "map_indicator": (0.0, 0.0, 0.32, 0.025),
    
    # Minimap
    "minimap": (0.016, 0.032, 0.250, 0.385),

    # Top HUD
    "top_hud": (0.335, 0.005, 0.330, 0.200),
    "top_left_score": (0.417, 0.009, 0.036, 0.055),
    "top_center_timer": (0.465, 0.010, 0.070, 0.045),
    "top_right_score": (0.555, 0.009, 0.036, 0.055),
    "top_spike_icon": (0.485, 0.065, 0.035, 0.058),
    "top_plant_text": (0.43, 0.127, 0.14, 0.070),
    
    # Team tags on top HUD - these show team abbreviations (e.g., "TH", "PRX")
    # Calibrated to capture just the text, excluding team logo icons
    # Left shifted right by 10px to avoid logo overlap, uses smaller width
    "top_left_team_tag": (0.379, 0.007, 0.038, 0.043),   # Left team tag (e.g., TH)
    "top_right_team_tag": (0.587, 0.007, 0.040, 0.043),  # Right team tag (e.g., PRX)

    # Kill feed - y=0.092 (~99px) aligns ROW 1 with first kill entry
    # h=0.318 (~343px) gives 38px per row to match actual kill entry spacing
    "killfeed": (0.690, 0.092, 0.305, 0.318),

    # Team Comms overlay detection
    # When "TEAM COMMS" text is visible in the upper-right, the game viewport
    # is compressed and killfeed positions shift. Detect the text region.
    "team_comms": (0.815, 0.055, 0.150, 0.060),

    # Bottom HUD
    "bottom_hud": (0.215, 0.870, 0.570, 0.125),

    # Frame state detection
    # replay_indicator covers bottom-right where overlay text appears during replays
    # Detects: REPLAY, CLUTCH, THRIFTY, FLAWLESS
    "replay_indicator": (0.780, 0.850, 0.210, 0.120),
    "score_bar": (0.350, 0.010, 0.300, 0.055),
    "left_panels": (0.000, 0.500, 0.185, 0.500),
    "right_panels": (0.815, 0.500, 0.185, 0.500),

    # Left team player cards
    "left_player_1": (LEFT_TEAM_X, LEFT_PLAYER_Y[0], PLAYER_CARD_WIDTH, PLAYER_CARD_HEIGHT),
    "left_player_2": (LEFT_TEAM_X, LEFT_PLAYER_Y[1], PLAYER_CARD_WIDTH, PLAYER_CARD_HEIGHT),
    "left_player_3": (LEFT_TEAM_X, LEFT_PLAYER_Y[2], PLAYER_CARD_WIDTH, PLAYER_CARD_HEIGHT),
    "left_player_4": (LEFT_TEAM_X, LEFT_PLAYER_Y[3], PLAYER_CARD_WIDTH, PLAYER_CARD_HEIGHT),
    "left_player_5": (LEFT_TEAM_X, LEFT_PLAYER_Y[4], PLAYER_CARD_WIDTH, PLAYER_CARD_HEIGHT),

    # Right team player cards
    "right_player_1": (RIGHT_TEAM_X, RIGHT_PLAYER_Y[0], PLAYER_CARD_WIDTH, PLAYER_CARD_HEIGHT),
    "right_player_2": (RIGHT_TEAM_X, RIGHT_PLAYER_Y[1], PLAYER_CARD_WIDTH, PLAYER_CARD_HEIGHT),
    "right_player_3": (RIGHT_TEAM_X, RIGHT_PLAYER_Y[2], PLAYER_CARD_WIDTH, PLAYER_CARD_HEIGHT),
    "right_player_4": (RIGHT_TEAM_X, RIGHT_PLAYER_Y[3], PLAYER_CARD_WIDTH, PLAYER_CARD_HEIGHT),
    "right_player_5": (RIGHT_TEAM_X, RIGHT_PLAYER_Y[4], PLAYER_CARD_WIDTH, PLAYER_CARD_HEIGHT),
}

# Sub-regions within player cards - FINAL CALIBRATED VALUES
# For LEFT side cards (right side is mirrored)
PLAYER_CARD_SUBREGIONS: Dict[str, Tuple[float, float, float, float]] = {
    "agent_icon":    (0.04, 0.04, 0.14, 0.46),
    "player_name":   (0.19, 0.02, 0.40, 0.36),
    "health_shield": (0.62, 0.02, 0.32, 0.36),
    "ult_charge":    (0.02, 0.46, 0.18, 0.52),
    "abilities":     (0.20, 0.50, 0.34, 0.44),
    "weapon":        (0.56, 0.48, 0.26, 0.44),
    "ability_1":     (0.22, 0.52, 0.09, 0.40),
    "ability_2":     (0.34, 0.52, 0.09, 0.40),
    "ability_3":     (0.46, 0.52, 0.09, 0.40),
    "money":         (0.73, 0.52, 0.20, 0.44),
}

# Bottom HUD sub-regions
BOTTOM_HUD_SUBREGIONS: Dict[str, Tuple[float, float, float, float]] = {
    "health":     (0.02, 0.30, 0.14, 0.38),
    "armor":      (0.16, 0.30, 0.12, 0.38),
    "abilities":  (0.28, 0.28, 0.44, 0.48),
    "ammo":       (0.74, 0.28, 0.24, 0.46),
    "ult_points": (0.44, 0.60, 0.12, 0.30),
}

# Per-detector effective FPS
DETECTOR_FPS: Dict[str, float] = {
    "killfeed": 15.0,  # Increased to 15 FPS for better coverage of fast kills
    "top_hud": 2.0,      # Simplified OCR approach - 2 FPS is sufficient for score
    "bottom_hud": 5.0,
    "minimap": 3.0,
    "player_card": 3.0,
}

# Team color ranges in HSV for killfeed detection
TEAM_COLORS = {
    "teal": {
        "lower": (75, 50, 80),
        "upper": (115, 255, 255)
    },
    "orange": {
        "lower": (0, 80, 100),
        "upper": (25, 255, 255),
        "lower2": (160, 80, 100),
        "upper2": (180, 255, 255)
    },
    "red": {
        "lower1": (0, 120, 140),
        "upper1": (10, 255, 255),
        "lower2": (170, 120, 140),
        "upper2": (179, 255, 255)
    },
    "white": {
        "lower": (0, 0, 200),
        "upper": (180, 30, 255)
    }
}

# Killfeed detection parameters
KILLFEED_ROW_HEIGHT_RANGE = (20, 50)
KILLFEED_ROW_MIN_WIDTH_RATIO = 0.4
KILLFEED_MAX_ROWS = 10

# Killfeed row sub-regions (9 rows for tracking individual kill entries)
# Row 1 = newest kill (top), Row 9 = oldest (bottom)
# Each row is ~36px tall at 1080p (330px / 9 = 36.7px)
# NOTE: These are currently UNUSED - vod_processor calculates rows dynamically
KILLFEED_ROW_ROIS: Dict[str, Tuple[float, float, float, float]] = {
    "killfeed_row_1": (0.6900, 0.0840, 0.3050, 0.0340),
    "killfeed_row_2": (0.6900, 0.1180, 0.3050, 0.0340),
    "killfeed_row_3": (0.6900, 0.1520, 0.3050, 0.0340),
    "killfeed_row_4": (0.6900, 0.1860, 0.3050, 0.0340),
    "killfeed_row_5": (0.6900, 0.2200, 0.3050, 0.0340),
    "killfeed_row_6": (0.6900, 0.2540, 0.3050, 0.0340),
    "killfeed_row_7": (0.6900, 0.2880, 0.3050, 0.0340),
    "killfeed_row_8": (0.6900, 0.3220, 0.3050, 0.0340),
    "killfeed_row_9": (0.6900, 0.3560, 0.3050, 0.0340),
}

# Number of killfeed rows to process
KILLFEED_NUM_ROWS = 5  # Primary rows (top 5) - reduced from 9 for speed
KILLFEED_EXTENDED_ROWS = 9  # Extended rows only checked when primary is full

# Legacy OCR corrections - DEPRECATED
# The primary name matching is now handled by DatabasePlayerMatcher.match_killfeed_name()
# which fuzzy matches against the actual player pool from the database.
# This dictionary is kept as a fallback for extreme edge cases only.
# For new matches, no changes to this dictionary are needed - the DB matcher handles it.
OCR_NAME_CORRECTIONS: Dict[str, str] = {
    # Only keep truly ambiguous corrections that fuzzy matching can't resolve
    # Most variations are now handled by db_player_matcher._ocr_similarity() with threshold 0.55
}

# Deduplication window (ms)
KILL_DEDUP_WINDOW_MS = 6000

# ── TEAM COMMS overlay ROI overrides ──────────────────────────────────
# When "TEAM COMMS" is detected, the broadcast compresses the game viewport
# to roughly the left 56% of the screen. These overrides replace the normal
# ROI positions for affected regions.  Only ROIs that move need entries here.
TEAM_COMMS_ROI_OVERRIDES: Dict[str, Tuple[float, float, float, float]] = {
    # Top HUD
    "map_indicator":      (0.000, 0.000, 0.240, 0.025),
    "minimap":            (0.009, 0.032, 0.170, 0.335),
    "top_hud":            (0.280, 0.005, 0.200, 0.140),
    "top_left_score":     (0.315, 0.009, 0.025, 0.042),
    "top_center_timer":   (0.360, 0.010, 0.045, 0.035),
    "top_right_score":    (0.420, 0.009, 0.025, 0.042),
    "top_spike_icon":     (0.365, 0.055, 0.025, 0.045),
    "top_plant_text":     (0.320, 0.090, 0.110, 0.055),
    "score_bar":          (0.290, 0.010, 0.180, 0.045),
    "top_left_team_tag":  (0.285, 0.007, 0.025, 0.035),
    "top_right_team_tag": (0.445, 0.007, 0.025, 0.035),
    # Killfeed
    "killfeed":           (0.520, 0.065, 0.230, 0.250),
    # Left panels & player cards
    "left_panels":        (0.000, 0.385, 0.130, 0.355),
    "left_player_1":      (0.003, 0.385, 0.130, 0.065),
    "left_player_2":      (0.003, 0.457, 0.130, 0.065),
    "left_player_3":      (0.003, 0.531, 0.130, 0.065),
    "left_player_4":      (0.003, 0.607, 0.130, 0.065),
    "left_player_5":      (0.003, 0.680, 0.130, 0.065),
    # Right panels & player cards
    "right_panels":       (0.620, 0.385, 0.130, 0.355),
    "right_player_1":     (0.620, 0.385, 0.130, 0.065),
    "right_player_2":     (0.620, 0.457, 0.130, 0.065),
    "right_player_3":     (0.620, 0.531, 0.130, 0.065),
    "right_player_4":     (0.620, 0.607, 0.130, 0.065),
    "right_player_5":     (0.620, 0.680, 0.130, 0.065),
    # Bottom
    "bottom_hud":         (0.200, 0.670, 0.320, 0.100),
    "replay_indicator":   (0.560, 0.600, 0.170, 0.120),
}
