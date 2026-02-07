"""
Player Name Extractor - Extracts player names from player card ROIs.

Parses player names from the left/right player card regions at the start
of a game, then uses these 10 names for fuzzy matching killfeed OCR results.
"""

import cv2
import numpy as np
from typing import List, Optional, Dict, Tuple, Set
from dataclasses import dataclass
from difflib import SequenceMatcher
import re
from itertools import product

from config import (
    ROI_CONFIG,
    PLAYER_CARD_SUBREGIONS,
)


# Common OCR character confusions (bidirectional)
OCR_CONFUSIONS = {
    '0': ['o', 'O', 'Q', 'D'],
    'o': ['0', 'O', 'Q', 'a'],
    'O': ['0', 'o', 'Q', 'D'],
    '1': ['l', 'I', 'i', '|', 'j'],
    'l': ['1', 'I', 'i', '|', 'j'],
    'I': ['1', 'l', 'i', '|'],
    'i': ['1', 'l', 'I', '|', 'j'],
    'j': ['i', '1', 'l'],
    '5': ['s', 'S'],
    's': ['5', 'S', 'z'],
    'S': ['5', 's', 'Z'],
    '8': ['B', 'b'],
    'B': ['8', 'b', 'R'],
    'b': ['8', 'B', '6'],
    'g': ['q', '9', 'y'],
    'q': ['g', '9'],
    '9': ['g', 'q'],
    'n': ['m', 'h', 'r'],  # rn often read as m
    'm': ['n', 'rn', 'nn'],
    'h': ['n', 'b'],
    'r': ['n', 't'],
    't': ['r', 'f', 'l'],
    'c': ['e', 'o', '('],
    'e': ['c', 'o', 'a'],
    'a': ['o', 'e', 'q'],
    'k': ['lc', 'le', 'h', 'x'],
    'u': ['v', 'n', 'a'],
    'v': ['u', 'w', 'y'],
    'w': ['vv', 'v', 'm'],
    'y': ['v', 'g'],
    'z': ['s', '2'],
    'f': ['t', 'r'],
    'd': ['cl', 'o', 'b'],
    'p': ['b', 'q'],
}


@dataclass
class ExtractedPlayer:
    """A player extracted from the HUD."""
    name: str
    team: str  # "left" or "right"  
    slot: int  # 1-5
    confidence: float
    team_color: Optional[str] = None  # "teal" or "orange"


class PlayerNameExtractor:
    """
    Extracts player names from the player card HUD regions.
    
    Strategy:
    1. At game start, extract names from all 10 player card slots
    2. Use OCR on the player_name subregion of each card
    3. Optionally validate against database of known players
    4. Use these 10 names as the fuzzy match candidates for killfeed
    """
    
    def __init__(self, db=None):
        """
        Args:
            db: Optional EsportsDatabase for validation
        """
        self.db = db
        self._ocr_engine = None
        self._ocr_initialized = False
        
        # Extracted players for the current match
        self.left_team_players: List[ExtractedPlayer] = []
        self.right_team_players: List[ExtractedPlayer] = []
        self._all_names: Set[str] = set()
        self._name_to_canonical: Dict[str, str] = {}  # lowercase -> canonical
        
    def _init_ocr(self):
        """Lazily initialize OCR using the shared OCREngine."""
        if self._ocr_initialized:
            return
        
        try:
            from vod_processor.app.services.ocr.ocr_engine import OCREngine
            self._ocr_engine = OCREngine()
            self._ocr_engine._lazy_init()
            print(f"PlayerNameExtractor: Using {self._ocr_engine.backend}")
        except Exception as e:
            print(f"PlayerNameExtractor: OCREngine unavailable ({e})")
            self._ocr_engine = None
        
        self._ocr_initialized = True
    
    def extract_players_from_frame(
        self,
        frame: np.ndarray,
        left_team_name: str = "left",
        right_team_name: str = "right",
    ) -> Tuple[List[ExtractedPlayer], List[ExtractedPlayer]]:
        """
        Extract player names from a single frame.
        
        Args:
            frame: Full frame image (BGR)
            left_team_name: Name for left team (e.g., "FNC")
            right_team_name: Name for right team (e.g., "NRG")
            
        Returns:
            (left_team_players, right_team_players)
        """
        self._init_ocr()
        h, w = frame.shape[:2]
        
        left_players = []
        right_players = []
        
        # Extract left team players (slots 1-5)
        for slot in range(1, 6):
            roi_key = f"left_player_{slot}"
            if roi_key not in ROI_CONFIG:
                continue
            
            player = self._extract_player_from_card(
                frame, w, h, roi_key, "left", slot, left_team_name
            )
            if player:
                left_players.append(player)
        
        # Extract right team players (slots 1-5)
        for slot in range(1, 6):
            roi_key = f"right_player_{slot}"
            if roi_key not in ROI_CONFIG:
                continue
            
            player = self._extract_player_from_card(
                frame, w, h, roi_key, "right", slot, right_team_name
            )
            if player:
                right_players.append(player)
        
        # Store for fuzzy matching
        self.left_team_players = left_players
        self.right_team_players = right_players
        self._build_name_lookup()
        
        print(f"Extracted {len(left_players)} left players, {len(right_players)} right players")
        for p in left_players:
            print(f"  Left {p.slot}: {p.name} (conf: {p.confidence:.2f})")
        for p in right_players:
            print(f"  Right {p.slot}: {p.name} (conf: {p.confidence:.2f})")
        
        return left_players, right_players
    
    def _extract_player_from_card(
        self,
        frame: np.ndarray,
        frame_w: int,
        frame_h: int,
        roi_key: str,
        team: str,
        slot: int,
        team_name: str,
    ) -> Optional[ExtractedPlayer]:
        """Extract player name from a single player card."""
        roi = ROI_CONFIG[roi_key]
        
        # Get player card region
        x = int(roi[0] * frame_w)
        y = int(roi[1] * frame_h)
        w = int(roi[2] * frame_w)
        h = int(roi[3] * frame_h)
        
        card = frame[y:y+h, x:x+w]
        if card.size == 0:
            return None
        
        # Get player name subregion within the card
        name_roi = PLAYER_CARD_SUBREGIONS["player_name"]
        nx = int(name_roi[0] * w)
        ny = int(name_roi[1] * h)
        nw = int(name_roi[2] * w)
        nh = int(name_roi[3] * h)
        
        name_region = card[ny:ny+nh, nx:nx+nw]
        if name_region.size == 0:
            return None
        
        # OCR the name region
        name, confidence = self._ocr_name(name_region)
        if not name:
            return None
        
        # Clean up the name
        name = self._clean_player_name(name, team_name)
        
        return ExtractedPlayer(
            name=name,
            team=team,
            slot=slot,
            confidence=confidence,
        )
    
    def _ocr_name(self, img: np.ndarray) -> Tuple[Optional[str], float]:
        """OCR a name region using OCREngine."""
        if not self._ocr_engine:
            return None, 0.0
        
        try:
            # Preprocess: scale up for better OCR
            scale = 3
            scaled = cv2.resize(img, None, fx=scale, fy=scale, 
                               interpolation=cv2.INTER_CUBIC)
            
            # Use OCREngine's read_text method
            results = self._ocr_engine.read_text(scaled, min_confidence=0.3)
            
            if not results:
                return None, 0.0
            
            # Get the best result
            best_text = ""
            best_conf = 0.0
            
            for r in results:
                if r.confidence > best_conf and len(r.text.strip()) >= 2:
                    best_text = r.text.strip()
                    best_conf = r.confidence
            
            return best_text if best_conf > 0.3 else None, best_conf
            
        except Exception as e:
            print(f"OCR error: {e}")
            return None, 0.0
    
    def _clean_player_name(self, name: str, team_prefix: str = "") -> str:
        """Clean up OCR'd player name."""
        # Remove common OCR artifacts
        name = name.strip()
        name = re.sub(r'[^\w\s\-_]', '', name)  # Keep alphanumeric, space, dash, underscore
        
        # If team prefix is in the name, it's likely correct
        # e.g., "FNC Chronicle" or "NRG Ethan"
        
        return name
    
    def _build_name_lookup(self):
        """Build lookup tables for fuzzy matching."""
        self._all_names.clear()
        self._name_to_canonical.clear()
        
        for p in self.left_team_players + self.right_team_players:
            canonical = p.name
            self._all_names.add(canonical)
            self._name_to_canonical[canonical.lower()] = canonical
            
            # Also add without team prefix
            parts = canonical.split()
            if len(parts) > 1:
                # "FNC Chronicle" -> also add "Chronicle"
                short_name = parts[-1]
                self._all_names.add(short_name)
                self._name_to_canonical[short_name.lower()] = canonical
    
    def get_all_player_names(self) -> List[str]:
        """Get all extracted player names."""
        return list(self._all_names)
    
    def fuzzy_match(self, ocr_text: str, threshold: float = 0.55) -> Optional[str]:
        """
        Fuzzy match OCR text against extracted player names.
        Returns canonical name if match found.
        
        Uses multiple strategies:
        1. Exact match (case-insensitive)
        2. OCR-aware matching considering common character confusions
        3. Fuzzy similarity with bonus for matching prefixes
        
        Args:
            ocr_text: The OCR'd text from killfeed
            threshold: Minimum similarity (0-1)
            
        Returns:
            Matched canonical name or None
        """
        if not ocr_text or len(ocr_text) < 2:
            return None
        
        if not self._all_names:
            return None
        
        ocr_clean = ocr_text.strip().lower()
        
        # Strategy 1: Exact match (case-insensitive)
        if ocr_clean in self._name_to_canonical:
            return self._name_to_canonical[ocr_clean]
        
        best_score = 0.0
        best_match = None
        
        for name in self._all_names:
            name_lower = name.lower()
            
            # Strategy 2: OCR-aware exact match
            if self._ocr_equivalent(ocr_clean, name_lower):
                return self._name_to_canonical.get(name_lower, name)
            
            # Strategy 3: Substring containment
            if len(name_lower) >= 3 and len(ocr_clean) >= 3:
                if name_lower in ocr_clean or ocr_clean in name_lower:
                    return self._name_to_canonical.get(name_lower, name)
            
            # Strategy 4: Check if just the player name part matches (without team tag)
            name_parts = name_lower.split()
            ocr_parts = ocr_clean.split()
            if len(name_parts) >= 2 and len(ocr_parts) >= 2:
                # Compare just the player name (second part)
                name_only = name_parts[-1]
                ocr_name_only = ocr_parts[-1]
                if self._ocr_similarity(ocr_name_only, name_only) > 0.75:
                    # Check if team prefix roughly matches
                    if self._ocr_equivalent(ocr_parts[0][:3], name_parts[0][:3]):
                        return self._name_to_canonical.get(name_lower, name)
            
            # Strategy 5: Fuzzy similarity with OCR awareness
            score = self._ocr_similarity(ocr_clean, name_lower)
            
            # Bonus for matching first characters
            if len(ocr_clean) >= 2 and len(name_lower) >= 2:
                if ocr_clean[:2] == name_lower[:2]:
                    score += 0.12
                elif self._ocr_equivalent(ocr_clean[:2], name_lower[:2]):
                    score += 0.08
            
            # Bonus for matching team prefix
            if len(ocr_clean) >= 3 and len(name_lower) >= 3:
                if ocr_clean[:3] == name_lower[:3]:  # e.g., "fnc" or "nrg"
                    score += 0.10
                elif self._ocr_equivalent(ocr_clean[:3], name_lower[:3]):
                    score += 0.06
            
            if score > best_score:
                best_score = score
                best_match = name
        
        if best_score >= threshold and best_match:
            return self._name_to_canonical.get(best_match.lower(), best_match)
        
        return None
    
    def _ocr_equivalent(self, s1: str, s2: str) -> bool:
        """
        Check if two strings are equivalent considering OCR confusions.
        e.g., "s0m" == "som", "Ethan" == "Fthan"
        """
        if len(s1) != len(s2):
            return False
        if s1 == s2:
            return True
        
        for c1, c2 in zip(s1, s2):
            if c1 == c2:
                continue
            # Check if characters are commonly confused
            if c1 in OCR_CONFUSIONS and c2 in OCR_CONFUSIONS.get(c1, []):
                continue
            if c2 in OCR_CONFUSIONS and c1 in OCR_CONFUSIONS.get(c2, []):
                continue
            return False
        return True
    
    def _ocr_similarity(self, ocr_text: str, target: str) -> float:
        """
        Calculate similarity considering OCR confusions.
        """
        if not ocr_text or not target:
            return 0.0
        if ocr_text == target:
            return 1.0
        
        # Base similarity
        base_score = SequenceMatcher(None, ocr_text, target).ratio()
        
        # Try with OCR corrections applied to ocr_text
        corrected = self._apply_common_corrections(ocr_text)
        if corrected != ocr_text:
            corrected_score = SequenceMatcher(None, corrected, target).ratio()
            base_score = max(base_score, corrected_score)
        
        # Calculate character-level match with OCR tolerance
        matches = 0
        min_len = min(len(ocr_text), len(target))
        for i in range(min_len):
            c1, c2 = ocr_text[i], target[i]
            if c1 == c2:
                matches += 1
            elif c1 in OCR_CONFUSIONS and c2 in OCR_CONFUSIONS.get(c1, []):
                matches += 0.9  # High score for known confusions
            elif c2 in OCR_CONFUSIONS and c1 in OCR_CONFUSIONS.get(c2, []):
                matches += 0.9
        
        max_len = max(len(ocr_text), len(target))
        ocr_score = matches / max_len if max_len > 0 else 0.0
        
        return max(base_score, ocr_score)
    
    def _apply_common_corrections(self, text: str) -> str:
        """Apply common OCR corrections."""
        result = text.lower()
        
        # Fix common multi-character errors
        corrections = [
            ('rn', 'm'),  # rn → m
            ('cl', 'd'),  # cl → d
            ('vv', 'w'),  # vv → w
            ('ii', 'n'),  # ii → n
        ]
        for old, new in corrections:
            result = result.replace(old, new)
        
        return result
    
    def _similarity(self, s1: str, s2: str) -> float:
        """Calculate string similarity."""
        return SequenceMatcher(None, s1, s2).ratio() if s1 and s2 else 0.0
    
    def _team_prefix_matches(self, ocr_prefix: str, expected_code: str) -> bool:
        """Check if an OCR-read team prefix matches the expected team code.
        
        Handles common OCR errors like:
        - 'nag', 'npg', 'wrg', 'nrc', 'nri' -> 'nrg'
        - 'fnc', 'fhc', 'enc' -> 'fnc'
        - Letters that look similar: a↔r, p↔r, w↔n, h↔n, e↔f
        """
        ocr_prefix = ocr_prefix.lower().strip()
        expected_code = expected_code.lower().strip()
        
        # Exact match
        if ocr_prefix == expected_code:
            return True
        
        # Substring match (one contains the other)
        if expected_code in ocr_prefix or ocr_prefix in expected_code:
            return True
        
        # Same length check for character-by-character similarity
        if len(ocr_prefix) == len(expected_code):
            # Count matching characters
            matches = sum(1 for a, b in zip(ocr_prefix, expected_code) if a == b)
            if matches >= len(expected_code) - 1:  # Allow 1 character difference
                return True
            
            # Check OCR-confusable characters
            ocr_confusable = {
                'a': ['r', 'o', 'd'],
                'r': ['a', 'n', 'i'],
                'n': ['r', 'h', 'm', 'w'],
                'g': ['9', 'q', 'c'],
                'p': ['r', 'b', 'd'],
                'w': ['n', 'vv', 'm'],
                'h': ['n', 'b'],
                'e': ['f', 'c'],
                'f': ['e', 't'],
                'c': ['e', 'o', 'g'],
                'o': ['0', 'a', 'c'],
                '0': ['o', 'O'],
                'i': ['1', 'l', 'j'],
                '1': ['i', 'l'],
                'l': ['1', 'i'],
            }
            
            # Check if differences are due to OCR-confusable characters
            ocr_match = True
            for ocr_char, expected_char in zip(ocr_prefix, expected_code):
                if ocr_char == expected_char:
                    continue
                # Check if ocr_char could be misread as expected_char
                if expected_char in ocr_confusable.get(ocr_char, []):
                    continue
                # Check reverse (expected_char could be misread as ocr_char)
                if ocr_char in ocr_confusable.get(expected_char, []):
                    continue
                ocr_match = False
                break
            
            if ocr_match:
                return True
        
        return False
    
    def _is_garbage_name(self, name: str) -> bool:
        """Check if a name is obviously garbage from OCR hallucination."""
        if not name or len(name) < 3:
            return True
        
        name_lower = name.lower()
        
        # Known garbage patterns from Surya hallucinations
        garbage_patterns = [
            'the state of', 'the second', 'the same of',
            'the party of', 'the property of', 'the person',
            'column', 'contractor', 'property', 'security',
            'reserve', 'residence', 'control of',
            'name of persons', 'math>', '<b>', '</b>',
            '----', '____', '. . .', '* * *',
            'william', 'college', 'secret', 'alcohol',
        ]
        for pattern in garbage_patterns:
            if pattern in name_lower:
                return True
        
        # Too long to be a player name
        if len(name) > 25:
            return True
        
        # Too many repeated words
        words = name_lower.split()
        if len(words) >= 4:
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            if word_counts and max(word_counts.values()) > len(words) * 0.4:
                return True
        
        # Mostly non-alpha characters
        alpha_count = sum(1 for c in name if c.isalpha())
        if alpha_count < len(name) * 0.4:
            return True
        
        return False
    
    def get_player_team(self, player_name: str, left_team_code: str = None, right_team_code: str = None) -> Optional[str]:
        """Get the team (left/right) for a player name.
        
        If left_team_code and right_team_code are provided (e.g., "NRG", "FNC"),
        we can use the team prefix in the kill feed to determine team.
        """
        # Early garbage filter - reject obvious hallucinations
        if self._is_garbage_name(player_name):
            return None
        
        name_lower = player_name.lower().strip()
        
        # Also extract just the player name part if it has a team prefix
        # e.g., "NRG s0m" -> ["nrg", "s0m"]
        name_parts = name_lower.split()
        name_only = name_parts[-1] if name_parts else name_lower  # Last part is player name
        
        # First check: if the name has a team prefix, use it directly
        # Note: Only log if codes are missing or prefix doesn't match
        if len(name_parts) >= 2:
            team_prefix = name_parts[0].lower()
            
            if left_team_code and right_team_code:
                left_code = left_team_code.lower()
                right_code = right_team_code.lower()
                
                # Check if prefix matches left team (with OCR error tolerance)
                if self._team_prefix_matches(team_prefix, left_code):
                    print(f"[DEBUG get_player_team] '{player_name}' -> left (prefix '{team_prefix}' matches '{left_code}')")
                    return "left"
                
                # Check if prefix matches right team (with OCR error tolerance)
                if self._team_prefix_matches(team_prefix, right_code):
                    print(f"[DEBUG get_player_team] '{player_name}' -> right (prefix '{team_prefix}' matches '{right_code}')")
                    return "right"
            else:
                print(f"[DEBUG get_player_team] '{player_name}' has prefix '{team_prefix}' but team codes not provided (left={left_team_code}, right={right_team_code})")
        
        print(f"[DEBUG get_player_team] input='{player_name}' name_only='{name_only}' left_players={[p.name for p in self.left_team_players]} right_players={[p.name for p in self.right_team_players]}")
        
        for p in self.left_team_players:
            p_lower = p.name.lower()
            # Check various matching strategies
            if p_lower == name_lower:  # Exact match
                print(f"[DEBUG] MATCH left (exact full): {p_lower}")
                return "left"
            if p_lower == name_only:  # Just player name matches
                print(f"[DEBUG] MATCH left (name_only): {p_lower}")
                return "left"
            if name_lower in p_lower or p_lower in name_lower:  # Substring
                print(f"[DEBUG] MATCH left (substring): {p_lower}")
                return "left"
            if name_only in p_lower or p_lower in name_only:  # Player name substring
                print(f"[DEBUG] MATCH left (name_only substring): {p_lower}")
                return "left"
            # OCR similarity check (0/O/o, l/1/I, etc.) - lower threshold for short names
            similarity = self._ocr_similarity(p_lower, name_only)
            if similarity > 0.6 or (len(name_only) <= 4 and similarity > 0.5):
                print(f"[DEBUG] MATCH left (OCR similarity {similarity:.2f}): {p_lower}")
                return "left"
        
        for p in self.right_team_players:
            p_lower = p.name.lower()
            # Check various matching strategies
            if p_lower == name_lower:  # Exact match
                print(f"[DEBUG] MATCH right (exact full): {p_lower}")
                return "right"
            if p_lower == name_only:  # Just player name matches
                print(f"[DEBUG] MATCH right (name_only): {p_lower}")
                return "right"
            if name_lower in p_lower or p_lower in name_lower:  # Substring
                print(f"[DEBUG] MATCH right (substring): {p_lower}")
                return "right"
            if name_only in p_lower or p_lower in name_only:  # Player name substring
                print(f"[DEBUG] MATCH right (name_only substring): {p_lower}")
                return "right"
            # OCR similarity check - lower threshold for short names
            similarity = self._ocr_similarity(p_lower, name_only)
            if similarity > 0.6 or (len(name_only) <= 4 and similarity > 0.5):
                print(f"[DEBUG] MATCH right (OCR similarity {similarity:.2f}): {p_lower}")
                return "right"
        
        print(f"[DEBUG] NO MATCH for '{player_name}'")
        return None
    
    def _ocr_similarity(self, s1: str, s2: str) -> float:
        """Calculate OCR-aware similarity between two strings."""
        if not s1 or not s2:
            return 0.0
        # Normalize for common OCR confusions
        s1_norm = self._ocr_normalize(s1)
        s2_norm = self._ocr_normalize(s2)
        if s1_norm == s2_norm:
            return 1.0
        return SequenceMatcher(None, s1_norm, s2_norm).ratio()
    
    def _ocr_normalize(self, s: str) -> str:
        """Normalize string for OCR comparison."""
        s = s.lower()
        # 0/O/o/u confusion - OCR often reads 0 as u or o
        s = re.sub(r'[0ouv]', 'o', s)
        # 1/l/I/| confusion
        s = re.sub(r'[1il|]', 'l', s)
        # 5/s/S confusion
        s = re.sub(r'[5s]', 's', s)
        return s
    
    def set_players_manually(
        self,
        left_names: List[str],
        right_names: List[str],
    ):
        """
        Manually set player names (e.g., from database or user input).
        Useful when OCR extraction fails.
        """
        self.left_team_players = [
            ExtractedPlayer(name=n, team="left", slot=i+1, confidence=1.0)
            for i, n in enumerate(left_names[:5])
        ]
        self.right_team_players = [
            ExtractedPlayer(name=n, team="right", slot=i+1, confidence=1.0)
            for i, n in enumerate(right_names[:5])
        ]
        self._build_name_lookup()
        print(f"Manually set {len(self.left_team_players)} left, {len(self.right_team_players)} right players")


# Database-backed player matching with smart filtering
class DatabasePlayerMatcher:
    """
    Matches OCR results against database with smart filtering.
    
    Since there are 27,000+ players in the database, we:
    1. First try to match against extracted HUD names (10 players max)
    2. Fall back to database only if no HUD match
    3. Use team tags to narrow down search
    """
    
    def __init__(self, db=None):
        self.db = db
        self.hud_extractor = PlayerNameExtractor(db)
        self._team_tag_cache: Dict[str, List[str]] = {}  # tag -> nicknames
        
    def initialize_from_frame(self, frame: np.ndarray):
        """Initialize player names from a game frame."""
        self.hud_extractor.extract_players_from_frame(frame)
    
    def set_match_players(self, left_names: List[str], right_names: List[str]):
        """Manually set match players."""
        self.hud_extractor.set_players_manually(left_names, right_names)
        
        # Also set filter in database if available
        if self.db:
            all_names = left_names + right_names
            self.db.set_match_player_filter(all_names)
    
    def match_player(self, ocr_text: str) -> Optional[str]:
        """
        Match OCR text to a player name.
        
        Priority:
        1. HUD-extracted names (10 match players)
        2. Database fuzzy match (filtered by match players if set)
        """
        # Sanity checks - reject obviously invalid names
        if not ocr_text or len(ocr_text) < 2:
            return None
        
        # Player names are typically max 16 characters
        if len(ocr_text) > 20:
            return None
        
        # Player names shouldn't contain colons or semicolons (format separators)
        if ':' in ocr_text or ';' in ocr_text:
            return None
        
        # First try HUD names
        match = self.hud_extractor.fuzzy_match(ocr_text)
        if match:
            return match
        
        # Fall back to database
        if self.db:
            player = self.db.fuzzy_match_player(ocr_text)
            if player:
                return player.nickname
        
        return None
    
    # Alias for compatibility
    def find_match(self, ocr_text: str) -> Optional[str]:
        """Alias for match_player."""
        return self.match_player(ocr_text)
    
    def set_player_names(self, names: List[str]):
        """
        Simple interface to set player names.
        Splits list roughly in half for left/right teams.
        """
        mid = len(names) // 2
        left = names[:mid] if mid > 0 else []
        right = names[mid:] if mid > 0 else names
        self.set_match_players(left, right)
    
    def get_player_team(self, player_name: str, left_team_code: str = None, right_team_code: str = None) -> Optional[str]:
        """Get team side (left/right) for a player name."""
        return self.hud_extractor.get_player_team(player_name, left_team_code, right_team_code)
    
    def get_player_team_color(self, player_name: str, left_team_code: str = None, right_team_code: str = None) -> Optional[str]:
        """Get team color for a player (teal or orange)."""
        team = self.hud_extractor.get_player_team(player_name, left_team_code, right_team_code)
        if team == "left":
            return "teal"  # Left team is teal
        elif team == "right":
            return "orange"  # Right team is orange
        return None
