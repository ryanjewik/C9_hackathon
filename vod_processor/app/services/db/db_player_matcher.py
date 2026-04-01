"""
Database-backed Player Matcher with Smart Fuzzy Search.

Extracts player names from HUD ROIs and validates against PostgreSQL database.
Uses efficient strategies to avoid scanning 27k+ players:
1. First try team-based filtering if team tags detected
2. Use trigram similarity for fuzzy matching (pg_trgm extension)
3. Cache results to avoid repeated queries
"""

import os
import re
from typing import List, Optional, Dict, Tuple, Set
from dataclasses import dataclass
from difflib import SequenceMatcher
import psycopg2
from psycopg2.extras import RealDictCursor


@dataclass
class MatchedPlayer:
    """A player matched from OCR against the database."""
    nickname: str
    team_id: Optional[int]
    team_tag: Optional[str]
    team_name: Optional[str]
    confidence: float
    source: str  # "exact", "fuzzy", "hud"


class DatabasePlayerMatcher:
    """
    Matches OCR results against database with smart filtering.
    
    Strategy:
    1. Extract raw names from HUD ROIs
    2. For each raw name, fuzzy search database (limited scope)
    3. Cache the 10 match players for fast killfeed matching
    """
    
    def __init__(self, db_config: Optional[Dict] = None):
        """
        Args:
            db_config: Database connection config. If None, uses env vars.
        """
        # Determine host: use host.docker.internal in Docker to reach host's database
        # Fall back to POSTGRES_HOST env var or localhost
        default_host = os.environ.get('POSTGRES_HOST', 'localhost')
        if default_host == 'postgres':
            # We're in Docker, but need to reach the host's database with esports data
            default_host = 'host.docker.internal'
        
        self.db_config = db_config or {
            'host': default_host,
            'port': int(os.environ.get('POSTGRES_PORT', 5432)),
            'user': os.environ.get('POSTGRES_USER', 'postgres'),
            'password': os.environ.get('POSTGRES_PASSWORD', ''),
            'database': os.environ.get('POSTGRES_DB', 'cloud9'),
        }
        
        self._conn = None
        self._match_players: Dict[str, MatchedPlayer] = {}  # lowercase -> MatchedPlayer
        self._left_team_players: Dict[str, MatchedPlayer] = {}  # Players for left team only
        self._right_team_players: Dict[str, MatchedPlayer] = {}  # Players for right team only
        self._left_players: List[str] = []
        self._right_players: List[str] = []
        self._left_team_code: Optional[str] = None
        self._right_team_code: Optional[str] = None
        
        # OCR confusion mappings for fuzzy matching
        # Maps characters that OCR frequently confuses to a canonical form
        self._ocr_confusions = {
            # Number/letter confusions
            '0': 'o', 'o': 'o', 'O': 'o',
            '1': 'l', 'l': 'l', 'I': 'l', 'i': 'l',
            '4': 'a', 'a': 'a', 'A': 'a',  # 4 often confused with A
            '5': 's', 's': 's', 'S': 's',
            '8': 'b', 'B': 'b',
            '6': 'g', 'G': 'g',
            '3': 'e', 'E': 'e',  # 3 can look like E
            '7': 't', 'T': 't',  # 7 can look like T
            '9': 'g',  # 9 can look like g
            # Common letter confusions
            'n': 'n', 'r': 'n',  # r and n look similar
            'c': 'c',  # removed e->c as it was too aggressive
            'u': 'u', 'v': 'u',  # u and v 
            'm': 'm', 'w': 'm',  # wide letters
        }
    
    def _get_connection(self):
        """Get or create database connection."""
        if self._conn is None or self._conn.closed:
            try:
                self._conn = psycopg2.connect(
                    host=self.db_config['host'],
                    port=self.db_config['port'],
                    user=self.db_config['user'],
                    password=self.db_config['password'],
                    dbname=self.db_config['database'],
                )
                print(f"[DBPlayerMatcher] Connected to database {self.db_config['database']}")
            except Exception as e:
                print(f"[DBPlayerMatcher] Database connection failed: {e}")
                return None
        return self._conn
    
    def _normalize_for_search(self, text: str) -> str:
        """Normalize text for fuzzy search (handle OCR confusions)."""
        result = text.lower().strip()
        # Normalize common OCR confusions
        for char, replacement in self._ocr_confusions.items():
            result = result.replace(char, replacement)
        return result
    
    def find_players_by_team(self, team_tag: str) -> List[Dict]:
        """
        Find ALL players who have ever been on a team by tag (e.g., 'NRG', 'FNC').
        Uses esports_rosters table to get historical roster entries and collects
        all unique player nicknames from p1-p5 positions.
        
        Returns list of {nickname, team_tag, team_name, team_id}.
        """
        conn = self._get_connection()
        if not conn:
            return []
        
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get all roster entries for this team and collect unique players from p1-p5
                cur.execute("""
                    SELECT DISTINCT ON (nickname) nickname, team_tag, team_name, team_id
                    FROM (
                        SELECT p1.nickname, t.team_tag, t.name as team_name, t.id as team_id
                        FROM esports_rosters r
                        JOIN esports_teams t ON r.team_id = t.id
                        JOIN esports_players p1 ON r.player_1 = p1.id
                        WHERE UPPER(t.team_tag) = UPPER(%s)
                        
                        UNION
                        
                        SELECT p2.nickname, t.team_tag, t.name as team_name, t.id as team_id
                        FROM esports_rosters r
                        JOIN esports_teams t ON r.team_id = t.id
                        JOIN esports_players p2 ON r.player_2 = p2.id
                        WHERE UPPER(t.team_tag) = UPPER(%s)
                        
                        UNION
                        
                        SELECT p3.nickname, t.team_tag, t.name as team_name, t.id as team_id
                        FROM esports_rosters r
                        JOIN esports_teams t ON r.team_id = t.id
                        JOIN esports_players p3 ON r.player_3 = p3.id
                        WHERE UPPER(t.team_tag) = UPPER(%s)
                        
                        UNION
                        
                        SELECT p4.nickname, t.team_tag, t.name as team_name, t.id as team_id
                        FROM esports_rosters r
                        JOIN esports_teams t ON r.team_id = t.id
                        JOIN esports_players p4 ON r.player_4 = p4.id
                        WHERE UPPER(t.team_tag) = UPPER(%s)
                        
                        UNION
                        
                        SELECT p5.nickname, t.team_tag, t.name as team_name, t.id as team_id
                        FROM esports_rosters r
                        JOIN esports_teams t ON r.team_id = t.id
                        JOIN esports_players p5 ON r.player_5 = p5.id
                        WHERE UPPER(t.team_tag) = UPPER(%s)
                    ) AS all_players
                    ORDER BY nickname
                """, (team_tag, team_tag, team_tag, team_tag, team_tag))
                return cur.fetchall()
        except Exception as e:
            try:
                conn.rollback()
            except Exception:
                pass
            print(f"[DBPlayerMatcher] Query error: {e}")
            return []
    
    def find_roster_by_date(self, team_tag: str, match_date: str = None) -> List[Dict]:
        """
        Find the 5-player roster for a team that was active at a specific date.
        Uses the esports_rosters table which has dated rosters.
        
        Args:
            team_tag: Team tag (e.g., 'NRG', 'FNC')
            match_date: Date string in format 'YYYY-MM-DD'. If None, uses most recent roster.
            
        Returns:
            List of 5 player dicts {nickname, team_tag, team_name, team_id}
        """
        conn = self._get_connection()
        if not conn:
            return []
        
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get the roster that was active on or before the match date
                # If no match_date, get the most recent roster
                if match_date:
                    cur.execute("""
                        SELECT r.id as roster_id, r.date_created, t.team_tag, t.name as team_name, t.id as team_id,
                               p1.nickname as p1, p2.nickname as p2, p3.nickname as p3, 
                               p4.nickname as p4, p5.nickname as p5
                        FROM esports_rosters r
                        JOIN esports_teams t ON r.team_id = t.id
                        JOIN esports_players p1 ON r.player_1 = p1.id
                        JOIN esports_players p2 ON r.player_2 = p2.id
                        JOIN esports_players p3 ON r.player_3 = p3.id
                        JOIN esports_players p4 ON r.player_4 = p4.id
                        JOIN esports_players p5 ON r.player_5 = p5.id
                        WHERE UPPER(t.team_tag) = UPPER(%s)
                          AND r.date_created <= %s
                        ORDER BY r.date_created DESC
                        LIMIT 1
                    """, (team_tag, match_date))
                else:
                    # Get most recent roster
                    cur.execute("""
                        SELECT r.id as roster_id, r.date_created, t.team_tag, t.name as team_name, t.id as team_id,
                               p1.nickname as p1, p2.nickname as p2, p3.nickname as p3, 
                               p4.nickname as p4, p5.nickname as p5
                        FROM esports_rosters r
                        JOIN esports_teams t ON r.team_id = t.id
                        JOIN esports_players p1 ON r.player_1 = p1.id
                        JOIN esports_players p2 ON r.player_2 = p2.id
                        JOIN esports_players p3 ON r.player_3 = p3.id
                        JOIN esports_players p4 ON r.player_4 = p4.id
                        JOIN esports_players p5 ON r.player_5 = p5.id
                        WHERE UPPER(t.team_tag) = UPPER(%s)
                        ORDER BY r.date_created DESC
                        LIMIT 1
                    """, (team_tag,))
                
                row = cur.fetchone()
                if not row:
                    print(f"[DBPlayerMatcher] No roster found for {team_tag}" + 
                          (f" on/before {match_date}" if match_date else ""))
                    return []
                
                # Convert roster row to list of player dicts
                players = []
                for i in range(1, 6):
                    players.append({
                        'nickname': row[f'p{i}'],
                        'team_tag': row['team_tag'],
                        'team_name': row['team_name'],
                        'team_id': row['team_id'],
                    })
                
                print(f"[DBPlayerMatcher] Found roster for {team_tag} dated {row['date_created']}: " +
                      f"{[p['nickname'] for p in players]}")
                return players
                
        except Exception as e:
            try:
                conn.rollback()
            except Exception:
                pass
            print(f"[DBPlayerMatcher] Query error in find_roster_by_date: {e}")
            return []
    
    def fuzzy_search_player(self, ocr_text: str, limit: int = 5) -> List[Dict]:
        """
        Fuzzy search for a player nickname in the database.
        Uses similarity matching to handle OCR errors.
        
        Returns list of {nickname, team_tag, team_name, similarity}.
        """
        conn = self._get_connection()
        if not conn:
            return []
        
        # Clean the OCR text
        clean_text = re.sub(r'[^\w\s]', '', ocr_text).strip()
        if len(clean_text) < 2:
            return []
        
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Use ILIKE for prefix match and similarity for fuzzy match
                # This is more efficient than full trigram search
                cur.execute("""
                    SELECT 
                        p.nickname,
                        t.team_tag,
                        t.name as team_name,
                        t.id as team_id,
                        CASE 
                            WHEN LOWER(p.nickname) = LOWER(%s) THEN 1.0
                            WHEN LOWER(p.nickname) LIKE LOWER(%s) THEN 0.9
                            ELSE similarity(LOWER(p.nickname), LOWER(%s))
                        END as match_score
                    FROM esports_players p
                    LEFT JOIN esports_teams t ON p.team_id = t.id
                    WHERE 
                        LOWER(p.nickname) = LOWER(%s)
                        OR LOWER(p.nickname) LIKE LOWER(%s)
                        OR similarity(LOWER(p.nickname), LOWER(%s)) > 0.3
                    ORDER BY match_score DESC
                    LIMIT %s
                """, (clean_text, f"{clean_text}%", clean_text, 
                      clean_text, f"{clean_text}%", clean_text, limit))
                return cur.fetchall()
        except psycopg2.Error as e:
            # Roll back the aborted transaction so subsequent queries work
            try:
                conn.rollback()
            except Exception:
                pass
            # If similarity function doesn't exist, fall back to LIKE
            if 'function similarity' in str(e).lower():
                print("[DBPlayerMatcher] pg_trgm not available, using LIKE fallback")
                return self._fuzzy_search_fallback(ocr_text, limit)
            print(f"[DBPlayerMatcher] Query error: {e}")
            return []
    
    def _fuzzy_search_fallback(self, ocr_text: str, limit: int = 5) -> List[Dict]:
        """Fallback fuzzy search without pg_trgm extension."""
        conn = self._get_connection()
        if not conn:
            return []
        
        clean_text = re.sub(r'[^\w\s]', '', ocr_text).strip()
        if len(clean_text) < 2:
            return []
        
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Use multiple LIKE patterns
                cur.execute("""
                    SELECT DISTINCT
                        p.nickname,
                        t.team_tag,
                        t.name as team_name,
                        t.id as team_id
                    FROM esports_players p
                    LEFT JOIN esports_teams t ON p.team_id = t.id
                    WHERE 
                        LOWER(p.nickname) = LOWER(%s)
                        OR LOWER(p.nickname) LIKE LOWER(%s)
                        OR LOWER(p.nickname) LIKE LOWER(%s)
                    LIMIT %s
                """, (clean_text, f"{clean_text}%", f"%{clean_text}%", limit))
                results = cur.fetchall()
                
                # Calculate similarity scores manually
                for r in results:
                    r['match_score'] = SequenceMatcher(
                        None, clean_text.lower(), r['nickname'].lower()
                    ).ratio()
                
                return sorted(results, key=lambda x: x['match_score'], reverse=True)
        except Exception as e:
            try:
                conn.rollback()
            except Exception:
                pass
            print(f"[DBPlayerMatcher] Fallback query error: {e}")
            return []
    
    def validate_player_name(self, ocr_name: str, team_tag: Optional[str] = None) -> Optional[MatchedPlayer]:
        """
        Validate an OCR'd player name against the database.
        
        Args:
            ocr_name: Raw OCR text
            team_tag: Optional team tag to narrow search
            
        Returns:
            MatchedPlayer if found, None otherwise
        """
        if not ocr_name or len(ocr_name) < 2:
            return None
        
        # Clean the name
        clean_name = re.sub(r'[^\w\s\-_]', '', ocr_name).strip()
        
        # Strategy 1: If team tag provided, search only that team's roster
        if team_tag:
            team_players = self.find_players_by_team(team_tag)
            for p in team_players:
                # Exact match
                if p['nickname'].lower() == clean_name.lower():
                    return MatchedPlayer(
                        nickname=p['nickname'],
                        team_id=p['team_id'],
                        team_tag=p['team_tag'],
                        team_name=p['team_name'],
                        confidence=1.0,
                        source="exact"
                    )
                # Fuzzy match within team
                similarity = SequenceMatcher(None, clean_name.lower(), p['nickname'].lower()).ratio()
                if similarity > 0.7:
                    return MatchedPlayer(
                        nickname=p['nickname'],
                        team_id=p['team_id'],
                        team_tag=p['team_tag'],
                        team_name=p['team_name'],
                        confidence=similarity,
                        source="fuzzy"
                    )
        
        # Strategy 2: Global fuzzy search (limited)
        results = self.fuzzy_search_player(clean_name)
        if results:
            best = results[0]
            score = best.get('match_score', 0)
            if score > 0.6:  # Threshold for accepting match
                return MatchedPlayer(
                    nickname=best['nickname'],
                    team_id=best.get('team_id'),
                    team_tag=best.get('team_tag'),
                    team_name=best.get('team_name'),
                    confidence=score,
                    source="fuzzy"
                )
        
        return None
    
    def extract_and_validate_from_hud(
        self,
        left_ocr_names: List[str],
        right_ocr_names: List[str],
        left_team_tag: Optional[str] = None,
        right_team_tag: Optional[str] = None,
    ) -> Tuple[List[str], List[str]]:
        """
        Take raw OCR names from HUD and validate against database.
        
        Args:
            left_ocr_names: Raw OCR text from left player cards
            right_ocr_names: Raw OCR text from right player cards
            left_team_tag: Optional team tag (e.g., 'NRG')
            right_team_tag: Optional team tag (e.g., 'FNC')
            
        Returns:
            (validated_left_names, validated_right_names)
        """
        self._left_team_code = left_team_tag
        self._right_team_code = right_team_tag
        
        validated_left = []
        validated_right = []
        
        print(f"[DBPlayerMatcher] Validating {len(left_ocr_names)} left, {len(right_ocr_names)} right OCR names")
        
        # If team tags provided, preload team rosters for faster matching
        left_roster = {}
        right_roster = {}
        
        if left_team_tag:
            for p in self.find_players_by_team(left_team_tag):
                left_roster[p['nickname'].lower()] = p['nickname']
            print(f"[DBPlayerMatcher] Loaded {len(left_roster)} players for {left_team_tag}")
        
        if right_team_tag:
            for p in self.find_players_by_team(right_team_tag):
                right_roster[p['nickname'].lower()] = p['nickname']
            print(f"[DBPlayerMatcher] Loaded {len(right_roster)} players for {right_team_tag}")
        
        # Validate left team names
        for ocr_name in left_ocr_names:
            matched = self._match_against_roster(ocr_name, left_roster, left_team_tag)
            if matched:
                validated_left.append(matched)
                self._match_players[matched.lower()] = MatchedPlayer(
                    nickname=matched, team_id=None, team_tag=left_team_tag,
                    team_name=None, confidence=1.0, source="hud"
                )
            else:
                # Try database fuzzy search
                db_match = self.validate_player_name(ocr_name, left_team_tag)
                if db_match:
                    validated_left.append(db_match.nickname)
                    self._match_players[db_match.nickname.lower()] = db_match
        
        # Validate right team names
        for ocr_name in right_ocr_names:
            matched = self._match_against_roster(ocr_name, right_roster, right_team_tag)
            if matched:
                validated_right.append(matched)
                self._match_players[matched.lower()] = MatchedPlayer(
                    nickname=matched, team_id=None, team_tag=right_team_tag,
                    team_name=None, confidence=1.0, source="hud"
                )
            else:
                # Try database fuzzy search
                db_match = self.validate_player_name(ocr_name, right_team_tag)
                if db_match:
                    validated_right.append(db_match.nickname)
                    self._match_players[db_match.nickname.lower()] = db_match
        
        self._left_players = validated_left
        self._right_players = validated_right
        
        print(f"[DBPlayerMatcher] Validated: left={validated_left}, right={validated_right}")
        
        return validated_left, validated_right
    
    def _match_against_roster(
        self, 
        ocr_name: str, 
        roster: Dict[str, str],
        team_tag: Optional[str] = None
    ) -> Optional[str]:
        """Match OCR name against a team roster."""
        if not ocr_name or not roster:
            return None
        
        clean = re.sub(r'[^\w\s\-_]', '', ocr_name).strip().lower()
        
        # Extract just the player name part (remove team prefix if present)
        parts = clean.split()
        name_only = parts[-1] if parts else clean
        
        # Also check if team prefix is in the name
        if team_tag and len(parts) >= 2:
            prefix = parts[0]
            if prefix.upper() == team_tag.upper() or self._fuzzy_prefix_match(prefix, team_tag):
                name_only = ' '.join(parts[1:])
        
        # Exact match
        if clean in roster:
            return roster[clean]
        if name_only in roster:
            return roster[name_only]
        
        # Fuzzy match
        best_match = None
        best_score = 0.0
        
        for roster_name_lower, roster_name in roster.items():
            # Check OCR-normalized similarity
            score = self._ocr_similarity(name_only, roster_name_lower)
            if score > best_score and score > 0.65:
                best_score = score
                best_match = roster_name
        
        return best_match
    
    def _fuzzy_prefix_match(self, ocr_prefix: str, team_tag: str) -> bool:
        """Check if OCR prefix matches team tag with error tolerance.
        
        Fix 12: handle OCR variants with different lengths (e.g. ENIVY/ENW/NV vs ENVY).
        Uses SequenceMatcher for short strings with ±1-2 char length tolerance.
        """
        ocr_prefix = ocr_prefix.lower()
        team_tag = team_tag.lower()
        
        if ocr_prefix == team_tag:
            return True
        
        # Same length: allow 1 character difference
        if len(ocr_prefix) == len(team_tag):
            diffs = sum(1 for a, b in zip(ocr_prefix, team_tag) if a != b)
            if diffs <= 1:
                return True
        
        # Different lengths (±1 char): OCR may add/drop a character.
        # Use SequenceMatcher ratio — require high similarity for short tags.
        len_diff = abs(len(ocr_prefix) - len(team_tag))
        if len_diff == 1 and 2 <= len(ocr_prefix) <= 7 and 2 <= len(team_tag) <= 7:
            ratio = SequenceMatcher(None, ocr_prefix, team_tag).ratio()
            if ratio >= 0.60:
                return True
        
        return False
    
    def _ocr_similarity(self, s1: str, s2: str) -> float:
        """Calculate OCR-aware similarity."""
        if not s1 or not s2:
            return 0.0
        
        # Normalize for OCR confusions
        s1_norm = self._normalize_for_search(s1)
        s2_norm = self._normalize_for_search(s2)
        
        if s1_norm == s2_norm:
            return 1.0
        
        return SequenceMatcher(None, s1_norm, s2_norm).ratio()
    
    def set_match_players(self, left_names: List[str], right_names: List[str], strict: bool = False):
        """
        Manually set match players (bypasses HUD extraction).
        
        Args:
            left_names: Player names for the left team.
            right_names: Player names for the right team.
            strict: If True, ONLY use the provided names for matching (no DB roster).
                   This is ideal when you know the exact 5v5 roster and want to avoid
                   false matches against historical players.
        
        When strict=False (default):
            Also loads the FULL team roster from database for fuzzy matching.
            This ensures OCR variants can match against ALL historical players on each team.
        
        IMPORTANT: Maintains SEPARATE player pools for each team to handle players
        who have played for both teams (e.g., Crashies was on NRG, now on FNC).
        """
        self._left_players = []
        self._right_players = []
        self._match_players.clear()
        self._left_team_players.clear()
        self._right_team_players.clear()
        self._strict_roster = strict
        
        if strict:
            # STRICT MODE: Only use the provided names — no DB loading
            print(f"[DBPlayerMatcher] STRICT roster mode — matching only against provided players")
            for name in left_names:
                player = MatchedPlayer(
                    nickname=name, team_id=None, team_tag=self._left_team_code,
                    team_name=None, confidence=1.0, source="manual_strict"
                )
                self._left_players.append(name)
                self._left_team_players[name.lower()] = player
                self._match_players[name.lower()] = player
            
            for name in right_names:
                player = MatchedPlayer(
                    nickname=name, team_id=None, team_tag=self._right_team_code,
                    team_name=None, confidence=1.0, source="manual_strict"
                )
                self._right_players.append(name)
                self._right_team_players[name.lower()] = player
                self._match_players[name.lower()] = player
            
            print(f"[DBPlayerMatcher] Set strict players: left={self._left_players}, right={self._right_players}")
            print(f"[DBPlayerMatcher] Total fuzzy match pool: {len(self._match_players)} players (strict)")
            return
        
        # NON-STRICT MODE: Load full team rosters from database - SEPARATE pools for each team
        if self._left_team_code:
            db_left_roster = self.find_players_by_team(self._left_team_code)
            for p in db_left_roster:
                nickname = p['nickname']
                player = MatchedPlayer(
                    nickname=nickname, team_id=p.get('team_id'), team_tag=self._left_team_code,
                    team_name=p.get('team_name'), confidence=0.9, source="db_roster"
                )
                self._left_team_players[nickname.lower()] = player
                # Also add to combined pool for backwards compatibility
                if nickname.lower() not in self._match_players:
                    self._match_players[nickname.lower()] = player
            print(f"[DBPlayerMatcher] Loaded {len(db_left_roster)} historical players for {self._left_team_code}")
        
        if self._right_team_code:
            db_right_roster = self.find_players_by_team(self._right_team_code)
            for p in db_right_roster:
                nickname = p['nickname']
                player = MatchedPlayer(
                    nickname=nickname, team_id=p.get('team_id'), team_tag=self._right_team_code,
                    team_name=p.get('team_name'), confidence=0.9, source="db_roster"
                )
                self._right_team_players[nickname.lower()] = player
                # Also add to combined pool (may override left team player with same name)
                if nickname.lower() not in self._match_players:
                    self._match_players[nickname.lower()] = player
            print(f"[DBPlayerMatcher] Loaded {len(db_right_roster)} historical players for {self._right_team_code}")
        
        # THEN: Validate and store the actual match players (higher confidence)
        for name in left_names:
            # Try to find in database
            db_match = self.validate_player_name(name)
            if db_match:
                self._left_players.append(db_match.nickname)
                self._match_players[db_match.nickname.lower()] = db_match
            else:
                # Accept as-is
                self._left_players.append(name)
                self._match_players[name.lower()] = MatchedPlayer(
                    nickname=name, team_id=None, team_tag=self._left_team_code,
                    team_name=None, confidence=1.0, source="manual"
                )
        
        # Validate and store right players
        for name in right_names:
            db_match = self.validate_player_name(name)
            if db_match:
                self._right_players.append(db_match.nickname)
                self._match_players[db_match.nickname.lower()] = db_match
            else:
                self._right_players.append(name)
                self._match_players[name.lower()] = MatchedPlayer(
                    nickname=name, team_id=None, team_tag=self._right_team_code,
                    team_name=None, confidence=1.0, source="manual"
                )
        
        print(f"[DBPlayerMatcher] Set players: left={self._left_players}, right={self._right_players}")
        print(f"[DBPlayerMatcher] Total fuzzy match pool: {len(self._match_players)} players")
    
    def get_player_team(self, player_name: str) -> Optional[str]:
        """Get team side (left/right) for a player name."""
        if not player_name:
            return None
        
        name_lower = player_name.lower().strip()
        
        # Extract player name part (remove team prefix)
        parts = name_lower.split()
        name_only = parts[-1] if parts else name_lower
        
        # Check left team
        for p in self._left_players:
            p_lower = p.lower()
            if p_lower == name_lower or p_lower == name_only:
                return "left"
            if name_only in p_lower or p_lower in name_only:
                return "left"
            if self._ocr_similarity(p_lower, name_only) > 0.7:
                return "left"
        
        # Check right team
        for p in self._right_players:
            p_lower = p.lower()
            if p_lower == name_lower or p_lower == name_only:
                return "right"
            if name_only in p_lower or p_lower in name_only:
                return "right"
            if self._ocr_similarity(p_lower, name_only) > 0.7:
                return "right"
        
        return None
    
    def match_killfeed_name(self, ocr_text: str, team_code: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
        """
        Match a killfeed OCR text to a known player.
        
        IMPORTANT: Extracts team tag from OCR text (e.g., "FNC" from "FNC Chronicle")
        and uses that to determine which team's player pool to search. This is critical
        for players who have played on multiple teams (e.g., crashies was NRG, now FNC).
        
        Args:
            ocr_text: Raw OCR text from killfeed (may include team prefix like "FNC Chronicle")
            team_code: Fallback team code from color detection. Only used if no team tag
                      can be extracted from the OCR text.
        
        Returns:
            Tuple of (canonical_player_name, detected_team_code) or (None, None).
            The detected_team_code is the team extracted from OCR text, which should be
            used for team assignment instead of color-based detection.
        """
        if not ocr_text or len(ocr_text) < 2:
            return None, None
        
        # Clean the text - remove special chars except alphanumeric, space, dash, underscore
        clean = re.sub(r'[^\w\s\-_]', '', ocr_text).strip().lower()
        
        # Build set of team tags (from detected teams)
        detected_teams = {}
        if self._left_team_code:
            detected_teams[self._left_team_code.lower()] = self._left_team_code
        if self._right_team_code:
            detected_teams[self._right_team_code.lower()] = self._right_team_code
        
        # Extract team tag from OCR text - this takes PRIORITY over color-based team_code
        extracted_team_code = None
        name_only = clean
        
        # Method 1: Split by space - check if first part is a team tag
        parts = clean.split()
        if len(parts) >= 2:
            potential_tag = parts[0]
            for tag_lower, tag_original in detected_teams.items():
                # Exact match or fuzzy match for team tag
                if potential_tag == tag_lower or self._fuzzy_prefix_match(potential_tag, tag_lower):
                    extracted_team_code = tag_original
                    name_only = ' '.join(parts[1:])  # Rest is the player name
                    break
        
        # Method 2: If no space, try stripping detected team prefix (e.g., "FNCChronicle")
        if not extracted_team_code and len(parts) == 1 and detected_teams:
            for tag_lower, tag_original in detected_teams.items():
                if clean.startswith(tag_lower) and len(clean) > len(tag_lower):
                    extracted_team_code = tag_original
                    name_only = clean[len(tag_lower):]
                    break
                # Fix 12: also try fuzzy prefix for ±1 char OCR variants
                # e.g. "enivyrossy" with tag "envy" — try lengths len(tag)±1
                for try_len in [len(tag_lower), len(tag_lower) + 1, len(tag_lower) - 1]:
                    if try_len < 2 or try_len >= len(clean):
                        continue
                    prefix = clean[:try_len]
                    if self._fuzzy_prefix_match(prefix, tag_lower):
                        extracted_team_code = tag_original
                        name_only = clean[try_len:]
                        break
                if extracted_team_code:
                    break
        
        # Determine which team code to use for pool selection
        # PRIORITY: extracted team tag from OCR > passed team_code from color detection
        effective_team_code = extracted_team_code if extracted_team_code else team_code
        
        # Choose which player pool to search based on effective team code
        if effective_team_code:
            effective_team_code_lower = effective_team_code.lower()
            if self._left_team_code and effective_team_code_lower == self._left_team_code.lower():
                search_pool = self._left_team_players
            elif self._right_team_code and effective_team_code_lower == self._right_team_code.lower():
                search_pool = self._right_team_players
            else:
                search_pool = self._match_players  # Fallback to combined pool
        else:
            search_pool = self._match_players  # No team specified, search all
        
        # Direct lookup in chosen pool
        if clean in search_pool:
            return search_pool[clean].nickname, extracted_team_code
        if name_only in search_pool:
            return search_pool[name_only].nickname, extracted_team_code
        
        # Fuzzy match against chosen pool
        best_match = None
        best_score = 0.0
        # In strict mode (only 10 players), use 0.67 threshold — high enough to
        # reject garbage OCR (row mixing, UI artifacts) but low enough that a
        # cleaner frame in the same killfeed display window will still match.
        fuzzy_threshold = 0.67 if getattr(self, '_strict_roster', False) else 0.70
        
        # Reject OCR artifacts: very short text that consists of a single
        # repeated character (e.g. 'III', 'lll', '|||') — these come from
        # weapon icons / UI elements, not player names.
        if len(name_only) <= 3 and len(set(name_only)) == 1:
            return None, extracted_team_code
        
        for player_lower, player_info in search_pool.items():
            # Try matching both the full clean text and extracted name
            score1 = self._ocr_similarity(name_only, player_lower)
            score2 = self._ocr_similarity(clean, player_lower) if clean != name_only else 0
            score = max(score1, score2)
            
            if score > best_score and score > fuzzy_threshold:
                best_score = score
                best_match = player_info.nickname
        
        if best_match and getattr(self, '_strict_roster', False) and best_score < 0.70:
            print(f"[DBPlayerMatcher] Strict fuzzy: '{ocr_text}' -> '{best_match}' (score={best_score:.2f})")
        
        return best_match, extracted_team_code
    
    @property
    def left_players(self) -> List[str]:
        return self._left_players
    
    @property
    def right_players(self) -> List[str]:
        return self._right_players
    
    def close(self):
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None


# Convenience function to load players by team tags
def load_match_players_from_db(
    left_team_tag: str,
    right_team_tag: str,
    db_config: Optional[Dict] = None,
    match_date: Optional[str] = None
) -> Tuple[List[str], List[str]]:
    """
    Load player rosters for two teams from database.
    
    Args:
        left_team_tag: Team tag (e.g., 'NRG')
        right_team_tag: Team tag (e.g., 'FNC')
        db_config: Optional database config
        match_date: Optional date string 'YYYY-MM-DD' to get roster active at that date.
                   If provided, uses esports_rosters table for exact 5-player roster.
                   If None, returns ALL players ever on the team (legacy behavior).
        
    Returns:
        (left_player_names, right_player_names)
    """
    matcher = DatabasePlayerMatcher(db_config)
    
    if match_date:
        # Use date-based roster lookup (recommended - returns exactly 5 players)
        left_players = [p['nickname'] for p in matcher.find_roster_by_date(left_team_tag, match_date)]
        right_players = [p['nickname'] for p in matcher.find_roster_by_date(right_team_tag, match_date)]
    else:
        # Legacy: get ALL players ever on team (may include former players)
        left_players = [p['nickname'] for p in matcher.find_players_by_team(left_team_tag)]
        right_players = [p['nickname'] for p in matcher.find_players_by_team(right_team_tag)]
    
    matcher.close()
    
    return left_players, right_players
