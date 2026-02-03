"""
Database service for accessing esports data from the existing PostgreSQL database.

Uses the schema from backup.sql with 9 tables:
  1. esports_players: id, nickname, first_name, last_name, country, team_id, stats...
  2. esports_teams: id, name, team_tag, location, current_roster_id...
  3. esports_rosters: id, team_id, player_1-5, date_created...
  4. esports_matches: Match records with team IDs, scores, dates
  5. esports_tournaments: Tournament metadata
  6. esports_tournament_placements: Team placements in tournaments
  7. esports_player_games: Per-player per-game statistics (kills, deaths, agent, etc.)
  8. esports_game_scores: Per-map scores within matches
  9. esports_map_veto: Map pick/ban information
"""

import os
import re
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from functools import lru_cache

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    HAS_POSTGRES = True
except ImportError:
    HAS_POSTGRES = False
    print("Warning: psycopg2 not installed. Database player lookup disabled.")


@dataclass
class Player:
    """Player information from database."""
    id: int
    nickname: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    country: Optional[str] = None
    team_id: Optional[int] = None
    team_name: Optional[str] = None
    team_tag: Optional[str] = None


@dataclass
class Team:
    """Team information from database."""
    id: int
    name: str
    team_tag: Optional[str] = None
    location: Optional[str] = None
    current_roster_id: Optional[int] = None


@dataclass
class Match:
    """Match information from esports_matches table."""
    id: int
    team1_id: Optional[int] = None
    team2_id: Optional[int] = None
    team1_score: Optional[int] = None
    team2_score: Optional[int] = None
    tournament_id: Optional[int] = None
    date: Optional[str] = None


@dataclass
class PlayerGame:
    """Per-player per-game stats from esports_player_games table."""
    id: int
    player_id: int
    match_id: int
    map_name: Optional[str] = None
    agent: Optional[str] = None
    kills: int = 0
    deaths: int = 0
    assists: int = 0
    acs: float = 0.0  # Average Combat Score
    adr: float = 0.0  # Average Damage per Round


@dataclass
class GameScore:
    """Per-map score from esports_game_scores table."""
    id: int
    match_id: int
    map_name: Optional[str] = None
    team1_score: int = 0
    team2_score: int = 0


@dataclass
class Tournament:
    """Tournament info from esports_tournaments table."""
    id: int
    name: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    region: Optional[str] = None


class EsportsDatabase:
    """
    Database service for esports data lookup.
    Provides player name fuzzy matching and team information.
    """
    
    _instance: Optional['EsportsDatabase'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self._connection_params = {
            'host': os.environ.get('POSTGRES_HOST', 'localhost'),
            'port': int(os.environ.get('POSTGRES_PORT', '5432')),
            'database': os.environ.get('POSTGRES_DB', 'cloud9'),
            'user': os.environ.get('POSTGRES_USER', 'postgres'),
            'password': os.environ.get('POSTGRES_PASSWORD', ''),
        }
        
        # Caches
        self._players_cache: Dict[int, Player] = {}
        self._players_by_nickname: Dict[str, Player] = {}
        self._teams_cache: Dict[int, Team] = {}
        self._teams_by_tag: Dict[str, Team] = {}
        self._all_nicknames: Set[str] = set()
        
        # Match-specific filter (for better OCR matching)
        self._match_player_filter: Optional[Set[str]] = None
        
        # Load initial data
        self._load_data()
    
    def _get_connection(self):
        """Get database connection."""
        if not HAS_POSTGRES:
            return None
        try:
            return psycopg2.connect(**self._connection_params)
        except Exception as e:
            print(f"Database connection failed: {e}")
            return None
    
    def _load_data(self):
        """Load players and teams from database into cache."""
        conn = self._get_connection()
        if not conn:
            print("Could not connect to database - using empty cache")
            return
        
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Load teams
                cur.execute("""
                    SELECT id, name, team_tag, location, current_roster_id 
                    FROM esports_teams
                """)
                for row in cur.fetchall():
                    team = Team(
                        id=row['id'],
                        name=row['name'],
                        team_tag=row['team_tag'],
                        location=row['location'],
                        current_roster_id=row['current_roster_id'],
                    )
                    self._teams_cache[team.id] = team
                    if team.team_tag:
                        self._teams_by_tag[team.team_tag.upper()] = team
                
                # Load players with team info
                cur.execute("""
                    SELECT p.id, p.nickname, p.first_name, p.last_name, 
                           p.country, p.team_id, t.name as team_name, t.team_tag
                    FROM esports_players p
                    LEFT JOIN esports_teams t ON p.team_id = t.id
                """)
                for row in cur.fetchall():
                    player = Player(
                        id=row['id'],
                        nickname=row['nickname'],
                        first_name=row['first_name'],
                        last_name=row['last_name'],
                        country=row['country'],
                        team_id=row['team_id'],
                        team_name=row['team_name'],
                        team_tag=row['team_tag'],
                    )
                    self._players_cache[player.id] = player
                    self._players_by_nickname[player.nickname.lower()] = player
                    self._all_nicknames.add(player.nickname)
            
            conn.close()
            print(f"Loaded {len(self._players_cache)} players and {len(self._teams_cache)} teams from database")
            
        except Exception as e:
            print(f"Error loading data from database: {e}")
            if conn:
                conn.close()
    
    def set_match_player_filter(self, player_names: List[str]):
        """
        Set a filter to only match against specific players in the current match.
        This significantly improves OCR matching accuracy.
        
        Args:
            player_names: List of player nicknames expected in this match
        """
        self._match_player_filter = set(name.lower() for name in player_names)
        print(f"Match filter set: {len(self._match_player_filter)} players")
    
    def clear_match_filter(self):
        """Clear the match player filter."""
        self._match_player_filter = None
    
    def get_all_players(self) -> List[Player]:
        """Get all players from cache."""
        return list(self._players_cache.values())
    
    def get_filtered_nicknames(self) -> Set[str]:
        """Get player nicknames, filtered by match if filter is set."""
        if self._match_player_filter:
            return {
                nick for nick in self._all_nicknames 
                if nick.lower() in self._match_player_filter
            }
        return self._all_nicknames
    
    def get_player_by_id(self, player_id: int) -> Optional[Player]:
        """Get player by ID."""
        return self._players_cache.get(player_id)
    
    def get_player_by_nickname(self, nickname: str) -> Optional[Player]:
        """Get player by exact nickname (case-insensitive)."""
        return self._players_by_nickname.get(nickname.lower())
    
    def get_team_by_id(self, team_id: int) -> Optional[Team]:
        """Get team by ID."""
        return self._teams_cache.get(team_id)
    
    def get_team_by_tag(self, tag: str) -> Optional[Team]:
        """Get team by tag (e.g., 'FNC', 'NRG')."""
        return self._teams_by_tag.get(tag.upper())
    
    def get_roster_players(self, team_id: int) -> List[Player]:
        """Get current roster players for a team."""
        conn = self._get_connection()
        if not conn:
            return []
        
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get current roster
                cur.execute("""
                    SELECT r.player_1, r.player_2, r.player_3, r.player_4, r.player_5
                    FROM esports_rosters r
                    JOIN esports_teams t ON t.current_roster_id = r.id
                    WHERE t.id = %s
                """, (team_id,))
                row = cur.fetchone()
                
                if not row:
                    return []
                
                player_ids = [row[f'player_{i}'] for i in range(1, 6) if row.get(f'player_{i}')]
                return [self._players_cache[pid] for pid in player_ids if pid in self._players_cache]
                
        except Exception as e:
            print(f"Error getting roster: {e}")
            return []
        finally:
            conn.close()
    
    def fuzzy_match_player(
        self, 
        ocr_text: str, 
        threshold: float = 0.55
    ) -> Optional[Player]:
        """
        Fuzzy match OCR output against player nicknames.
        Returns best match if above threshold, else None.
        
        Args:
            ocr_text: The OCR'd player name
            threshold: Minimum similarity score (0-1)
        """
        if not ocr_text or len(ocr_text) < 2:
            return None
        
        nicknames = self.get_filtered_nicknames()
        if not nicknames:
            return None
        
        ocr_clean = self._normalize_ocr(ocr_text)
        
        best_score = 0.0
        best_player = None
        
        for nickname in nicknames:
            nick_lower = nickname.lower()
            nick_normalized = self._normalize_ocr(nickname)
            
            # Exact match
            if ocr_clean == nick_lower:
                return self._players_by_nickname.get(nick_lower)
            
            # Normalized exact match
            if ocr_clean == nick_normalized:
                return self._players_by_nickname.get(nick_lower)
            
            # Substring containment (for longer names)
            if len(nick_lower) >= 4 and len(ocr_clean) >= 4:
                if nick_lower in ocr_clean or ocr_clean in nick_lower:
                    return self._players_by_nickname.get(nick_lower)
            
            # Calculate similarity
            score = self._calculate_similarity(ocr_clean, nick_lower)
            
            # Bonus for matching prefix
            if len(ocr_clean) >= 2 and len(nick_lower) >= 2:
                if ocr_clean[:2] == nick_lower[:2]:
                    score += 0.15
            
            if score > best_score:
                best_score = score
                best_player = self._players_by_nickname.get(nick_lower)
        
        return best_player if best_score >= threshold else None
    
    def _normalize_ocr(self, text: str) -> str:
        """Normalize text for OCR comparison."""
        s = text.lower().strip()
        
        # Remove common team tag prefixes
        s = re.sub(r'^(nrg|fnc|100t|sen|c9|eg|loud|drx|prx|fut|lev|kru|g2|th|geng|t1|edg|fpx|blg)[\s_]?', '', s, flags=re.IGNORECASE)
        
        # Common OCR confusions
        s = s.replace('0', 'o').replace('1', 'l').replace('|', 'l').replace('!', 'i')
        s = s.replace('$', 's').replace('8', 'b').replace('5', 's')
        s = re.sub(r'rn', 'm', s)  # rn looks like m
        s = re.sub(r'cl', 'd', s)  # cl looks like d
        s = re.sub(r'vv', 'w', s)  # vv looks like w
        
        return s
    
    def _calculate_similarity(self, s1: str, s2: str) -> float:
        """Calculate similarity between two strings."""
        if not s1 or not s2:
            return 0.0
        
        if s1 == s2:
            return 1.0
        
        # LCS-based similarity
        matches = 0
        j = 0
        for c in s1:
            while j < len(s2):
                if s2[j] == c:
                    matches += 1
                    j += 1
                    break
                j += 1
        
        lcs_score = (2.0 * matches) / (len(s1) + len(s2))
        
        # Character frequency overlap
        from collections import Counter
        c1 = Counter(s1)
        c2 = Counter(s2)
        common = sum((c1 & c2).values())
        total = sum((c1 | c2).values())
        freq_score = common / total if total > 0 else 0
        
        return max(lcs_score, freq_score)
    
    # ==========================================
    # Additional table queries
    # ==========================================
    
    def get_match_by_id(self, match_id: int) -> Optional[Match]:
        """Get match details by ID."""
        conn = self._get_connection()
        if not conn:
            return None
        
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, team1_id, team2_id, team1_score, team2_score, 
                           tournament_id, date
                    FROM esports_matches WHERE id = %s
                """, (match_id,))
                row = cur.fetchone()
                if row:
                    return Match(
                        id=row['id'],
                        team1_id=row.get('team1_id'),
                        team2_id=row.get('team2_id'),
                        team1_score=row.get('team1_score'),
                        team2_score=row.get('team2_score'),
                        tournament_id=row.get('tournament_id'),
                        date=str(row.get('date')) if row.get('date') else None,
                    )
        except Exception as e:
            print(f"Error getting match: {e}")
        finally:
            conn.close()
        return None
    
    def get_match_players(self, match_id: int) -> List[Player]:
        """
        Get all players who participated in a specific match.
        Uses esports_player_games to find participants.
        """
        conn = self._get_connection()
        if not conn:
            return []
        
        players = []
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT DISTINCT p.id, p.nickname, p.first_name, p.last_name,
                           p.country, p.team_id, t.name as team_name, t.team_tag
                    FROM esports_player_games pg
                    JOIN esports_players p ON pg.player_id = p.id
                    LEFT JOIN esports_teams t ON p.team_id = t.id
                    WHERE pg.match_id = %s
                """, (match_id,))
                for row in cur.fetchall():
                    players.append(Player(
                        id=row['id'],
                        nickname=row['nickname'],
                        first_name=row['first_name'],
                        last_name=row['last_name'],
                        country=row['country'],
                        team_id=row['team_id'],
                        team_name=row['team_name'],
                        team_tag=row['team_tag'],
                    ))
        except Exception as e:
            print(f"Error getting match players: {e}")
        finally:
            conn.close()
        return players
    
    def get_player_game_stats(self, match_id: int) -> List[PlayerGame]:
        """Get per-player stats for a specific match."""
        conn = self._get_connection()
        if not conn:
            return []
        
        stats = []
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, player_id, match_id, map_name, agent,
                           kills, deaths, assists, acs, adr
                    FROM esports_player_games
                    WHERE match_id = %s
                """, (match_id,))
                for row in cur.fetchall():
                    stats.append(PlayerGame(
                        id=row['id'],
                        player_id=row['player_id'],
                        match_id=row['match_id'],
                        map_name=row.get('map_name'),
                        agent=row.get('agent'),
                        kills=row.get('kills', 0),
                        deaths=row.get('deaths', 0),
                        assists=row.get('assists', 0),
                        acs=row.get('acs', 0.0),
                        adr=row.get('adr', 0.0),
                    ))
        except Exception as e:
            print(f"Error getting player game stats: {e}")
        finally:
            conn.close()
        return stats
    
    def get_game_scores(self, match_id: int) -> List[GameScore]:
        """Get per-map scores for a match."""
        conn = self._get_connection()
        if not conn:
            return []
        
        scores = []
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, match_id, map_name, team1_score, team2_score
                    FROM esports_game_scores
                    WHERE match_id = %s
                """, (match_id,))
                for row in cur.fetchall():
                    scores.append(GameScore(
                        id=row['id'],
                        match_id=row['match_id'],
                        map_name=row.get('map_name'),
                        team1_score=row.get('team1_score', 0),
                        team2_score=row.get('team2_score', 0),
                    ))
        except Exception as e:
            print(f"Error getting game scores: {e}")
        finally:
            conn.close()
        return scores
    
    def get_player_agent_history(self, player_id: int, limit: int = 20) -> List[str]:
        """Get list of agents a player has recently played."""
        conn = self._get_connection()
        if not conn:
            return []
        
        agents = []
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT agent, COUNT(*) as cnt
                    FROM esports_player_games
                    WHERE player_id = %s AND agent IS NOT NULL
                    GROUP BY agent
                    ORDER BY cnt DESC
                    LIMIT %s
                """, (player_id, limit))
                agents = [row['agent'] for row in cur.fetchall()]
        except Exception as e:
            print(f"Error getting player agents: {e}")
        finally:
            conn.close()
        return agents


# Global instance getter
def get_esports_db() -> EsportsDatabase:
    """Get the global EsportsDatabase instance."""
    return EsportsDatabase()
