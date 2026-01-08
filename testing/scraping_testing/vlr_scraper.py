"""
VLR.gg Esports Data Scraper

Scrapes VCT VALORANT esports data from VLR.gg and populates a PostgreSQL database.
Handles: tournaments, teams, placements, matches, players, rosters, map vetos, and game stats.

Rate limit: ~10 requests/second (we use 0.15s delay to stay safe)
"""

import os
import re
import time
import logging
from datetime import datetime, date
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
BASE_URL = "https://www.vlr.gg"
REQUEST_DELAY = 0.15  # seconds between requests (safe for ~6 req/sec)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

# Database connection settings (loaded from .env file)
DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "database": os.getenv("POSTGRES_DB", "cloud9"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
    "port": int(os.getenv("POSTGRES_PORT", "5432"))
}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class Tournament:
    id: int
    name: str
    tier: Optional[str]  # VCT, VCL, or Offseason
    start_date: Optional[date]
    end_date: Optional[date]
    prize_pool: Optional[str]
    location: Optional[str]
    status: str


@dataclass
class Team:
    id: int
    name: str
    team_tag: Optional[str]
    location: Optional[str]


@dataclass
class Player:
    id: int
    nickname: str
    first_name: Optional[str]
    last_name: Optional[str]
    country: Optional[str]
    team_id: Optional[int]


@dataclass
class Match:
    id: int
    phase: Optional[str]
    date: Optional[datetime]
    patch: Optional[str]
    tournament_id: int
    tournament_name: str
    team_1_name: str
    team_1_id: int
    team_1_score: int
    team_2_name: str
    team_2_id: int
    team_2_score: int
    winner: Optional[int]
    format: Optional[str]
    maps: List[str]


@dataclass
class MapVeto:
    match_id: int
    veto_type: str  # 'ban' or 'pick'
    team_id: int
    map_selected: str
    turn: int


@dataclass
class GameScore:
    id: int
    match_id: int
    team_1_score: int
    team_2_score: int
    team_1_id: int
    team_2_id: int
    team_1_name: str
    team_2_name: str
    map_name: str
    winner: Optional[int] = None


@dataclass
class PlayerGameStats:
    match_id: int
    game_id: int
    player_id: int
    team_id: int
    roster_id: Optional[int]
    tournament_id: int
    map_name: str
    agent: str
    rating: Optional[float]
    acs: Optional[int]
    kills: int
    deaths: int
    assists: int
    kast: Optional[str]
    adr: Optional[int]
    hs_percent: Optional[str]
    fk: Optional[int]
    fd: Optional[int]
    opponent_roster_id: Optional[int]
    opponent_team_id: Optional[int]


# ============================================================================
# HTTP Fetcher with Rate Limiting
# ============================================================================

class VLRFetcher:
    """Handles HTTP requests with rate limiting and retry logic."""
    
    def __init__(self, delay: float = REQUEST_DELAY, max_retries: int = 10):
        self.delay = delay
        self.max_retries = max_retries
        self.last_request_time = 0
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
    
    def fetch(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch a page and return BeautifulSoup object with retry logic."""
        # Rate limiting
        elapsed = time.time() - self.last_request_time
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        
        full_url = urljoin(BASE_URL, url) if not url.startswith('http') else url
        
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Fetching: {full_url}")
                
                response = self.session.get(full_url, timeout=30)
                response.raise_for_status()
                
                self.last_request_time = time.time()
                return BeautifulSoup(response.text, 'html.parser')
            
            except requests.HTTPError as e:
                # Don't retry 4xx client errors (404, 403, etc.) - they're permanent
                if e.response is not None and 400 <= e.response.status_code < 500:
                    logger.warning(f"Client error {e.response.status_code} for {url}, not retrying")
                    return None
                # Retry 5xx server errors
                wait_time = min(2 ** attempt * 5, 300)
                if attempt < self.max_retries - 1:
                    logger.warning(f"Fetch failed (attempt {attempt + 1}/{self.max_retries}): {url}")
                    logger.warning(f"Error: {e}")
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to fetch {url} after {self.max_retries} attempts: {e}")
                    return None
            
            except requests.RequestException as e:
                wait_time = min(2 ** attempt * 5, 300)  # 5s, 10s, 20s, 40s, 80s (max 5min)
                
                if attempt < self.max_retries - 1:
                    logger.warning(f"Fetch failed (attempt {attempt + 1}/{self.max_retries}): {url}")
                    logger.warning(f"Error: {e}")
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to fetch {url} after {self.max_retries} attempts: {e}")
                    return None
        
        return None


# ============================================================================
# Parsers
# ============================================================================

class VLRParser:
    """Parses VLR.gg HTML pages."""
    
    @staticmethod
    def parse_completed_events(soup: BeautifulSoup, tier: int = 60) -> List[Tuple[int, str]]:
        """Parse events list page to get completed event IDs and names."""
        events = []
        
        # Find all event items
        event_items = soup.select('a.event-item')
        
        for item in event_items:
            href = item.get('href', '')
            # Check if completed
            status_elem = item.select_one('.event-item-desc-item-status')
            if status_elem and 'completed' in status_elem.get_text().lower():
                # Extract event ID from URL like /event/2283/valorant-champions-2025
                match = re.search(r'/event/(\d+)/', href)
                if match:
                    event_id = int(match.group(1))
                    title_elem = item.select_one('.event-item-title')
                    name = title_elem.get_text(strip=True) if title_elem else f"Event {event_id}"
                    events.append((event_id, name))
        
        return events
    
    @staticmethod
    def parse_max_page(soup: BeautifulSoup) -> int:
        """Parse pagination to get max page number from events list."""
        max_page = 1
        
        # Look for page buttons: <a href="/events/?tier=60&page=2" class="btn mod-page">2</a>
        for link in soup.select('a.btn.mod-page'):
            page_text = link.get_text(strip=True)
            try:
                page_num = int(page_text)
                max_page = max(max_page, page_num)
            except ValueError:
                continue
        
        return max_page
    
    @staticmethod
    def parse_tournament(soup: BeautifulSoup, event_id: int, tier: Optional[str] = None) -> Optional[Tournament]:
        """Parse tournament/event page for tournament info."""
        try:
            # Name
            name_elem = soup.select_one('h1.wf-title')
            name = name_elem.get_text(strip=True) if name_elem else f"Event {event_id}"
            
            # Dates
            dates_elem = soup.select_one('.event-desc-item-value')
            start_date = None
            end_date = None
            
            # Find dates item
            for item in soup.select('.event-desc-item'):
                label = item.select_one('.event-desc-item-label')
                if label and 'dates' in label.get_text().lower():
                    value = item.select_one('.event-desc-item-value')
                    if value:
                        date_text = value.get_text(strip=True)
                        # Parse dates like "Sep 12, 2025 - Oct 5, 2025"
                        dates_match = re.search(r'(\w+ \d+, \d+)\s*-\s*(\w+ \d+, \d+)', date_text)
                        if dates_match:
                            try:
                                start_date = datetime.strptime(dates_match.group(1), '%b %d, %Y').date()
                                end_date = datetime.strptime(dates_match.group(2), '%b %d, %Y').date()
                            except ValueError:
                                pass
            
            # Prize pool
            prize_pool = None
            for item in soup.select('.event-desc-item'):
                label = item.select_one('.event-desc-item-label')
                if label and 'prize' in label.get_text().lower():
                    value = item.select_one('.event-desc-item-value')
                    if value:
                        prize_pool = value.get_text(strip=True)
            
            # Location
            location = None
            for item in soup.select('.event-desc-item'):
                label = item.select_one('.event-desc-item-label')
                if label and 'location' in label.get_text().lower():
                    value = item.select_one('.event-desc-item-value')
                    if value:
                        location = value.get_text(strip=True)
            
            # Status
            today = date.today()
            if end_date and today > end_date:
                status = 'completed'
            elif start_date and today < start_date:
                status = 'upcoming'
            else:
                status = 'ongoing'
            
            return Tournament(
                id=event_id,
                name=name,
                tier=tier,
                start_date=start_date,
                end_date=end_date,
                prize_pool=prize_pool,
                location=location,
                status=status
            )
        
        except Exception as e:
            logger.error(f"Error parsing tournament {event_id}: {e}")
            return None
    
    @staticmethod
    def parse_placements(soup: BeautifulSoup, tournament_id: int, stage: str = "playoffs") -> List[Tuple[str, int, Optional[str]]]:
        """Parse prize distribution table for placements. Returns list of (placement, team_id, prize_money)."""
        placements = []
        
        # Find the prize distribution table
        table = soup.select_one('.wf-table.mod-simple')
        if not table:
            return placements
        
        for row in table.select('tbody tr'):
            cells = row.select('td')
            if len(cells) >= 3:
                # Placement
                place_cell = cells[0]
                place_text = place_cell.get_text(strip=True)
                # Normalize: "1st" "2nd" "5th–6th" etc
                place_text = re.sub(r'<sup>.*?</sup>', '', str(place_cell))
                place_text = BeautifulSoup(place_text, 'html.parser').get_text(strip=True)
                
                # Prize money from second cell
                prize_money = None
                prize_cell = cells[1]
                if prize_cell:
                    prize_text = prize_cell.get_text(strip=True)
                    # Extract amount like "$1,000,000" from "$1,000,000USD"
                    prize_match = re.search(r'(\$[\d,]+)', prize_text)
                    if prize_match:
                        prize_money = prize_match.group(1)
                
                # Team ID from href
                team_link = cells[2].select_one('a[href*="/team/"]')
                if team_link:
                    href = team_link.get('href', '')
                    team_match = re.search(r'/team/(\d+)/', href)
                    if team_match:
                        team_id = int(team_match.group(1))
                        placements.append((place_text, team_id, prize_money))
        
        return placements
    
    @staticmethod
    def parse_stages(soup: BeautifulSoup, event_id: int) -> List[Tuple[str, str, str]]:
        """Parse event page to get available stages. Returns list of (stage_slug, stage_name, full_path)."""
        stages = []
        
        # Find subnav items with stage links like /event/2283/valorant-champions-2025/playoffs
        for link in soup.select('.wf-subnav-item'):
            href = link.get('href', '')
            match = re.search(rf'/event/{event_id}/[^/]+/([^/?]+)', href)
            if match:
                stage_slug = match.group(1)
                title_elem = link.select_one('.wf-subnav-item-title')
                stage_name = title_elem.get_text(strip=True) if title_elem else stage_slug
                # Check if this stage is active (main page)
                is_active = 'mod-active' in link.get('class', [])
                full_path = href if href.startswith('/') else f"/{href}"
                if (stage_slug, stage_name, full_path) not in stages:
                    stages.append((stage_slug, stage_name, full_path))
        
        return stages
    
    @staticmethod
    def parse_team(soup: BeautifulSoup, team_id: int) -> Optional[Team]:
        """Parse team page for team info."""
        try:
            # Name
            name_elem = soup.select_one('h1.wf-title')
            name = name_elem.get_text(strip=True) if name_elem else f"Team {team_id}"
            
            # Team tag
            tag_elem = soup.select_one('.team-header-tag')
            team_tag = tag_elem.get_text(strip=True) if tag_elem else name[:3].upper()
            
            # Location/Country
            country_elem = soup.select_one('.team-header-country')
            location = None
            if country_elem:
                # Remove flag icon text
                location = country_elem.get_text(strip=True)
            
            return Team(
                id=team_id,
                name=name,
                team_tag=team_tag,
                location=location
            )
        
        except Exception as e:
            logger.error(f"Error parsing team {team_id}: {e}")
            return None
    
    @staticmethod
    def parse_matches_list(soup: BeautifulSoup) -> List[int]:
        """Parse matches list page to get match IDs."""
        match_ids = []
        
        # Find all match links
        for link in soup.select('a[href*="/"]'):
            href = link.get('href', '')
            # Match URLs like /542195/paper-rex-vs-xi-lai-gaming...
            match = re.search(r'^/(\d{5,})/[a-z]', href)
            if match:
                match_id = int(match.group(1))
                if match_id not in match_ids:
                    match_ids.append(match_id)
        
        return match_ids
    
    @staticmethod
    def parse_match(soup: BeautifulSoup, match_id: int, tournament_id: int) -> Optional[Match]:
        """Parse match page for match info."""
        try:
            # Tournament name
            event_elem = soup.select_one('.match-header-event')
            tournament_name = ""
            if event_elem:
                name_div = event_elem.select_one('div[style*="font-weight: 700"]')
                if name_div:
                    tournament_name = name_div.get_text(strip=True)
            
            # Phase/Stage
            phase_elem = soup.select_one('.match-header-event-series')
            phase = ' '.join(phase_elem.get_text().split()) if phase_elem else None
            
            # Date
            date_elem = soup.select_one('.moment-tz-convert[data-utc-ts]')
            match_date = None
            if date_elem:
                ts = date_elem.get('data-utc-ts', '')
                try:
                    # Handle both timestamp and date string formats
                    if ts.isdigit():
                        match_date = datetime.fromtimestamp(int(ts))
                    else:
                        match_date = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    pass
            
            # Patch
            patch = None
            patch_elem = soup.select_one('.match-header-date div[style*="font-style: italic"]')
            if patch_elem:
                patch_text = patch_elem.get_text(strip=True)
                if 'Patch' in patch_text:
                    patch = patch_text.replace('Patch', '').strip()
            
            # Teams
            team_1_link = soup.select_one('.match-header-link.mod-1')
            team_2_link = soup.select_one('.match-header-link.mod-2')
            
            team_1_id = team_1_name = team_2_id = team_2_name = None
            
            if team_1_link:
                href = team_1_link.get('href', '')
                match = re.search(r'/team/(\d+)/', href)
                if match:
                    team_1_id = int(match.group(1))
                name_elem = team_1_link.select_one('.wf-title-med')
                team_1_name = name_elem.get_text(strip=True) if name_elem else None
            
            if team_2_link:
                href = team_2_link.get('href', '')
                match = re.search(r'/team/(\d+)/', href)
                if match:
                    team_2_id = int(match.group(1))
                name_elem = team_2_link.select_one('.wf-title-med')
                team_2_name = name_elem.get_text(strip=True) if name_elem else None
            
            # Scores
            score_elem = soup.select_one('.match-header-vs-score')
            team_1_score = team_2_score = 0
            
            if score_elem:
                # Score spans are in order: team_1_score : team_2_score
                # The class (winner/loser) tells us who won, not the position
                spans = score_elem.select('span')
                score_spans = [s for s in spans if 'match-header-vs-score-winner' in (s.get('class') or []) 
                              or 'match-header-vs-score-loser' in (s.get('class') or [])]
                
                if len(score_spans) >= 2:
                    # First span is team_1's score, second is team_2's score
                    team_1_score = int(score_spans[0].get_text(strip=True))
                    team_2_score = int(score_spans[1].get_text(strip=True))
            
            # Format (bo1, bo3, bo5)
            format_elem = soup.select_one('.match-header-vs-note')
            match_format = None
            for note in soup.select('.match-header-vs-note'):
                text = note.get_text(strip=True).lower()
                if text in ['bo1', 'bo3', 'bo5']:
                    match_format = text
                    break
            
            # Determine winner
            winner = None
            if team_1_score > team_2_score:
                winner = team_1_id
            elif team_2_score > team_1_score:
                winner = team_2_id
            
            # Maps (from game tabs)
            maps = []
            for game_tab in soup.select('.vm-stats-gamesnav-item'):
                map_name = game_tab.get_text(strip=True)
                if map_name.lower() not in ['all maps', 'all']:
                    # Remove leading numbers (e.g., "1Bind" -> "Bind")
                    map_name = re.sub(r'^\d+', '', map_name)
                    maps.append(map_name)
            
            return Match(
                id=match_id,
                phase=phase,
                date=match_date,
                patch=patch,
                tournament_id=tournament_id,
                tournament_name=tournament_name,
                team_1_name=team_1_name or "Unknown",
                team_1_id=team_1_id or 0,
                team_1_score=team_1_score,
                team_2_name=team_2_name or "Unknown",
                team_2_id=team_2_id or 0,
                team_2_score=team_2_score,
                winner=winner,
                format=match_format,
                maps=maps[:5]  # Max 5 maps
            )
        
        except Exception as e:
            logger.error(f"Error parsing match {match_id}: {e}")
            return None
    
    @staticmethod
    def parse_map_veto(soup: BeautifulSoup, match_id: int, team_tags: Dict[str, int]) -> List[MapVeto]:
        """Parse map veto from match-header-note. team_tags maps tag -> team_id."""
        vetos = []
        
        veto_elem = soup.select_one('.match-header-note')
        if not veto_elem:
            return vetos
        
        veto_text = veto_elem.get_text(strip=True)
        # Parse: "XLG ban Lotus; PRX ban Abyss; XLG pick Bind; PRX pick Sunset; ..."
        
        parts = veto_text.split(';')
        turn = 1
        
        for part in parts:
            part = part.strip()
            if 'remains' in part.lower():
                continue  # Skip "Ascent remains"
            
            # Pattern: "TAG ban/pick MapName"
            match = re.match(r'(\w+)\s+(ban|pick)\s+(\w+)', part, re.IGNORECASE)
            if match:
                tag = match.group(1).upper()
                veto_type = match.group(2).lower()
                map_name = match.group(3)
                
                team_id = team_tags.get(tag)
                if team_id:
                    vetos.append(MapVeto(
                        match_id=match_id,
                        veto_type=veto_type,
                        team_id=team_id,
                        map_selected=map_name,
                        turn=turn
                    ))
                    turn += 1
        
        return vetos
    
    @staticmethod
    def parse_player(soup: BeautifulSoup, player_id: int) -> Optional[Player]:
        """Parse player page for player info."""
        try:
            # Nickname from h1.wf-title or title
            nickname = f"Player {player_id}"
            h1_elem = soup.select_one('h1.wf-title')
            if h1_elem:
                nickname = h1_elem.get_text(strip=True)
            else:
                title_elem = soup.select_one('title')
                if title_elem:
                    title_text = title_elem.get_text()
                    # "something: Valorant Player Profile | VLR.gg"
                    match = re.match(r'^([^:]+):', title_text)
                    if match:
                        nickname = match.group(1).strip()
            
            # Name from meta description
            first_name = last_name = None
            meta_desc = soup.select_one('meta[name="description"]')
            if meta_desc:
                content = meta_desc.get('content', '')
                # "something (Ilya Petrov) Valorant player..."
                name_match = re.search(r'\(([^)]+)\)', content)
                if name_match:
                    full_name = name_match.group(1)
                    parts = full_name.split()
                    if len(parts) >= 2:
                        first_name = parts[0]
                        last_name = ' '.join(parts[1:])
                    elif len(parts) == 1:
                        first_name = parts[0]
            
            # Country from flag
            country = None
            flag_elem = soup.select_one('.player-header i.flag')
            if flag_elem:
                classes = flag_elem.get('class', [])
                for cls in classes:
                    if cls.startswith('mod-') and cls != 'mod-':
                        country_code = cls.replace('mod-', '').upper()
                        # Map codes to names (simplified)
                        country_map = {
                            'RU': 'Russia', 'US': 'United States', 'KR': 'South Korea',
                            'CN': 'China', 'JP': 'Japan', 'SG': 'Singapore',
                            'ID': 'Indonesia', 'BR': 'Brazil', 'FR': 'France',
                            'DE': 'Germany', 'ES': 'Spain', 'PH': 'Philippines',
                            'TH': 'Thailand', 'VN': 'Vietnam', 'TR': 'Turkey',
                            'PL': 'Poland', 'SE': 'Sweden', 'DK': 'Denmark',
                            'CA': 'Canada', 'AR': 'Argentina', 'CL': 'Chile',
                            'MX': 'Mexico', 'AU': 'Australia', 'FI': 'Finland',
                            'MY': 'MY',
                        }
                        country = country_map.get(country_code, country_code)
                        break
            
            # Current team from first team link in wf-module-item mod-first
            current_team_id = None
            team_link = soup.select_one('a.wf-module-item.mod-first[href*="/team/"]')
            if team_link:
                href = team_link.get('href', '')
                team_match = re.search(r'/team/(\d+)/', href)
                if team_match:
                    current_team_id = int(team_match.group(1))
            
            return Player(
                id=player_id,
                nickname=nickname,
                first_name=first_name,
                last_name=last_name,
                country=country,
                team_id=current_team_id
            )
        
        except Exception as e:
            logger.error(f"Error parsing player {player_id}: {e}")
            return None
    
    @staticmethod
    def parse_game_ids(soup: BeautifulSoup) -> List[int]:
        """Parse match page to get individual game IDs from map nav tabs."""
        game_ids = []
        
        # Look for game tabs with data-game-id attribute (skip 'all' tab and disabled maps)
        for item in soup.select('.vm-stats-gamesnav-item[data-game-id]'):
            game_id_str = item.get('data-game-id', '')
            disabled = item.get('data-disabled', '0')
            
            # Skip 'all' tab and disabled maps (not played)
            if game_id_str == 'all' or disabled == '1':
                continue
            
            try:
                game_id = int(game_id_str)
                if game_id not in game_ids:
                    game_ids.append(game_id)
            except ValueError:
                continue
        
        return game_ids
    
    @staticmethod
    def parse_player_ids_from_match(soup: BeautifulSoup) -> List[int]:
        """Extract all player IDs from a match page."""
        player_ids = []
        
        for link in soup.select('a[href*="/player/"]'):
            href = link.get('href', '')
            match = re.search(r'/player/(\d+)/', href)
            if match:
                player_id = int(match.group(1))
                if player_id not in player_ids:
                    player_ids.append(player_id)
        
        return player_ids
    
    @staticmethod
    def parse_game_stats(soup: BeautifulSoup, match_id: int, game_id: int, 
                         tournament_id: int, team_1_id: int, team_2_id: int) -> Tuple[Optional[GameScore], List[PlayerGameStats]]:
        """Parse individual game/map page for scores and player stats."""
        
        game_score = None
        player_stats = []
        
        try:
            # Find the specific game container by game_id
            game_container = soup.select_one(f'.vm-stats-game[data-game-id="{game_id}"]')
            if not game_container:
                logger.warning(f"Game container not found for game {game_id}")
                return None, []
            
            # Find the map name from the game nav
            map_name = "Unknown"
            game_nav = soup.select_one(f'.vm-stats-gamesnav-item[data-game-id="{game_id}"]')
            if game_nav:
                map_name = game_nav.get_text(strip=True)
                # Remove leading number prefix (e.g., "1Bind" -> "Bind")
                map_name = re.sub(r'^\d+', '', map_name)
            
            # Score from game header in the container
            team_1_score = team_2_score = 0
            game_header = game_container.select_one('.vm-stats-game-header')
            if game_header:
                scores = game_header.select('.score')
                if len(scores) >= 2:
                    try:
                        team_1_score = int(scores[0].get_text(strip=True))
                        team_2_score = int(scores[1].get_text(strip=True))
                    except ValueError:
                        pass
            
            # Get team names
            team_1_name = team_2_name = ""
            team_headers = soup.select('.match-header-link-name .wf-title-med')
            if len(team_headers) >= 2:
                team_1_name = team_headers[0].get_text(strip=True)
                team_2_name = team_headers[1].get_text(strip=True)
            
            # Determine winner based on scores
            winner = None
            if team_1_score > team_2_score:
                winner = team_1_id
            elif team_2_score > team_1_score:
                winner = team_2_id
            
            game_score = GameScore(
                id=game_id,
                match_id=match_id,
                team_1_score=team_1_score,
                team_2_score=team_2_score,
                team_1_id=team_1_id,
                team_2_id=team_2_id,
                team_1_name=team_1_name,
                team_2_name=team_2_name,
                map_name=map_name,
                winner=winner
            )
            
            # Player stats from tables within this game's container
            stats_tables = game_container.select('.wf-table-inset.mod-overview')
            
            # Process first two tables (team 1 and team 2)
            for table_idx, stats_table in enumerate(stats_tables[:2]):
                current_team_id = team_1_id if table_idx == 0 else team_2_id
                tbody = stats_table.select_one('tbody')
                if not tbody:
                    continue
                
                for row in tbody.select('tr'):
                    player_cell = row.select_one('.mod-player a')
                    if not player_cell:
                        continue
                    
                    href = player_cell.get('href', '')
                    player_match = re.search(r'/player/(\d+)/', href)
                    if not player_match:
                        continue
                    
                    player_id = int(player_match.group(1))
                    
                    # Agent
                    agent = "Unknown"
                    agent_img = row.select_one('.mod-agents img')
                    if agent_img:
                        agent = agent_img.get('alt', 'Unknown')
                    
                    # Stats
                    def get_stat(selector: str, default=None):
                        elem = row.select_one(selector)
                        if elem:
                            text = elem.get_text(strip=True)
                            return text if text else default
                        return default
                    
                    def get_stat_int(selector: str) -> Optional[int]:
                        val = get_stat(selector)
                        if val:
                            try:
                                return int(val.replace('%', '').replace('+', '').replace('-', ''))
                            except ValueError:
                                return None
                        return None
                    
                    def get_stat_float(selector: str) -> Optional[float]:
                        val = get_stat(selector)
                        if val:
                            try:
                                return float(val)
                            except ValueError:
                                return None
                        return None
                    
                    # Extract stats from the row
                    stats = PlayerGameStats(
                        match_id=match_id,
                        game_id=game_id,
                        player_id=player_id,
                        team_id=current_team_id,
                        roster_id=None,  # Set later
                        tournament_id=tournament_id,
                        map_name=map_name,
                        agent=agent,
                        rating=None,
                        acs=None,
                        kills=0,
                        deaths=0,
                        assists=0,
                        kast=None,
                        adr=None,
                        hs_percent=None,
                        fk=None,
                        fd=None,
                        opponent_roster_id=None,
                        opponent_team_id=team_2_id if current_team_id == team_1_id else team_1_id
                    )
                    
                    # Try to extract numeric stats from cells
                    cells = row.select('td.mod-stat')
                    stat_values = []
                    for cell in cells:
                        # Get the "both" side value
                        both_elem = cell.select_one('.mod-both')
                        if both_elem:
                            stat_values.append(both_elem.get_text(strip=True))
                        else:
                            stat_values.append(cell.get_text(strip=True))
                    
                    # Assign based on expected order: R, ACS, K, D, A, +/-, KAST, ADR, HS%, FK, FD, +/-
                    if len(stat_values) >= 12:
                        try:
                            stats.rating = float(stat_values[0]) if stat_values[0] else None
                            stats.acs = int(stat_values[1]) if stat_values[1] else None
                            stats.kills = int(stat_values[2]) if stat_values[2] else 0
                            stats.deaths = int(stat_values[3].replace('/', '').strip()) if stat_values[3] else 0
                            stats.assists = int(stat_values[4]) if stat_values[4] else 0
                            stats.kast = stat_values[6] if stat_values[6] else None
                            stats.adr = int(stat_values[7]) if stat_values[7] else None
                            stats.hs_percent = stat_values[8] if stat_values[8] else None
                            stats.fk = int(stat_values[9]) if stat_values[9] else None
                            stats.fd = int(stat_values[10]) if stat_values[10] else None
                        except (ValueError, IndexError):
                            pass
                    
                    player_stats.append(stats)
        
        except Exception as e:
            logger.error(f"Error parsing game stats for game {game_id}: {e}")
        
        return game_score, player_stats


# ============================================================================
# Database Operations
# ============================================================================

class Database:
    """Handles PostgreSQL database operations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.conn = None
    
    def connect(self):
        """Connect to the database."""
        self.conn = psycopg2.connect(**self.config)
        self.conn.autocommit = False
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
    
    def commit(self):
        """Commit the current transaction."""
        if self.conn:
            self.conn.commit()
    
    def rollback(self):
        """Rollback the current transaction."""
        if self.conn:
            self.conn.rollback()
    
    def execute_schema(self, schema_file: str):
        """Execute schema SQL file."""
        with open(schema_file, 'r') as f:
            sql = f.read()
        
        with self.conn.cursor() as cur:
            cur.execute(sql)
        self.commit()
        logger.info("Schema created successfully")
    
    # Tournament operations
    def upsert_tournament(self, t: Tournament):
        """Insert or update a tournament."""
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO esports_tournaments (id, name, tier, start_date, end_date, prize_pool, location, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name,
                    tier = EXCLUDED.tier,
                    start_date = EXCLUDED.start_date,
                    end_date = EXCLUDED.end_date,
                    prize_pool = EXCLUDED.prize_pool,
                    location = EXCLUDED.location,
                    status = EXCLUDED.status
            """, (t.id, t.name, t.tier, t.start_date, t.end_date, t.prize_pool, t.location, t.status))
    
    def tournament_exists(self, tournament_id: int) -> bool:
        """Check if a tournament exists."""
        with self.conn.cursor() as cur:
            cur.execute("SELECT 1 FROM esports_tournaments WHERE id = %s", (tournament_id,))
            return cur.fetchone() is not None
    
    # Team operations
    def upsert_team(self, t: Team):
        """Insert or update a team."""
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO esports_teams (id, name, team_tag, location)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name,
                    team_tag = EXCLUDED.team_tag,
                    location = EXCLUDED.location
            """, (t.id, t.name, t.team_tag, t.location))
    
    def team_exists(self, team_id: int) -> bool:
        """Check if a team exists."""
        with self.conn.cursor() as cur:
            cur.execute("SELECT 1 FROM esports_teams WHERE id = %s", (team_id,))
            return cur.fetchone() is not None
    
    def player_exists(self, player_id: int) -> bool:
        """Check if a player exists."""
        with self.conn.cursor() as cur:
            cur.execute("SELECT 1 FROM esports_players WHERE id = %s", (player_id,))
            return cur.fetchone() is not None
    
    def get_team_tag(self, team_id: int) -> Optional[str]:
        """Get team tag by ID."""
        with self.conn.cursor() as cur:
            cur.execute("SELECT team_tag FROM esports_teams WHERE id = %s", (team_id,))
            row = cur.fetchone()
            return row[0] if row else None
    
    def get_team_id_by_tag(self, tag: str) -> Optional[int]:
        """Get team ID by tag."""
        with self.conn.cursor() as cur:
            cur.execute("SELECT id FROM esports_teams WHERE UPPER(team_tag) = %s", (tag.upper(),))
            row = cur.fetchone()
            return row[0] if row else None
    
    # Placement operations
    def upsert_placement(self, tournament_id: int, placement: str, team_id: int, 
                         prize_money: Optional[str] = None, stage: str = "playoffs"):
        """Insert or update a tournament placement."""
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO esports_tournament_placements (tournament_id, placement, esports_team_id, prize_money, stage)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (tournament_id, esports_team_id, stage) DO UPDATE SET
                    placement = EXCLUDED.placement,
                    prize_money = EXCLUDED.prize_money
            """, (tournament_id, placement, team_id, prize_money, stage))
    
    # Player operations
    def upsert_player(self, p: Player):
        """Insert or update a player."""
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO esports_players (id, nickname, first_name, last_name, country, team_id)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    nickname = EXCLUDED.nickname,
                    first_name = COALESCE(EXCLUDED.first_name, esports_players.first_name),
                    last_name = COALESCE(EXCLUDED.last_name, esports_players.last_name),
                    country = COALESCE(EXCLUDED.country, esports_players.country),
                    team_id = COALESCE(EXCLUDED.team_id, esports_players.team_id)
            """, (p.id, p.nickname, p.first_name, p.last_name, p.country, p.team_id))
    
    def player_exists(self, player_id: int) -> bool:
        """Check if a player exists."""
        with self.conn.cursor() as cur:
            cur.execute("SELECT 1 FROM esports_players WHERE id = %s", (player_id,))
            return cur.fetchone() is not None
    
    # Match operations
    def upsert_match(self, m: Match):
        """Insert or update a match."""
        maps = m.maps + [None] * (5 - len(m.maps))  # Pad to 5
        
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO esports_matches (id, phase, date, patch, tournament_id, tournament_name,
                    team_1_name, team_1_id, team_1_score, team_2_name, team_2_id, team_2_score,
                    winner, format, map_1, map_2, map_3, map_4, map_5)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    phase = EXCLUDED.phase,
                    date = EXCLUDED.date,
                    patch = EXCLUDED.patch,
                    tournament_id = EXCLUDED.tournament_id,
                    tournament_name = EXCLUDED.tournament_name,
                    team_1_name = EXCLUDED.team_1_name,
                    team_1_id = EXCLUDED.team_1_id,
                    team_1_score = EXCLUDED.team_1_score,
                    team_2_name = EXCLUDED.team_2_name,
                    team_2_id = EXCLUDED.team_2_id,
                    team_2_score = EXCLUDED.team_2_score,
                    winner = EXCLUDED.winner,
                    format = EXCLUDED.format,
                    map_1 = EXCLUDED.map_1,
                    map_2 = EXCLUDED.map_2,
                    map_3 = EXCLUDED.map_3,
                    map_4 = EXCLUDED.map_4,
                    map_5 = EXCLUDED.map_5
            """, (m.id, m.phase, m.date, m.patch, m.tournament_id, m.tournament_name,
                  m.team_1_name, m.team_1_id, m.team_1_score, m.team_2_name, m.team_2_id, m.team_2_score,
                  m.winner, m.format, maps[0], maps[1], maps[2], maps[3], maps[4]))
    
    def match_exists(self, match_id: int) -> bool:
        """Check if a match exists."""
        with self.conn.cursor() as cur:
            cur.execute("SELECT 1 FROM esports_matches WHERE id = %s", (match_id,))
            return cur.fetchone() is not None
    
    # Map veto operations
    def insert_map_veto(self, v: MapVeto):
        """Insert a map veto."""
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO esports_map_veto (match_id, type, team_id, map_selected, turn)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
            """, (v.match_id, v.veto_type, v.team_id, v.map_selected, v.turn))
    
    # Roster operations
    def get_or_create_roster(self, team_id: int, player_ids: List[int], match_date: Optional[date]) -> Optional[int]:
        """Get existing roster or create new one. Returns roster ID or None if players missing."""
        # Filter out any player IDs that don't exist in the database
        valid_player_ids = []
        with self.conn.cursor() as cur:
            for pid in player_ids:
                if pid:
                    cur.execute("SELECT 1 FROM esports_players WHERE id = %s", (pid,))
                    if cur.fetchone():
                        valid_player_ids.append(pid)
                    else:
                        logger.warning(f"Player {pid} not found in database, excluding from roster")
        
        if len(valid_player_ids) < 3:  # Need at least 3 valid players for a roster
            logger.warning(f"Not enough valid players for roster (have {len(valid_player_ids)}), skipping")
            return None
        
        # Sort player IDs for consistent matching
        sorted_ids = sorted(valid_player_ids)
        # Pad to 5 players
        while len(sorted_ids) < 5:
            sorted_ids.append(None)
        
        with self.conn.cursor() as cur:
            # Check if roster exists
            cur.execute("""
                SELECT id, date_created FROM esports_rosters
                WHERE team_id = %s AND player_1 = %s AND player_2 = %s 
                    AND player_3 = %s AND player_4 = %s AND player_5 = %s
            """, (team_id, sorted_ids[0], sorted_ids[1], sorted_ids[2], sorted_ids[3], sorted_ids[4]))
            
            row = cur.fetchone()
            if row:
                roster_id, existing_date = row
                # Update date if this match is older
                if match_date and existing_date and match_date < existing_date:
                    cur.execute("""
                        UPDATE esports_rosters SET date_created = %s WHERE id = %s
                    """, (match_date, roster_id))
                return roster_id
            
            # Create new roster
            cur.execute("""
                INSERT INTO esports_rosters (team_id, player_1, player_2, player_3, player_4, player_5, date_created)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (team_id, sorted_ids[0], sorted_ids[1], sorted_ids[2], sorted_ids[3], sorted_ids[4], match_date))
            
            return cur.fetchone()[0]
    
    def update_team_current_roster(self, team_id: int, roster_id: int, roster_date: Optional[date]):
        """Update team's current roster if this one is newer."""
        with self.conn.cursor() as cur:
            # Get current roster date
            cur.execute("""
                SELECT r.date_created FROM esports_teams t
                LEFT JOIN esports_rosters r ON t.current_roster_id = r.id
                WHERE t.id = %s
            """, (team_id,))
            
            row = cur.fetchone()
            current_date = row[0] if row else None
            
            # Update if no current roster or this one is newer
            if current_date is None or (roster_date and roster_date > current_date):
                cur.execute("""
                    UPDATE esports_teams SET current_roster_id = %s WHERE id = %s
                """, (roster_id, team_id))
    
    # Game score operations
    def upsert_game_score(self, g: GameScore):
        """Insert or update a game score."""
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO esports_game_scores (id, match_id, team_1_score, team_2_score,
                    team_1_id, team_2_id, team_1_name, team_2_name, map, winner)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    match_id = EXCLUDED.match_id,
                    team_1_score = EXCLUDED.team_1_score,
                    team_2_score = EXCLUDED.team_2_score,
                    team_1_id = EXCLUDED.team_1_id,
                    team_2_id = EXCLUDED.team_2_id,
                    team_1_name = EXCLUDED.team_1_name,
                    team_2_name = EXCLUDED.team_2_name,
                    map = EXCLUDED.map,
                    winner = EXCLUDED.winner
            """, (g.id, g.match_id, g.team_1_score, g.team_2_score,
                  g.team_1_id, g.team_2_id, g.team_1_name, g.team_2_name, g.map_name, g.winner))
    
    # Player game stats operations
    def insert_player_game_stats(self, stats: PlayerGameStats):
        """Insert player game stats."""
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO esports_player_games (match_id, game_id, player_id, team_id, roster_id,
                    tournament_id, map, agent, rating, acs, kills, deaths, assists, kast, adr,
                    hs_percent, fk, fd, opponent_roster_id, opponent_team_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (stats.match_id, stats.game_id, stats.player_id, stats.team_id, stats.roster_id,
                  stats.tournament_id, stats.map_name, stats.agent, stats.rating, stats.acs,
                  stats.kills, stats.deaths, stats.assists, stats.kast, stats.adr,
                  stats.hs_percent, stats.fk, stats.fd, stats.opponent_roster_id, stats.opponent_team_id))

    # Team match wins/losses
    def update_team_match_result(self, winner_id: int, loser_id: int):
        """Update match wins/losses for teams after a match."""
        with self.conn.cursor() as cur:
            if winner_id:
                cur.execute("""
                    UPDATE esports_teams SET match_wins = match_wins + 1 WHERE id = %s
                """, (winner_id,))
            if loser_id:
                cur.execute("""
                    UPDATE esports_teams SET match_losses = match_losses + 1 WHERE id = %s
                """, (loser_id,))
    
    # Roster map wins/losses
    def update_roster_map_result(self, winner_roster_id: Optional[int], loser_roster_id: Optional[int]):
        """Update map wins/losses for rosters after a game."""
        with self.conn.cursor() as cur:
            if winner_roster_id:
                cur.execute("""
                    UPDATE esports_rosters SET map_wins = map_wins + 1 WHERE id = %s
                """, (winner_roster_id,))
            if loser_roster_id:
                cur.execute("""
                    UPDATE esports_rosters SET map_losses = map_losses + 1 WHERE id = %s
                """, (loser_roster_id,))
    
    # Team titles
    def add_team_title(self, team_id: int, tournament_id: int):
        """Add a tournament title to a team's titles array."""
        with self.conn.cursor() as cur:
            cur.execute("""
                UPDATE esports_teams 
                SET titles = array_append(titles, %s) 
                WHERE id = %s AND NOT (%s = ANY(titles))
            """, (tournament_id, team_id, tournament_id))
    
    # Player titles
    def add_player_title(self, player_id: int, tournament_id: int):
        """Add a tournament title to a player's titles array."""
        with self.conn.cursor() as cur:
            cur.execute("""
                UPDATE esports_players 
                SET titles = array_append(titles, %s) 
                WHERE id = %s AND NOT (%s = ANY(titles))
            """, (tournament_id, player_id, tournament_id))
    
    # Placement player tracking
    def add_player_to_placement(self, tournament_id: int, team_id: int, player_id: int, stage: str = "playoffs"):
        """Add a player to a team's placement players array."""
        with self.conn.cursor() as cur:
            cur.execute("""
                UPDATE esports_tournament_placements 
                SET players = array_append(players, %s) 
                WHERE tournament_id = %s AND esports_team_id = %s AND stage = %s 
                    AND NOT (%s = ANY(players))
            """, (player_id, tournament_id, team_id, stage, player_id))
    
    def get_placement_players(self, tournament_id: int, team_id: int, stage: str = "playoffs") -> List[int]:
        """Get all player IDs from a team's placement."""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT players FROM esports_tournament_placements 
                WHERE tournament_id = %s AND esports_team_id = %s AND stage = %s
            """, (tournament_id, team_id, stage))
            row = cur.fetchone()
            return row[0] if row and row[0] else []
    
    # Player stats calculation
    def update_player_stats(self, player_id: int):
        """Calculate and update all-time and last-60-days stats for a player."""
        with self.conn.cursor() as cur:
            # All-time stats
            cur.execute("""
                SELECT 
                    COUNT(*) as total_maps,
                    AVG(pg.rating) as avg_rating,
                    AVG(pg.acs) as avg_acs,
                    SUM(pg.kills) as total_kills,
                    SUM(pg.deaths) as total_deaths,
                    SUM(pg.assists) as total_assists,
                    AVG(pg.kills) as avg_kills,
                    AVG(pg.deaths) as avg_deaths,
                    AVG(pg.assists) as avg_assists,
                    AVG(CAST(REPLACE(pg.kast, '%%', '') AS DECIMAL)) as avg_kast,
                    AVG(pg.adr) as avg_adr,
                    AVG(CAST(REPLACE(pg.hs_percent, '%%', '') AS DECIMAL)) as avg_hs,
                    SUM(pg.fk) as total_fk,
                    SUM(pg.fd) as total_fd,
                    AVG(pg.fk) as avg_fk,
                    AVG(pg.fd) as avg_fd
                FROM esports_player_games pg
                WHERE pg.player_id = %s
            """, (player_id,))
            all_time = cur.fetchone()
            
            # All-time map wins/losses
            cur.execute("""
                SELECT 
                    COUNT(*) FILTER (WHERE gs.winner = pg.team_id) as map_wins,
                    COUNT(*) FILTER (WHERE gs.winner IS NOT NULL AND gs.winner != pg.team_id) as map_losses
                FROM esports_player_games pg
                JOIN esports_game_scores gs ON pg.game_id = gs.id
                WHERE pg.player_id = %s
            """, (player_id,))
            all_time_wl = cur.fetchone()
            
            # Last 60 days stats (using match date)
            cur.execute("""
                SELECT 
                    COUNT(*) as total_maps,
                    AVG(pg.rating) as avg_rating,
                    AVG(pg.acs) as avg_acs,
                    SUM(pg.kills) as total_kills,
                    SUM(pg.deaths) as total_deaths,
                    SUM(pg.assists) as total_assists,
                    AVG(pg.kills) as avg_kills,
                    AVG(pg.deaths) as avg_deaths,
                    AVG(pg.assists) as avg_assists,
                    AVG(CAST(REPLACE(pg.kast, '%%', '') AS DECIMAL)) as avg_kast,
                    AVG(pg.adr) as avg_adr,
                    AVG(CAST(REPLACE(pg.hs_percent, '%%', '') AS DECIMAL)) as avg_hs,
                    SUM(pg.fk) as total_fk,
                    SUM(pg.fd) as total_fd,
                    AVG(pg.fk) as avg_fk,
                    AVG(pg.fd) as avg_fd
                FROM esports_player_games pg
                JOIN esports_matches m ON pg.match_id = m.id
                WHERE pg.player_id = %s 
                    AND m.date >= (CURRENT_TIMESTAMP AT TIME ZONE 'America/Los_Angeles') - INTERVAL '60 days'
            """, (player_id,))
            last_60 = cur.fetchone()
            
            # Last 60 days map wins/losses
            cur.execute("""
                SELECT 
                    COUNT(*) FILTER (WHERE gs.winner = pg.team_id) as map_wins,
                    COUNT(*) FILTER (WHERE gs.winner IS NOT NULL AND gs.winner != pg.team_id) as map_losses
                FROM esports_player_games pg
                JOIN esports_game_scores gs ON pg.game_id = gs.id
                JOIN esports_matches m ON pg.match_id = m.id
                WHERE pg.player_id = %s
                    AND m.date >= (CURRENT_TIMESTAMP AT TIME ZONE 'America/Los_Angeles') - INTERVAL '60 days'
            """, (player_id,))
            last_60_wl = cur.fetchone()
            
            # Update player with calculated stats
            # Set last_60 values to None if no games in last 60 days
            has_last_60 = (last_60[0] or 0) > 0
            
            cur.execute("""
                UPDATE esports_players SET
                    -- All-time stats
                    all_time_maps = %s,
                    all_time_map_wins = %s,
                    all_time_map_losses = %s,
                    all_time_rating = %s,
                    all_time_acs = %s,
                    all_time_kills = %s,
                    all_time_deaths = %s,
                    all_time_assists = %s,
                    all_time_avg_kills = %s,
                    all_time_avg_deaths = %s,
                    all_time_avg_assists = %s,
                    all_time_kast = %s,
                    all_time_adr = %s,
                    all_time_hs_percent = %s,
                    all_time_fk = %s,
                    all_time_fd = %s,
                    all_time_avg_fk = %s,
                    all_time_avg_fd = %s,
                    -- Last 60 days stats
                    last_60_maps = %s,
                    last_60_map_wins = %s,
                    last_60_map_losses = %s,
                    last_60_rating = %s,
                    last_60_acs = %s,
                    last_60_kills = %s,
                    last_60_deaths = %s,
                    last_60_assists = %s,
                    last_60_avg_kills = %s,
                    last_60_avg_deaths = %s,
                    last_60_avg_assists = %s,
                    last_60_kast = %s,
                    last_60_adr = %s,
                    last_60_hs_percent = %s,
                    last_60_fk = %s,
                    last_60_fd = %s,
                    last_60_avg_fk = %s,
                    last_60_avg_fd = %s
                WHERE id = %s
            """, (
                # All-time values
                all_time[0] or 0,  # total_maps
                all_time_wl[0] or 0,  # map_wins
                all_time_wl[1] or 0,  # map_losses
                all_time[1],  # rating
                all_time[2],  # acs
                all_time[3] or 0,  # kills
                all_time[4] or 0,  # deaths
                all_time[5] or 0,  # assists
                all_time[6],  # avg_kills
                all_time[7],  # avg_deaths
                all_time[8],  # avg_assists
                all_time[9],  # kast
                all_time[10],  # adr
                all_time[11],  # hs
                all_time[12] or 0,  # fk
                all_time[13] or 0,  # fd
                all_time[14],  # avg_fk
                all_time[15],  # avg_fd
                # Last 60 days values (None if no games)
                last_60[0] if has_last_60 else None,  # maps
                last_60_wl[0] if has_last_60 else None,  # map_wins
                last_60_wl[1] if has_last_60 else None,  # map_losses
                last_60[1] if has_last_60 else None,  # rating
                last_60[2] if has_last_60 else None,  # acs
                last_60[3] if has_last_60 else None,  # kills
                last_60[4] if has_last_60 else None,  # deaths
                last_60[5] if has_last_60 else None,  # assists
                last_60[6] if has_last_60 else None,  # avg_kills
                last_60[7] if has_last_60 else None,  # avg_deaths
                last_60[8] if has_last_60 else None,  # avg_assists
                last_60[9] if has_last_60 else None,  # kast
                last_60[10] if has_last_60 else None,  # adr
                last_60[11] if has_last_60 else None,  # hs
                last_60[12] if has_last_60 else None,  # fk
                last_60[13] if has_last_60 else None,  # fd
                last_60[14] if has_last_60 else None,  # avg_fk
                last_60[15] if has_last_60 else None,  # avg_fd
                player_id
            ))
    
    def update_all_player_stats(self):
        """Update stats for all players in the database."""
        with self.conn.cursor() as cur:
            cur.execute("SELECT id FROM esports_players")
            player_ids = [row[0] for row in cur.fetchall()]
        
        for player_id in player_ids:
            self.update_player_stats(player_id)
        
        logger.info(f"Updated stats for {len(player_ids)} players")


# ============================================================================
# Main Scraper
# ============================================================================

class VLRScraper:
    """Main scraper that orchestrates fetching and database operations."""
    
    def __init__(self, db_config: Dict[str, Any]):
        self.fetcher = VLRFetcher()
        self.parser = VLRParser()
        self.db = Database(db_config)
        self.processed_teams = set()
        self.processed_players = set()
    
    def setup_database(self, schema_file: str):
        """Initialize database with schema."""
        self.db.connect()
        self.db.execute_schema(schema_file)
    
    def scrape_team(self, team_id: int) -> Optional[Team]:
        """Scrape and save a team if not already processed."""
        if team_id in self.processed_teams:
            return None
        
        if self.db.team_exists(team_id):
            self.processed_teams.add(team_id)
            return None
        
        logger.info(f"Scraping team {team_id}")
        soup = self.fetcher.fetch(f"/team/{team_id}/")
        if not soup:
            return None
        
        team = self.parser.parse_team(soup, team_id)
        if team:
            self.db.upsert_team(team)
            self.processed_teams.add(team_id)
        
        return team
    
    def scrape_player(self, player_id: int, team_id: Optional[int] = None) -> Optional[Player]:
        """Scrape and save a player if not already processed."""
        if player_id in self.processed_players:
            return None
        
        if self.db.player_exists(player_id):
            self.processed_players.add(player_id)
            return None
        
        logger.info(f"Scraping player {player_id}")
        soup = self.fetcher.fetch(f"/player/{player_id}/")
        if not soup:
            return None
        
        player = self.parser.parse_player(soup, player_id)
        if player:
            # Ensure player's current team exists before inserting player
            if player.team_id:
                self.scrape_team(player.team_id)
            
            self.db.upsert_player(player)
            self.processed_players.add(player_id)
        
        return player
    
    def scrape_tournament(self, event_id: int, tier: Optional[str] = None) -> bool:
        """Scrape a complete tournament."""
        logger.info(f"Scraping tournament {event_id}")
        
        # Get tournament info
        soup = self.fetcher.fetch(f"/event/{event_id}/")
        if not soup:
            return False
        
        tournament = self.parser.parse_tournament(soup, event_id, tier)
        if not tournament:
            return False
        
        self.db.upsert_tournament(tournament)
        logger.info(f"Saved tournament: {tournament.name}")
        
        # Get available stages
        stages = self.parser.parse_stages(soup, event_id)
        if not stages:
            # No stages found, treat main page as single stage
            stages = [("playoffs", "Playoffs", f"/event/{event_id}/")]
        
        logger.info(f"Found {len(stages)} stages: {[s[0] for s in stages]}")
        
        # Get placements from each stage
        total_placements = 0
        winning_team_id = None  # Track the 1st place team in playoffs
        for stage_slug, stage_name, stage_path in stages:
            # Check if this is the active/main page we already fetched
            if 'mod-active' in str(soup.select_one(f'.wf-subnav-item[href*="{stage_slug}"]')):
                stage_soup = soup
            else:
                stage_soup = self.fetcher.fetch(stage_path)
                if not stage_soup:
                    logger.warning(f"Failed to fetch stage: {stage_path}")
                    continue
            
            placements = self.parser.parse_placements(stage_soup, event_id, stage_slug)
            for placement, team_id, prize_money in placements:
                # Ensure team exists
                self.scrape_team(team_id)
                self.db.upsert_placement(event_id, placement, team_id, prize_money, stage_slug)
                
                # Track 1st place team in playoffs for title awards later
                if placement == "1" and stage_slug == "playoffs":
                    self.db.add_team_title(team_id, event_id)
                    winning_team_id = team_id
            
            logger.info(f"  Stage '{stage_slug}': {len(placements)} placements")
            total_placements += len(placements)
        
        logger.info(f"Saved {total_placements} total placements across all stages")
        
        # Get matches
        matches_soup = self.fetcher.fetch(f"/event/matches/{event_id}/?series_id=all")
        if matches_soup:
            match_ids = self.parser.parse_matches_list(matches_soup)
            logger.info(f"Found {len(match_ids)} matches")
            
            for match_id in match_ids:
                self.scrape_match(match_id, event_id)
        
        # Award titles to all players who played for the winning team
        if winning_team_id:
            winning_players = self.db.get_placement_players(event_id, winning_team_id, "playoffs")
            for player_id in winning_players:
                self.db.add_player_title(player_id, event_id)
            logger.info(f"Awarded titles to {len(winning_players)} players on the winning team")
        
        # Update stats for all players who played in this tournament
        logger.info("Calculating player stats...")
        self.db.update_all_player_stats()
        
        self.db.commit()
        return True
    
    def scrape_match(self, match_id: int, tournament_id: int):
        """Scrape a complete match."""
        if self.db.match_exists(match_id):
            logger.debug(f"Match {match_id} already exists, skipping")
            return
        
        logger.info(f"Scraping match {match_id}")
        soup = self.fetcher.fetch(f"/{match_id}/")
        if not soup:
            return
        
        match = self.parser.parse_match(soup, match_id, tournament_id)
        if not match:
            return
        
        # Skip matches with invalid team IDs (0 or None indicates TBD/forfeit/unparseable)
        if not match.team_1_id or match.team_1_id == 0:
            logger.warning(f"Match {match_id} has invalid team_1_id ({match.team_1_id}), skipping")
            return
        if not match.team_2_id or match.team_2_id == 0:
            logger.warning(f"Match {match_id} has invalid team_2_id ({match.team_2_id}), skipping")
            return
        
        # Ensure teams exist
        self.scrape_team(match.team_1_id)
        self.scrape_team(match.team_2_id)
        
        self.db.upsert_match(match)
        
        # Update team match wins/losses
        if match.winner and match.team_1_id and match.team_2_id:
            loser_id = match.team_2_id if match.winner == match.team_1_id else match.team_1_id
            self.db.update_team_match_result(match.winner, loser_id)
        
        # Build team tag mapping for vetos
        team_tags = {}
        tag1 = self.db.get_team_tag(match.team_1_id)
        tag2 = self.db.get_team_tag(match.team_2_id)
        if tag1:
            team_tags[tag1.upper()] = match.team_1_id
        if tag2:
            team_tags[tag2.upper()] = match.team_2_id
        
        # Map vetos
        vetos = self.parser.parse_map_veto(soup, match_id, team_tags)
        for veto in vetos:
            self.db.insert_map_veto(veto)
        
        # Get player IDs from match
        player_ids = self.parser.parse_player_ids_from_match(soup)
        for player_id in player_ids:
            self.scrape_player(player_id)
        
        # Get individual game stats
        game_ids = self.parser.parse_game_ids(soup)
        for game_id in game_ids:
            self.scrape_game(match_id, game_id, tournament_id, 
                           match.team_1_id, match.team_2_id, match.date)
        
        self.db.commit()
    
    def scrape_game(self, match_id: int, game_id: int, tournament_id: int,
                    team_1_id: int, team_2_id: int, match_date: Optional[datetime]):
        """Scrape individual game/map stats."""
        logger.debug(f"Scraping game {game_id}")
        
        soup = self.fetcher.fetch(f"/{match_id}/?game={game_id}&tab=overview")
        if not soup:
            return
        
        game_score, player_stats = self.parser.parse_game_stats(
            soup, match_id, game_id, tournament_id, team_1_id, team_2_id
        )
        
        if game_score:
            self.db.upsert_game_score(game_score)
        
        # Group players by team for roster creation
        team_1_players = [s.player_id for s in player_stats if s.team_id == team_1_id]
        team_2_players = [s.player_id for s in player_stats if s.team_id == team_2_id]
        
        match_d = match_date.date() if match_date else None
        
        roster_1_id = None
        roster_2_id = None
        
        # Create rosters for both teams if we have players
        if team_1_players:
            roster_1_id = self.db.get_or_create_roster(team_1_id, team_1_players, match_d)
            if roster_1_id:
                self.db.update_team_current_roster(team_1_id, roster_1_id, match_d)
        
        if team_2_players:
            roster_2_id = self.db.get_or_create_roster(team_2_id, team_2_players, match_d)
            if roster_2_id:
                self.db.update_team_current_roster(team_2_id, roster_2_id, match_d)
        
        # Log warning if a roster couldn't be created
        if not roster_1_id:
            logger.warning(f"No roster created for team {team_1_id} in game {game_id}")
        if not roster_2_id:
            logger.warning(f"No roster created for team {team_2_id} in game {game_id}")
        
        # Update roster map wins/losses based on game score
        if game_score and roster_1_id and roster_2_id:
            if game_score.team_1_score > game_score.team_2_score:
                self.db.update_roster_map_result(roster_1_id, roster_2_id)
            elif game_score.team_2_score > game_score.team_1_score:
                self.db.update_roster_map_result(roster_2_id, roster_1_id)
        
        # Insert player stats with roster IDs
        for stats in player_stats:
            # Skip if player doesn't exist in database (e.g., deleted player page)
            if not self.db.player_exists(stats.player_id):
                logger.warning(f"Player {stats.player_id} not in database, skipping stats")
                continue
            
            if stats.team_id == team_1_id:
                stats.roster_id = roster_1_id
                stats.opponent_roster_id = roster_2_id
                stats.opponent_team_id = team_2_id
            else:
                stats.roster_id = roster_2_id
                stats.opponent_roster_id = roster_1_id
                stats.opponent_team_id = team_1_id
            
            self.db.insert_player_game_stats(stats)
            
            # Add player to their team's placement(s) in this tournament
            # Try both playoffs and group-stage since a player might participate in both
            for stage in ["playoffs", "group-stage"]:
                self.db.add_player_to_placement(tournament_id, stats.team_id, stats.player_id, stage)
    
    def scrape_events_by_tier(self, tier: int, limit: Optional[int] = None):
        """Scrape all completed events for a given tier, iterating through all pages."""
        # Map tier ID to tier name
        tier_names = {60: "VCT", 61: "VCL", 67: "Offseason"}
        tier_name = tier_names.get(tier)
        
        logger.info(f"Fetching events with tier={tier} ({tier_name})")
        
        # Fetch first page to get pagination info
        soup = self.fetcher.fetch(f"/events/?tier={tier}")
        if not soup:
            return
        
        max_page = self.parser.parse_max_page(soup)
        logger.info(f"Found {max_page} pages of events")
        
        all_events = []
        
        # Collect events from all pages
        for page in range(1, max_page + 1):
            if page > 1:
                soup = self.fetcher.fetch(f"/events/?tier={tier}&page={page}")
                if not soup:
                    continue
            
            events = self.parser.parse_completed_events(soup, tier)
            logger.info(f"Page {page}: Found {len(events)} completed events")
            all_events.extend(events)
            
            # Check limit early to avoid unnecessary page fetches
            if limit and len(all_events) >= limit:
                break
        
        logger.info(f"Total: Found {len(all_events)} completed events across all pages")
        
        if limit:
            all_events = all_events[:limit]
        
        for event_id, event_name in all_events:
            if self.db.tournament_exists(event_id):
                logger.info(f"Tournament {event_id} already exists, skipping")
                continue
            
            logger.info(f"Processing: {event_name} (ID: {event_id})")
            self.scrape_tournament(event_id, tier_name)
    
    def close(self):
        """Clean up resources."""
        self.db.close()


# ============================================================================
# Entry Point
# ============================================================================

def main():
    """Main entry point."""
    import argparse
    
    # Tier definitions
    TIERS = {
        60: "VCT",
        61: "VCL", 
        67: "Offseason"
    }
    
    parser = argparse.ArgumentParser(description='Scrape VLR.gg esports data')
    parser.add_argument('--setup-db', action='store_true', help='Initialize database schema')
    parser.add_argument('--tournament', type=int, help='Scrape a specific tournament by ID')
    parser.add_argument('--tier', type=int, choices=[60, 61, 67], help='Scrape events by tier (60=VCT, 61=VCL, 67=Offseason)')
    parser.add_argument('--all-tiers', action='store_true', help='Scrape all tiers (VCT, VCL, Offseason)')
    parser.add_argument('--limit', type=int, help='Limit number of events to scrape per tier')
    parser.add_argument('--db-host', default=os.getenv('POSTGRES_HOST', 'localhost'), help='Database host')
    parser.add_argument('--db-name', default=os.getenv('POSTGRES_DB', 'cloud9'), help='Database name')
    parser.add_argument('--db-user', default=os.getenv('POSTGRES_USER', 'postgres'), help='Database user')
    parser.add_argument('--db-pass', default=os.getenv('POSTGRES_PASSWORD', 'postgres'), help='Database password')
    
    args = parser.parse_args()
    
    # Update DB config
    DB_CONFIG['host'] = args.db_host
    DB_CONFIG['database'] = args.db_name
    DB_CONFIG['user'] = args.db_user
    DB_CONFIG['password'] = args.db_pass
    
    scraper = VLRScraper(DB_CONFIG)
    
    try:
        if args.setup_db:
            schema_path = os.path.join(os.path.dirname(__file__), 'schema.sql')
            scraper.setup_database(schema_path)
            logger.info("Database schema created")
        else:
            scraper.db.connect()
        
        if args.tournament:
            scraper.scrape_tournament(args.tournament)
        elif args.all_tiers:
            # Scrape all three tiers: VCT, VCL, Offseason
            for tier_id, tier_name in TIERS.items():
                logger.info(f"=" * 60)
                logger.info(f"Starting {tier_name} tier (tier={tier_id})")
                logger.info(f"=" * 60)
                scraper.scrape_events_by_tier(tier_id, args.limit)
            logger.info("=" * 60)
            logger.info("Completed scraping all tiers!")
        elif args.tier:
            scraper.scrape_events_by_tier(args.tier, args.limit)
        elif not args.setup_db:
            # Default: scrape VCT tier 60
            scraper.scrape_events_by_tier(60, limit=1)
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")
        scraper.db.rollback()
        raise
    finally:
        scraper.close()


if __name__ == '__main__':
    main()
