"""
VLR.gg CRON Scraper

Handles two cases:
1. Ongoing events: scrapes any newly-completed matches and updates player stats.
   Placements are NOT scraped (tournament isn't over yet).
2. Newly completed events: tournaments that were 'ongoing' in our DB but now show
   as 'completed' on VLR. Scrapes remaining matches, placements, and awards titles.

Designed to run periodically (e.g. every hour) via CRON / Task Scheduler.

Usage:
    python vlr_cron_scraper.py                  # all three tiers
    python vlr_cron_scraper.py --tier 60        # VCT only
    python vlr_cron_scraper.py --tier 61        # VCL only
"""

import os
import re
import sys
import logging
import argparse
from typing import List, Tuple, Optional, Dict

# Make sure the parent directory is on the path so we can import vlr_scraper
sys.path.insert(0, os.path.dirname(__file__))

from bs4 import BeautifulSoup
from vlr_scraper import VLRScraper, VLRParser, Database, DB_CONFIG

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Extended Parser
# ============================================================================

class CronParser(VLRParser):
    """Extends VLRParser with event-status-aware parsing for the CRON workflow."""

    @staticmethod
    def parse_events_with_status(soup: BeautifulSoup) -> List[Tuple[int, str, str]]:
        """Parse events list page and return ALL non-upcoming events.

        Returns a list of (event_id, name, vlr_status) where vlr_status is
        either 'ongoing' or 'completed'. Upcoming events are intentionally
        excluded — we only care about events that are currently running or
        that have just finished.
        """
        events = []
        for item in soup.select('a.event-item'):
            href = item.get('href', '')
            status_elem = item.select_one('.event-item-desc-item-status')
            if not status_elem:
                continue

            status_text = status_elem.get_text(strip=True).lower()
            if status_text not in ('ongoing', 'completed'):
                continue  # skip upcoming

            match = re.search(r'/event/(\d+)/', href)
            if not match:
                continue

            event_id = int(match.group(1))
            title_elem = item.select_one('.event-item-title')
            name = title_elem.get_text(strip=True) if title_elem else f"Event {event_id}"
            events.append((event_id, name, status_text))

        return events


# ============================================================================
# CRON Scraper
# ============================================================================

class CronScraper(VLRScraper):
    """Extends VLRScraper with CRON-specific logic for ongoing tournaments."""

    def __init__(self, db_config: dict):
        super().__init__(db_config)
        # Override parent's VLRParser instance with the extended one
        self.parser = CronParser()

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    def get_db_ongoing_tournaments(self) -> Dict[int, Optional[str]]:
        """Return {tournament_id: tier} for all tournaments stored as 'ongoing'."""
        with self.db.conn.cursor() as cur:
            cur.execute("SELECT id, tier FROM esports_tournaments WHERE status = 'ongoing'")
            return {row[0]: row[1] for row in cur.fetchall()}

    # ------------------------------------------------------------------
    # Core scrape routines
    # ------------------------------------------------------------------

    def scrape_ongoing_tournament(self, event_id: int, tier: Optional[str]):
        """Upsert tournament metadata and scrape any newly-completed matches.

        Placements are intentionally skipped — the tournament is still running.
        """
        logger.info(f"[ONGOING] Tournament {event_id} ({tier})")

        soup = self.fetcher.fetch(f"/event/{event_id}/")
        if not soup:
            logger.warning(f"  Could not fetch event page for {event_id}")
            return

        tournament = self.parser.parse_tournament(soup, event_id, tier, status_override='ongoing')
        if not tournament:
            return

        self.db.upsert_tournament(tournament)

        matches_soup = self.fetcher.fetch(f"/event/matches/{event_id}/?series_id=all")
        if not matches_soup:
            return

        match_ids = self.parser.parse_matches_list(matches_soup)
        new_ids = [mid for mid in match_ids if not self.db.match_exists(mid)]
        logger.info(f"  {len(match_ids)} completed matches on VLR, {len(new_ids)} not yet in DB")

        for match_id in new_ids:
            self.scrape_match(match_id, event_id)

        if new_ids:
            logger.info(f"  Updating player stats...")
            self.db.update_all_player_stats()

        self.db.commit()

    def finish_completed_tournament(self, event_id: int, tier: Optional[str]):
        """Called when a tournament transitions ongoing → completed.

        Scrapes any remaining matches, all stage placements, awards titles,
        and marks the tournament as completed.
        """
        logger.info(f"[COMPLETED] Tournament {event_id} ({tier}) transitioned to completed")

        soup = self.fetcher.fetch(f"/event/{event_id}/")
        if not soup:
            logger.warning(f"  Could not fetch event page for {event_id}")
            return

        tournament = self.parser.parse_tournament(soup, event_id, tier, status_override='completed')
        if not tournament:
            return

        self.db.upsert_tournament(tournament)

        # --- Placements (all stages) ---
        stages = self.parser.parse_stages(soup, event_id)
        if not stages:
            stages = [("playoffs", "Playoffs", f"/event/{event_id}/")]

        logger.info(f"  Found {len(stages)} stages: {[s[0] for s in stages]}")

        winning_team_id = None
        total_placements = 0

        for stage_slug, stage_name, stage_path in stages:
            # Reuse already-fetched soup if this stage is the active/main page
            active_link = soup.select_one(f'.wf-subnav-item[href*="{stage_slug}"]')
            if active_link and 'mod-active' in active_link.get('class', []):
                stage_soup = soup
            else:
                stage_soup = self.fetcher.fetch(stage_path)
                if not stage_soup:
                    logger.warning(f"  Failed to fetch stage: {stage_path}")
                    continue

            placements = self.parser.parse_placements(stage_soup, event_id, stage_slug)
            for placement, team_id, prize_money in placements:
                self.scrape_team(team_id)
                self.db.upsert_placement(event_id, placement, team_id, prize_money, stage_slug)
                if placement == "1" and stage_slug == "playoffs":
                    self.db.add_team_title(team_id, event_id)
                    winning_team_id = team_id

            logger.info(f"  Stage '{stage_slug}': {len(placements)} placements")
            total_placements += len(placements)

        logger.info(f"  Saved {total_placements} total placements")

        # --- Remaining matches ---
        matches_soup = self.fetcher.fetch(f"/event/matches/{event_id}/?series_id=all")
        if matches_soup:
            match_ids = self.parser.parse_matches_list(matches_soup)
            new_ids = [mid for mid in match_ids if not self.db.match_exists(mid)]
            logger.info(f"  {len(new_ids)} remaining matches to scrape")
            for match_id in match_ids:
                self.scrape_match(match_id, event_id)

        # --- Award titles to winning team's players ---
        if winning_team_id:
            winning_players = self.db.get_placement_players(event_id, winning_team_id, "playoffs")
            for player_id in winning_players:
                self.db.add_player_title(player_id, event_id)
            logger.info(f"  Awarded titles to {len(winning_players)} players on the winning team")

        logger.info(f"  Updating player stats...")
        self.db.update_all_player_stats()
        self.db.commit()

    # ------------------------------------------------------------------
    # Main CRON entrypoint
    # ------------------------------------------------------------------

    def run(self, tiers: List[int]):
        """Scan VLR for ongoing/recently-completed events and process them."""
        tier_names = {60: "VCT", 61: "VCL", 67: "Offseason"}

        # Tournaments currently stored as 'ongoing' in our DB: {id: tier_name}
        db_ongoing: Dict[int, Optional[str]] = self.get_db_ongoing_tournaments()
        logger.info(f"DB has {len(db_ongoing)} tournaments with status='ongoing'")

        vlr_seen: set = set()        # all event IDs encountered on VLR this run

        # Categorised results
        newly_completed: List[Tuple[int, Optional[str]]] = []
        still_ongoing:   List[Tuple[int, Optional[str]]] = []
        new_ongoing:     List[Tuple[int, str, Optional[str]]] = []  # not in DB yet

        for tier in tiers:
            tier_name = tier_names.get(tier)
            soup = self.fetcher.fetch(f"/events/?tier={tier}")
            if not soup:
                continue

            max_page = self.parser.parse_max_page(soup)

            for page in range(1, max_page + 1):
                if page > 1:
                    soup = self.fetcher.fetch(f"/events/?tier={tier}&page={page}")
                    if not soup:
                        continue

                events = self.parser.parse_events_with_status(soup)

                for event_id, name, vlr_status in events:
                    vlr_seen.add(event_id)

                    if vlr_status == 'completed' and event_id in db_ongoing:
                        # Was ongoing in our DB, now completed on VLR → finish it
                        newly_completed.append((event_id, db_ongoing[event_id] or tier_name))

                    elif vlr_status == 'ongoing':
                        if event_id in db_ongoing:
                            # Already tracking this one
                            still_ongoing.append((event_id, db_ongoing[event_id] or tier_name))
                        elif not self.db.tournament_exists(event_id):
                            # Brand new ongoing tournament we haven't seen before
                            new_ongoing.append((event_id, name, tier_name))
                        # If it exists in DB but not as ongoing, it was already fully
                        # completed — nothing to do.

        # Warn about DB-ongoing tournaments not visible on VLR at all (edge case:
        # very old bad-status records that the SQL fix missed, or off-tier events)
        unseen = [eid for eid in db_ongoing if eid not in vlr_seen]
        if unseen:
            logger.warning(
                f"{len(unseen)} DB-ongoing tournaments were not found on any VLR "
                f"events page this run. They may be from a previous bad-status scrape. "
                f"IDs: {unseen[:20]}"
            )

        logger.info(
            f"Plan — newly completed: {len(newly_completed)}, "
            f"still ongoing: {len(still_ongoing)}, "
            f"new ongoing: {len(new_ongoing)}"
        )

        # Process newly completed first (highest priority)
        for event_id, tier_name in newly_completed:
            try:
                self.finish_completed_tournament(event_id, tier_name)
            except Exception as e:
                logger.error(f"Error finishing completed tournament {event_id}: {e}")
                self.db.rollback()

        # Process ongoing tournaments (update matches + stats, no placements)
        for event_id, tier_name in still_ongoing:
            try:
                self.scrape_ongoing_tournament(event_id, tier_name)
            except Exception as e:
                logger.error(f"Error scraping ongoing tournament {event_id}: {e}")
                self.db.rollback()

        for event_id, name, tier_name in new_ongoing:
            logger.info(f"[NEW] Discovered ongoing tournament: {name} (ID: {event_id})")
            try:
                self.scrape_ongoing_tournament(event_id, tier_name)
            except Exception as e:
                logger.error(f"Error scraping new ongoing tournament {event_id}: {e}")
                self.db.rollback()

        logger.info("CRON run complete.")


# ============================================================================
# Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='CRON scraper for ongoing and recently-completed VLR events'
    )
    parser.add_argument(
        '--tier', type=int, choices=[60, 61, 67],
        help='Process a single tier (60=VCT, 61=VCL, 67=Offseason). Defaults to all tiers.'
    )
    parser.add_argument('--db-host', default=os.getenv('POSTGRES_HOST', 'localhost'))
    parser.add_argument('--db-name', default=os.getenv('POSTGRES_DB', 'cloud9'))
    parser.add_argument('--db-user', default=os.getenv('POSTGRES_USER', 'postgres'))
    parser.add_argument('--db-pass', default=os.getenv('POSTGRES_PASSWORD', 'postgres'))

    args = parser.parse_args()

    DB_CONFIG['host'] = args.db_host
    DB_CONFIG['database'] = args.db_name
    DB_CONFIG['user'] = args.db_user
    DB_CONFIG['password'] = args.db_pass

    tiers = [args.tier] if args.tier else [60, 61, 67]

    scraper = CronScraper(DB_CONFIG)
    try:
        scraper.db.connect()
        scraper.run(tiers)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        scraper.db.rollback()
        raise
    finally:
        scraper.close()


if __name__ == '__main__':
    main()
