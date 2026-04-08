"""
Celery tasks for VLR scraping.
"""

import os
import logging

from scraping_job.celery_app import celery_app
from scraping_job.vlr_cron_scraper import CronScraper
from scraping_job.vlr_scraper import DB_CONFIG

logger = logging.getLogger(__name__)


@celery_app.task(name='tasks.run_vlr_scrape', bind=True, max_retries=3)
def run_vlr_scrape(self, tiers=None):
    """Run the VLR cron scraper for the given list of tier IDs.

    Args:
        tiers: List of VLR tier IDs to scrape. Defaults to [60, 61, 67].
    """
    if tiers is None:
        tiers = [60, 61, 67]

    db_config = {
        **DB_CONFIG,
        'host': os.getenv('POSTGRES_HOST', 'postgres'),
        'database': os.getenv('POSTGRES_DB', 'cloud9'),
        'user': os.getenv('POSTGRES_USER', 'postgres'),
        'password': os.getenv('POSTGRES_PASSWORD', 'postgres'),
        'port': int(os.getenv('POSTGRES_PORT', 5432)),
    }

    logger.info(f"Starting VLR scrape for tiers: {tiers}")
    scraper = CronScraper(db_config)
    try:
        scraper.db.connect()
        scraper.run(tiers)
        logger.info("VLR scrape completed successfully.")
    except Exception as exc:
        logger.error(f"VLR scrape failed: {exc}")
        raise self.retry(exc=exc, countdown=60)
    finally:
        scraper.close()
