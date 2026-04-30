"""
Celery tasks for VLR scraping.
"""

import os
import time
import logging

from opentelemetry import trace, metrics

from celery_app import celery_app
from vlr_cron_scraper import CronScraper
from vlr_scraper import DB_CONFIG

logger = logging.getLogger(__name__)

_tracer = trace.get_tracer(__name__)
_meter  = metrics.get_meter(__name__)

_scrape_runs = _meter.create_counter(
    "vlr_scrape_runs_total",
    description="Total VLR scrape task invocations, labelled by status (success|failure)",
)
_scrape_duration = _meter.create_histogram(
    "vlr_scrape_duration_seconds",
    description="End-to-end wall-clock duration of a VLR scrape run",
    unit="s",
)


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

    logger.info("Starting VLR scrape for tiers: %s", tiers)
    scraper = CronScraper(db_config)
    t0 = time.monotonic()

    with _tracer.start_as_current_span(
        "vlr_scrape",
        attributes={"tiers": str(tiers)},
    ) as span:
        try:
            scraper.db.connect()
            scraper.run(tiers)
            _scrape_runs.add(1, {"status": "success"})
            logger.info("VLR scrape completed successfully.")
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(trace.StatusCode.ERROR, str(exc))
            _scrape_runs.add(1, {"status": "failure"})
            logger.error("VLR scrape failed: %s", exc)
            raise self.retry(exc=exc, countdown=60)
        finally:
            _scrape_duration.record(time.monotonic() - t0)
            scraper.close()
