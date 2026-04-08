"""
Celery application for the VLR scraper.

Uses Redis db/1 to avoid conflicts with the VOD processor (which uses db/0).
Beat schedule runs the scraper every hour.
"""

import os

from celery import Celery
from celery.schedules import crontab

REDIS_URL = os.getenv('REDIS_URL', 'redis://redis:6379/1')

celery_app = Celery(
    'vlr_scraper',
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=['tasks'],
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    # Beat schedule: run all tiers every hour on the hour
    beat_schedule={
        'scrape-vlr-hourly': {
            'task': 'tasks.run_vlr_scrape',
            'schedule': crontab(minute=0),
            'args': ([60, 61, 67],),
        },
    },
)
