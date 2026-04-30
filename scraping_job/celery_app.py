"""
Celery application for the VLR scraper.

Uses Redis db/1 to avoid conflicts with the VOD processor (which uses db/0).
Beat schedule runs the scraper every hour.
"""

import os
import re

from celery import Celery
from celery.schedules import crontab

# ---------------------------------------------------------------------------
# OpenTelemetry: configure traces + metrics, then instrument Celery
# ---------------------------------------------------------------------------
from opentelemetry import trace, metrics
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.instrumentation.celery import CeleryInstrumentor


def _grpc_endpoint(url: str) -> str:
    """Strip http(s):// scheme — gRPC exporter expects bare host:port."""
    return re.sub(r'^https?://', '', url)


_OTEL_ENDPOINT = os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT', 'http://otel-collector:4317')
_SERVICE_NAME  = os.getenv('OTEL_SERVICE_NAME', 'vlr-scraper')
_grpc_host     = _grpc_endpoint(_OTEL_ENDPOINT)
_resource      = Resource.create({"service.name": _SERVICE_NAME})

# Traces
_tracer_provider = TracerProvider(resource=_resource)
_tracer_provider.add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint=_grpc_host, insecure=True))
)
trace.set_tracer_provider(_tracer_provider)

# Metrics (push every 30 s)
metrics.set_meter_provider(
    MeterProvider(
        resource=_resource,
        metric_readers=[
            PeriodicExportingMetricReader(
                OTLPMetricExporter(endpoint=_grpc_host, insecure=True),
                export_interval_millis=30_000,
            )
        ],
    )
)

CeleryInstrumentor().instrument()
# ---------------------------------------------------------------------------

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
