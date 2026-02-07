"""VOD Processor application package.

This module intentionally does not import ``app.main`` to avoid a
circular import when other modules import ``vod_processor.app``. Import
the FastAPI instance directly from ``app.main:app`` where needed.
"""

__all__ = []
