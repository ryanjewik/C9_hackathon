"""
Celery Worker for background VOD processing.

Redis is used as the message broker - it stores:
- Pending tasks in a queue
- Task state (started, progress, completed, failed)
- Task results

This enables asynchronous processing where the API returns immediately
and the VOD processing happens in the background.
"""

from celery import Celery
from config import get_settings

settings = get_settings()

# Initialize Celery with Redis as broker
celery_app = Celery(
    "vod_processor",
    broker=settings.redis_url,  # Redis URL for task queue
    backend=settings.redis_url,  # Redis URL for results storage
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=21600,  # 6 hours max per task (long VODs with multi-pass OCR)
    worker_prefetch_multiplier=1,  # Only fetch one task at a time
    worker_concurrency=settings.worker_concurrency,
)


@celery_app.task(bind=True, name="process_vod")
def process_vod_task(
    self,
    job_id: str,
    video_path: str,
    output_dir: str,
    match_id: str = None,
    match_players: list = None,
    map_name: str = None,
    team_a: str = None,
    team_b: str = None,
    use_pipeline: bool = True,
):
    """
    Celery task for processing a VOD.
    
    Args:
        job_id: Unique job identifier
        video_path: Path to the uploaded VOD file
        output_dir: Directory to save output files
        match_id: Optional match ID for database lookup
        match_players: Optional list of player names to filter for
        map_name: Optional map name
        team_a: Team A name
        team_b: Team B name
        use_pipeline: If True, use the full pipeline (recommended)
    """
    from vod_processor.app.services.io.job_manager import JobManager
    
    job_manager = JobManager()
    
    # Update task state
    self.update_state(state="PROCESSING", meta={"job_id": job_id})
    
    if use_pipeline:
        # Use the full architecture-compliant pipeline
        from vod_processor.app.services.processing.pipeline import VODPipeline
        
        # Get database connection string if available
        db_url = settings.database_url if hasattr(settings, 'database_url') else None
        
        processor = VODPipeline(db_connection_string=db_url)
        processor.set_job_manager(job_manager)
        
        result = processor.process(
            job_id=job_id,
            video_path=video_path,
            output_dir=output_dir,
            match_id=match_id,
            match_players=match_players,
            map_name=map_name,
            team_a=team_a,
            team_b=team_b,
        )
    else:
        # Use the basic VOD processor
        from vod_processor.app.services.processing.vod_processor import VODProcessor
        
        processor = VODProcessor()
        processor.set_job_manager(job_manager)
        
        result = processor.process_vod(
            job_id=job_id,
            video_path=video_path,
            output_dir=output_dir,
            match_players=match_players,
        )
    
    return result
