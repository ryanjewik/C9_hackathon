"""
VALORANT VOD Timeline Processor - Main Application
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from contextlib import asynccontextmanager
import os
import uuid
from datetime import datetime
from typing import Optional

from app.schemas import (
    JobStatus,
    JobResponse,
    TimelineResponse,
    EventsResponse,
    RoundEventsResponse,
    HealthResponse,
)
from app.services.job_manager import JobManager
from app.services.vod_processor import VODProcessor
from config import get_settings

settings = get_settings()

# Initialize services
job_manager = JobManager()
vod_processor = VODProcessor()
vod_processor.set_job_manager(job_manager)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - startup and shutdown."""
    # Startup
    os.makedirs(settings.upload_dir, exist_ok=True)
    os.makedirs(settings.output_dir, exist_ok=True)
    print(f"VOD Processor starting up...")
    print(f"Upload directory: {settings.upload_dir}")
    print(f"Output directory: {settings.output_dir}")
    yield
    # Shutdown
    print("VOD Processor shutting down...")


app = FastAPI(
    title="VALORANT VOD Timeline Processor",
    description="Process VALORANT VCT Match VODs and extract structured timeline data",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0"
    )


@app.post("/api/v1/vod/upload", response_model=JobResponse)
async def upload_vod(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    match_players: Optional[str] = Form(None),
    left_team: Optional[str] = Form(None),
    right_team: Optional[str] = Form(None),
    map_name: Optional[str] = Form(None),
):
    """
    Upload a VOD file for processing.
    
    - **file**: The VOD file (mp4, mkv, etc.)
    - **match_players**: Optional comma-separated list of player names expected in the match
    - **left_team**: Team code/name for left side of HUD (e.g., "NRG")
    - **right_team**: Team code/name for right side of HUD (e.g., "FNC")
    - **map_name**: Map name (e.g., "Abyss")
    
    Returns a job_id to track the processing status.
    """
    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    valid_extensions = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in valid_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Supported: {', '.join(valid_extensions)}"
        )
    
    # Check file size (rough estimate from content-length header if available)
    max_size_bytes = settings.max_upload_size_mb * 1024 * 1024
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Save uploaded file
    upload_path = os.path.join(settings.upload_dir, f"{job_id}{ext}")
    
    try:
        # Stream file to disk
        with open(upload_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                buffer.write(chunk)
                if os.path.getsize(upload_path) > max_size_bytes:
                    os.remove(upload_path)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Maximum size: {settings.max_upload_size_mb}MB"
                    )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    # Create job
    job = job_manager.create_job(
        job_id=job_id,
        filename=file.filename,
        upload_path=upload_path,
        match_players=[match_players] if match_players else None  # Pass as single-item list to preserve format
    )
    
    # Queue processing task
    background_tasks.add_task(
        vod_processor.process_vod,
        job_id=job_id,
        video_path=upload_path,
        output_dir=settings.output_dir,
        match_players=[match_players] if match_players else None,  # Pass as single-item list to preserve format
        left_team=left_team,
        right_team=right_team,
        map_name=map_name,
    )
    
    return JobResponse(
        job_id=job_id,
        status=job.status,
        message="VOD uploaded successfully. Processing started.",
        created_at=job.created_at
    )


@app.get("/api/v1/vod/{job_id}/status", response_model=JobResponse)
async def get_job_status(job_id: str):
    """Get the status of a processing job."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobResponse(
        job_id=job_id,
        status=job.status,
        message=job.message,
        created_at=job.created_at,
        completed_at=job.completed_at,
        progress=job.progress,
        error=job.error
    )


@app.get("/api/v1/vod/{job_id}/timeline", response_model=TimelineResponse)
async def get_timeline(job_id: str):
    """Get the generated timeline for a completed job."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed. Current status: {job.status}"
        )
    
    # Load timeline from output file
    timeline_path = os.path.join(settings.output_dir, f"{job_id}_timeline.json")
    if not os.path.exists(timeline_path):
        raise HTTPException(status_code=404, detail="Timeline file not found")
    
    import json
    with open(timeline_path, "r") as f:
        timeline_data = json.load(f)
    
    return TimelineResponse(**timeline_data)


@app.get("/api/v1/vod/{job_id}/events", response_model=EventsResponse)
async def get_events(
    job_id: str,
    event_type: Optional[str] = None,
    start_time_ms: Optional[int] = None,
    end_time_ms: Optional[int] = None,
    limit: int = 1000,
):
    """
    Get all events extracted from a VOD.
    
    - **event_type**: Filter by event type (KILL_EVENT, DEATH_EVENT, etc.)
    - **start_time_ms**: Filter events after this timestamp
    - **end_time_ms**: Filter events before this timestamp
    - **limit**: Maximum number of events to return
    """
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed. Current status: {job.status}"
        )
    
    events_path = os.path.join(settings.output_dir, f"{job_id}_events.json")
    if not os.path.exists(events_path):
        raise HTTPException(status_code=404, detail="Events file not found")
    
    import json
    with open(events_path, "r") as f:
        all_events = json.load(f)
    
    # Apply filters
    filtered = all_events
    if event_type:
        filtered = [e for e in filtered if e.get("type") == event_type]
    if start_time_ms is not None:
        filtered = [e for e in filtered if e.get("t_ms", 0) >= start_time_ms]
    if end_time_ms is not None:
        filtered = [e for e in filtered if e.get("t_ms", 0) <= end_time_ms]
    
    return EventsResponse(
        job_id=job_id,
        total_events=len(all_events),
        filtered_events=len(filtered),
        events=filtered[:limit]
    )


@app.get("/api/v1/vod/{job_id}/round/{round_number}", response_model=RoundEventsResponse)
async def get_round_events(job_id: str, round_number: int):
    """Get events for a specific round."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed. Current status: {job.status}"
        )
    
    timeline_path = os.path.join(settings.output_dir, f"{job_id}_timeline.json")
    if not os.path.exists(timeline_path):
        raise HTTPException(status_code=404, detail="Timeline file not found")
    
    import json
    with open(timeline_path, "r") as f:
        timeline_data = json.load(f)
    
    rounds = timeline_data.get("rounds", [])
    round_data = next((r for r in rounds if r.get("round_number") == round_number), None)
    
    if not round_data:
        raise HTTPException(status_code=404, detail=f"Round {round_number} not found")
    
    return RoundEventsResponse(
        job_id=job_id,
        round_number=round_number,
        **round_data
    )


@app.get("/api/v1/vod/{job_id}/download/{file_type}")
async def download_output(job_id: str, file_type: str):
    """
    Download output files.
    
    - **file_type**: 'timeline', 'events', 'summary', or 'debug_video'
    """
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    file_map = {
        "timeline": f"{job_id}_timeline.json",
        "events": f"{job_id}_events.json",
        "summary": f"{job_id}_summary.json",
        "debug_video": f"{job_id}_debug.mp4",
    }
    
    if file_type not in file_map:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Options: {', '.join(file_map.keys())}"
        )
    
    file_path = os.path.join(settings.output_dir, file_map[file_type])
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {file_type}")
    
    return FileResponse(
        file_path,
        filename=file_map[file_type],
        media_type="application/json" if file_type != "debug_video" else "video/mp4"
    )


@app.delete("/api/v1/vod/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its associated files."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Remove files
    import glob
    for pattern in [f"{job_id}*"]:
        for f in glob.glob(os.path.join(settings.upload_dir, pattern)):
            os.remove(f)
        for f in glob.glob(os.path.join(settings.output_dir, pattern)):
            os.remove(f)
    
    job_manager.delete_job(job_id)
    
    return {"message": f"Job {job_id} deleted successfully"}
