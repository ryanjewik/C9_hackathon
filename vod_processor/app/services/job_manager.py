"""
Job Manager - Tracks processing jobs in memory.
For production, replace with Redis or database storage.
"""

from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from app.schemas import JobStatus


@dataclass
class Job:
    """Represents a VOD processing job."""
    job_id: str
    filename: str
    upload_path: str
    status: JobStatus = JobStatus.PENDING
    message: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    error: Optional[str] = None
    match_players: Optional[List[str]] = None
    
    # Processing results
    total_frames: int = 0
    processed_frames: int = 0
    events_detected: int = 0
    output_files: List[str] = field(default_factory=list)


class JobManager:
    """
    Manages processing jobs.
    Singleton pattern to share state across requests.
    In-memory storage for simplicity. Use Redis for production.
    """
    
    _instance: Optional['JobManager'] = None
    _jobs: Dict[str, Job] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        # Don't reset _jobs on each init since we're a singleton
        pass
    
    def create_job(
        self,
        job_id: str,
        filename: str,
        upload_path: str,
        match_players: Optional[List[str]] = None,
    ) -> Job:
        """Create a new job."""
        job = Job(
            job_id=job_id,
            filename=filename,
            upload_path=upload_path,
            match_players=match_players,
        )
        self._jobs[job_id] = job
        return job
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by ID."""
        return self._jobs.get(job_id)
    
    def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        message: str = "",
        progress: Optional[float] = None,
        error: Optional[str] = None,
    ) -> Optional[Job]:
        """Update job status."""
        job = self._jobs.get(job_id)
        if not job:
            return None
        
        job.status = status
        job.message = message
        
        if progress is not None:
            job.progress = progress
        
        if error:
            job.error = error
        
        if status == JobStatus.PROCESSING and job.started_at is None:
            job.started_at = datetime.utcnow()
        
        if status in (JobStatus.COMPLETED, JobStatus.FAILED):
            job.completed_at = datetime.utcnow()
        
        return job
    
    def update_progress(
        self,
        job_id: str,
        processed_frames: int,
        total_frames: int,
        events_detected: int = 0,
    ) -> Optional[Job]:
        """Update job progress."""
        job = self._jobs.get(job_id)
        if not job:
            return None
        
        job.processed_frames = processed_frames
        job.total_frames = total_frames
        job.events_detected = events_detected
        job.progress = (processed_frames / total_frames * 100) if total_frames > 0 else 0
        
        return job
    
    def add_output_file(self, job_id: str, file_path: str) -> None:
        """Add an output file to the job."""
        job = self._jobs.get(job_id)
        if job:
            job.output_files.append(file_path)
    
    def delete_job(self, job_id: str) -> bool:
        """Delete a job."""
        if job_id in self._jobs:
            del self._jobs[job_id]
            return True
        return False
    
    def list_jobs(self, status: Optional[JobStatus] = None) -> List[Job]:
        """List all jobs, optionally filtered by status."""
        jobs = list(self._jobs.values())
        if status:
            jobs = [j for j in jobs if j.status == status]
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)
