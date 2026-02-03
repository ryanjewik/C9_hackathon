#!/usr/bin/env python3
"""
Example script to process a VOD file directly (without the API).
"""

import sys
import os
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.vod_processor import VODProcessor
from app.services.job_manager import JobManager


def main():
    if len(sys.argv) < 2:
        print("Usage: python process_vod.py <video_path> [output_dir]")
        print("\nExample:")
        print("  python process_vod.py match_vod.mp4 ./output")
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./output"
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate job ID from filename
    job_id = os.path.splitext(os.path.basename(video_path))[0]
    
    print(f"Processing VOD: {video_path}")
    print(f"Output directory: {output_dir}")
    print(f"Job ID: {job_id}")
    print("-" * 50)
    
    # Initialize processor
    processor = VODProcessor()
    job_manager = JobManager()
    processor.set_job_manager(job_manager)
    
    # Create a job
    job_manager.create_job(
        job_id=job_id,
        filename=os.path.basename(video_path),
        upload_path=video_path
    )
    
    # Process the VOD
    result = processor.process_vod(
        job_id=job_id,
        video_path=video_path,
        output_dir=output_dir,
        match_players=None  # Can specify player names to filter
    )
    
    print("-" * 50)
    print("Processing complete!")
    print(f"Status: {result.get('status', 'unknown')}")
    
    if result.get("status") == "completed":
        print(f"Events detected: {result.get('events_count', 0)}")
        print(f"Kills detected: {result.get('kills_count', 0)}")
        print("\nOutput files:")
        for f in result.get("output_files", []):
            print(f"  - {f}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
