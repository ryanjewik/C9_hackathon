"""Quick crop extraction test for VOD 8."""
import sys
sys.path.insert(0, '/app')
from app.services.processing.vod_processor import VODProcessor

vp = VODProcessor()
result = vp.process_vod(
    job_id='crops-vod8-hashfix',
    video_path='/app/data/vods/val_vod8.mp4',
    output_dir='/app/outputs',
)
print('DONE:', result.get('total_kills', 'N/A') if isinstance(result, dict) else result)
