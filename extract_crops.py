"""
Extract weapon icon crops from VOD files (no stat comparison).

Usage:
    python extract_crops.py                     # Process VODs 4-7
    python extract_crops.py --vods 4 5          # Process specific VODs
    python extract_crops.py --vods 1 2 3 4 5 6 7  # All VODs

Crops are saved to /app/outputs/crops/ with filenames encoding the VOD
number and timestamp: vod{N}_crop_{NNNNN}_t{ms}ms.png
"""
import os
import sys
import argparse
from datetime import datetime

sys.path.insert(0, '/app')

from app.services.processing.vod_processor import VODProcessor


def extract_crops_from_vod(vod_number: int, output_dir: str, crops_dir: str):
    """Run VOD processing on a single file, collecting weapon icon crops."""
    if vod_number == 1:
        video_filename = "match_vod.mp4"
    else:
        video_filename = f"match_vod_{vod_number}.mp4"

    video_path = f"/app/uploads/{video_filename}"

    if not os.path.exists(video_path):
        print(f"SKIP: {video_filename} not found at {video_path}")
        return 0

    print("=" * 70)
    print(f"EXTRACTING CROPS: {video_filename}")
    print("=" * 70)

    job_id = f"crops-vod{vod_number}-{datetime.now().strftime('%H%M%S')}"

    processor = VODProcessor()

    # Run processing with auto-detect (no team/player info passed)
    result = processor.process_vod(
        job_id=job_id,
        video_path=video_path,
        output_dir=output_dir,
        left_team=None,
        right_team=None,
        left_player_pool=None,
        right_player_pool=None,
    )

    # Rename crops from generic crop_NNNNN to include VOD number
    # so crops from different VODs don't overwrite each other.
    crop_files = sorted(f for f in os.listdir(crops_dir) if f.startswith("crop_") and f.endswith(".png"))
    renamed = 0
    for f in crop_files:
        old_path = os.path.join(crops_dir, f)
        new_name = f"vod{vod_number}_{f}"
        new_path = os.path.join(crops_dir, new_name)
        os.rename(old_path, new_path)
        renamed += 1

    print(f"\n  -> {renamed} crops saved (prefix: vod{vod_number}_)")
    return renamed


def main():
    parser = argparse.ArgumentParser(description="Extract weapon icon crops from VOD files")
    parser.add_argument(
        '--vods', nargs='+', type=int, default=[4, 5, 6, 7],
        help='VOD numbers to process (default: 4 5 6 7)'
    )
    args = parser.parse_args()

    output_dir = "/app/outputs"
    crops_dir = os.path.join(output_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)

    total_crops = 0
    for vod_num in args.vods:
        count = extract_crops_from_vod(vod_num, output_dir, crops_dir)
        total_crops += count
        print(f"\n  Running total: {total_crops} crops\n")

    print("\n" + "=" * 70)
    print(f"DONE — {total_crops} total crops across VODs {args.vods}")
    print(f"Crops directory: {crops_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
