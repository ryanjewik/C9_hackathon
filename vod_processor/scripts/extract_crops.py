"""
Extract weapon icon crops from VOD files (no stat comparison).

Usage:
    python extract_crops.py                     # Process VODs 4-7
    python extract_crops.py --vods 4 5          # Process specific VODs
    python extract_crops.py --vods 1 2 3 4 5 6 7  # All VODs

    # Strict roster mode (match only against these 10 players):
    python extract_crops.py --vods 8 \\
        --left-players keznit Rossy P0PPIN Eggsterr Inspire \\
        --right-players penny Demon1 Zellsis v1c Xeppaa

Crops are saved to /app/outputs/crops/ with filenames encoding the VOD
number and timestamp: vod{N}_crop_{NNNNN}_t{ms}ms.png
"""
import json
import os
import sys
import argparse
from datetime import datetime

sys.path.insert(0, '/app')

from app.services.processing.vod_processor import VODProcessor


def extract_crops_from_vod(vod_number: int, output_dir: str, crops_dir: str,
                           left_players: list = None, right_players: list = None):
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
    if left_players and right_players:
        print(f"  STRICT ROSTER: left={left_players}, right={right_players}")
    print("=" * 70)

    job_id = f"crops-vod{vod_number}-{datetime.now().strftime('%H%M%S')}"

    processor = VODProcessor()

    # Determine if strict roster mode
    strict = bool(left_players and right_players)

    # Run processing with auto-detect (no team/player info passed)
    result = processor.process_vod(
        job_id=job_id,
        video_path=video_path,
        output_dir=output_dir,
        left_team=None,
        right_team=None,
        left_player_pool=left_players,
        right_player_pool=right_players,
        strict_roster=strict,
    )

    # Rename crops from generic crop_NNNNN to include VOD number
    # so crops from different VODs don't overwrite each other.
    # Crops are bucketed into method subfolders (threshold/, ocr_bright/, etc.)
    renamed = 0
    for root, dirs, files in os.walk(crops_dir):
        for f in sorted(files):
            if f.startswith("crop_") and f.endswith(".png"):
                old_path = os.path.join(root, f)
                new_name = f"vod{vod_number}_{f}"
                new_path = os.path.join(root, new_name)
                os.rename(old_path, new_path)
                renamed += 1

    # Save ult badge diagnostics to JSON for offline threshold analysis
    if hasattr(processor, '_killfeed_detector') and processor._killfeed_detector:
        det = processor._killfeed_detector
    else:
        det = processor
    ult_diags = getattr(det, '_ult_diagnostics', [])
    if ult_diags:
        diag_path = os.path.join(crops_dir, f"vod{vod_number}_ult_diagnostics.json")
        with open(diag_path, 'w') as f:
            json.dump(ult_diags, f, indent=2)
        print(f"  -> Ult badge diagnostics: {diag_path} ({len(ult_diags)} entries)")

    print(f"\n  -> {renamed} crops saved (prefix: vod{vod_number}_)")
    return renamed


def main():
    parser = argparse.ArgumentParser(description="Extract weapon icon crops from VOD files")
    parser.add_argument(
        '--vods', nargs='+', type=int, default=[4, 5, 6, 7],
        help='VOD numbers to process (default: 4 5 6 7)'
    )
    parser.add_argument(
        '--left-players', nargs='+', type=str, default=None,
        help='Exact player names for the left (teal) team (enables strict roster mode)'
    )
    parser.add_argument(
        '--right-players', nargs='+', type=str, default=None,
        help='Exact player names for the right (orange) team (enables strict roster mode)'
    )
    args = parser.parse_args()

    output_dir = "/app/outputs"
    crops_dir = os.path.join(output_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)

    total_crops = 0
    for vod_num in args.vods:
        count = extract_crops_from_vod(
            vod_num, output_dir, crops_dir,
            left_players=args.left_players,
            right_players=args.right_players,
        )
        total_crops += count
        print(f"\n  Running total: {total_crops} crops\n")

    print("\n" + "=" * 70)
    print(f"DONE — {total_crops} total crops across VODs {args.vods}")
    print(f"Crops directory: {crops_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
