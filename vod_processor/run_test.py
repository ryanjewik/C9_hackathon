#!/usr/bin/env python3
"""Test script to process VOD and output results."""

import sys
import json

sys.path.insert(0, '/app')

from vod_processor.app.services.processing.vod_processor import VODProcessor

def main():
    processor = VODProcessor()
    job_id = 'fresh_test_003'
    
    print('Starting processing...', flush=True)
    result = processor.process_vod(
        job_id, 
        '/app/uploads/match_vod.mp4', 
        '/app/outputs'
    )
    print('Processing complete!', flush=True)
    
    print(f"\n=== RESULTS ===")
    print(f"Total kills: {result.get('total_kills', 'N/A')}")
    print(f"Total rounds: {result.get('total_rounds', 'N/A')}")
    
    # Save result summary
    with open('/app/outputs/fresh_test_003_quick_summary.json', 'w') as f:
        json.dump({
            'total_kills': result.get('total_kills', 0),
            'total_rounds': result.get('total_rounds', 0),
        }, f, indent=2)
    
    print("Quick summary saved to fresh_test_003_quick_summary.json")

if __name__ == '__main__':
    main()
