"""
Run VOD processing tests on all 3 match VODs and compare to expected stats.
Uses AUTO-DETECT mode - no team/player information is passed to the processor.
"""
import os
import sys
import json
from datetime import datetime

# Add app to path
sys.path.insert(0, '/app')

from app.services.vod_processor import VODProcessor


# Expected stats from screenshots (K/D for each player)
# VOD 1: NRG vs FNC
# VOD 2: TH vs PRX  
# VOD 3: DRX vs TL
EXPECTED_STATS = {
    "match_vod.mp4": {
        "teams": ("NRG", "FNC"),
        "players": {
            # NRG
            "skuba": {"kills": 22, "deaths": 15},
            "brawk": {"kills": 20, "deaths": 17},
            "Ethan": {"kills": 17, "deaths": 16},
            "mada": {"kills": 15, "deaths": 18},
            "s0m": {"kills": 13, "deaths": 22},
            # FNC
            "crashies": {"kills": 22, "deaths": 15},
            "kaajak": {"kills": 24, "deaths": 19},
            "Chronicle": {"kills": 17, "deaths": 19},
            "Alfajer": {"kills": 14, "deaths": 17},
            "Boaster": {"kills": 13, "deaths": 17},
        },
        "total_kills": 177,  # Sum of all kills (87 NRG + 90 FNC)
    },
    "match_vod_2.mp4": {
        "teams": ("TH", "PRX"),
        "players": {
            # PRX (right side in screenshot)
            "PatMen": {"kills": 15, "deaths": 8},
            "f0rsakeN": {"kills": 14, "deaths": 10},
            "d4v41": {"kills": 17, "deaths": 10},
            "something": {"kills": 13, "deaths": 11},
            "Jinggg": {"kills": 12, "deaths": 13},
            # TH (left side in screenshot)
            "benjyfishy": {"kills": 11, "deaths": 14},
            "RieNs": {"kills": 14, "deaths": 14},
            "Wo0t": {"kills": 8, "deaths": 12},
            "Boo": {"kills": 9, "deaths": 15},
            "MiniBoo": {"kills": 10, "deaths": 16},
        },
        "total_kills": 120,  # Sum of all kills (71 PRX + 49 TH)
    },
    "match_vod_3.mp4": {
        "teams": ("DRX", "TL"),
        "players": {
            # DRX
            "HYUNMIN": {"kills": 24, "deaths": 14},
            "Flashback": {"kills": 18, "deaths": 12},
            "MaKo": {"kills": 11, "deaths": 11},
            "free1ng": {"kills": 13, "deaths": 13},
            "BeYN": {"kills": 10, "deaths": 11},
            # TL
            "keiko": {"kills": 18, "deaths": 15},
            "kamo": {"kills": 16, "deaths": 14},
            "trexx": {"kills": 12, "deaths": 15},
            "nAts": {"kills": 11, "deaths": 17},
            "paTiTek": {"kills": 4, "deaths": 15},
        },
        "total_kills": 137,  # Sum of all kills (76 DRX + 61 TL)
    },
}


def run_single_vod(video_filename: str, output_dir: str):
    """Run VOD processing on a single file using auto-detect mode."""
    video_path = f"/app/{video_filename}"
    
    if not os.path.exists(video_path):
        print(f"ERROR: Video file not found: {video_path}")
        return None
    
    expected = EXPECTED_STATS.get(video_filename, {})
    expected_teams = expected.get("teams", ("Unknown", "Unknown"))
    
    print("="*70)
    print(f"PROCESSING: {video_filename}")
    print(f"Expected teams (for reference): {expected_teams[0]} vs {expected_teams[1]}")
    print("Using AUTO-DETECT mode - no team/player info passed")
    print("="*70)
    
    # Generate job ID
    job_id = f"test-{video_filename.replace('.mp4', '').replace('_', '-')}-{datetime.now().strftime('%H%M%S')}"
    
    # Initialize processor
    processor = VODProcessor()
    
    # Run processing with AUTO-DETECT - no team tags passed
    result = processor.process_vod(
        job_id=job_id,
        video_path=video_path,
        output_dir=output_dir,
        # Auto-detect mode: don't pass any team/player info
        left_team=None,
        right_team=None,
        left_player_pool=None,
        right_player_pool=None,
    )
    
    return job_id


def compare_stats(job_id: str, video_filename: str, output_dir: str):
    """Compare detected stats to expected stats."""
    expected = EXPECTED_STATS.get(video_filename, {})
    expected_players = expected.get("players", {})
    expected_total = expected.get("total_kills", 0)
    
    # Load detected stats
    stats_file = os.path.join(output_dir, f"{job_id}_stats.json")
    if not os.path.exists(stats_file):
        print(f"ERROR: Stats file not found: {stats_file}")
        return
    
    with open(stats_file, 'r') as f:
        stats = json.load(f)
    
    detected_kills = stats.get('kills_by_player', {})
    detected_deaths = stats.get('deaths_by_player', {})
    detected_total = stats.get('total_kills', 0)
    
    print("\n" + "="*70)
    print(f"COMPARISON: {video_filename}")
    print("="*70)
    
    print(f"\nTotal Kills: Detected={detected_total}, Expected={expected_total}, Diff={detected_total - expected_total}")
    
    print(f"\n{'Player':<15} {'Det K':>6} {'Exp K':>6} {'Diff':>6} | {'Det D':>6} {'Exp D':>6} {'Diff':>6}")
    print("-" * 70)
    
    total_kill_diff = 0
    total_death_diff = 0
    matched_players = 0
    
    # Normalize player names for matching (case-insensitive)
    detected_kills_lower = {k.lower(): v for k, v in detected_kills.items()}
    detected_deaths_lower = {k.lower(): v for k, v in detected_deaths.items()}
    
    for player, exp_stats in expected_players.items():
        player_lower = player.lower()
        
        # Find detected stats (case-insensitive match)
        det_k = detected_kills_lower.get(player_lower, 0)
        det_d = detected_deaths_lower.get(player_lower, 0)
        
        # Also try to find by partial match
        if det_k == 0 and det_d == 0:
            for det_name in detected_kills.keys():
                if player_lower in det_name.lower() or det_name.lower() in player_lower:
                    det_k = detected_kills.get(det_name, 0)
                    det_d = detected_deaths.get(det_name, 0)
                    break
        
        exp_k = exp_stats["kills"]
        exp_d = exp_stats["deaths"]
        
        k_diff = det_k - exp_k
        d_diff = det_d - exp_d
        
        total_kill_diff += abs(k_diff)
        total_death_diff += abs(d_diff)
        
        if det_k > 0 or det_d > 0:
            matched_players += 1
        
        # Color coding for terminal
        k_status = "✓" if k_diff == 0 else ("+" if k_diff > 0 else "")
        d_status = "✓" if d_diff == 0 else ("+" if d_diff > 0 else "")
        
        print(f"{player:<15} {det_k:>6} {exp_k:>6} {k_diff:>+6} | {det_d:>6} {exp_d:>6} {d_diff:>+6}")
    
    print("-" * 70)
    print(f"{'TOTAL DIFF':<15} {'':<6} {'':<6} {total_kill_diff:>6} | {'':<6} {'':<6} {total_death_diff:>6}")
    print(f"\nMatched players: {matched_players}/{len(expected_players)}")
    
    # Show any detected players not in expected list
    expected_lower = {p.lower() for p in expected_players.keys()}
    extra_detected = []
    for player in detected_kills.keys():
        if player.lower() not in expected_lower:
            # Check if partial match
            is_match = any(player.lower() in e or e in player.lower() for e in expected_lower)
            if not is_match:
                extra_detected.append(player)
    
    if extra_detected:
        print(f"\nExtra detected players (not in expected): {extra_detected}")
    
    return {
        "total_kill_diff": total_kill_diff,
        "total_death_diff": total_death_diff,
        "matched_players": matched_players,
        "total_players": len(expected_players),
        "detected_total": detected_total,
        "expected_total": expected_total,
    }


def main():
    """Run all VOD tests."""
    output_dir = "/app/outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    # Process each VOD
    for video_filename in ["match_vod.mp4", "match_vod_2.mp4", "match_vod_3.mp4"]:
        print(f"\n\n{'#'*70}")
        print(f"# Starting: {video_filename}")
        print(f"{'#'*70}\n")
        
        job_id = run_single_vod(video_filename, output_dir)
        
        if job_id:
            comparison = compare_stats(job_id, video_filename, output_dir)
            results[video_filename] = {
                "job_id": job_id,
                "comparison": comparison,
            }
    
    # Final summary
    print("\n\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    for video, data in results.items():
        comp = data.get("comparison", {})
        if comp:
            print(f"\n{video}:")
            print(f"  Total: Detected={comp['detected_total']}, Expected={comp['expected_total']}, Diff={comp['detected_total'] - comp['expected_total']}")
            print(f"  Kill accuracy error: {comp['total_kill_diff']} (sum of per-player diffs)")
            print(f"  Death accuracy error: {comp['total_death_diff']} (sum of per-player diffs)")
            print(f"  Player matching: {comp['matched_players']}/{comp['total_players']}")


if __name__ == "__main__":
    main()
