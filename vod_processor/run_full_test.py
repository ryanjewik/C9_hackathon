#!/usr/bin/env python3
"""Run full VOD processing and analyze results."""
import sys
import json
sys.path.insert(0, '/app')

from vod_processor.app.services.processing.vod_processor import process_vod

print("Starting full processing of match_vod_3.mp4...")
print("=" * 60)

result = process_vod('/app/match_vod_3.mp4', 'test_rounds_fix')

print("\n" + "=" * 60)
print("Processing complete!")

events = result.get('events', [])
print(f"Total events: {len(events)}")

# Count kills per round
kills = [e for e in events if e.get('type') == 'KILL']
print(f"Total kills detected: {len(kills)}")

rounds = {}
for kill in kills:
    r = kill.get('payload', {}).get('round_number', 0)
    rounds[r] = rounds.get(r, 0) + 1

print("\nKills per round:")
for r in sorted(rounds.keys()):
    count = rounds[r]
    bar = '#' * min(count, 30)
    print(f"  Round {r:2d}: {count:3d} kills {bar}")

# Player kill counts
print("\n" + "=" * 60)
print("Kill counts by player:")

player_kills = {}
player_deaths = {}

for kill in kills:
    payload = kill.get('payload', {})
    killer = payload.get('killer', 'Unknown')
    victim = payload.get('victim', 'Unknown')
    
    player_kills[killer] = player_kills.get(killer, 0) + 1
    player_deaths[victim] = player_deaths.get(victim, 0) + 1

# Sort by kills descending
sorted_players = sorted(player_kills.items(), key=lambda x: -x[1])

print("\nKills:")
for player, count in sorted_players:
    print(f"  {player:20s}: {count:3d}")

print("\nDeaths:")
sorted_deaths = sorted(player_deaths.items(), key=lambda x: -x[1])
for player, count in sorted_deaths:
    print(f"  {player:20s}: {count:3d}")

# Save full results
with open('/app/outputs/test_results.json', 'w') as f:
    json.dump({
        'total_events': len(events),
        'total_kills': len(kills),
        'kills_per_round': rounds,
        'player_kills': player_kills,
        'player_deaths': player_deaths
    }, f, indent=2)

print("\nResults saved to /app/outputs/test_results.json")
