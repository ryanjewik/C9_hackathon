"""Parse extract logs for ghost player + REPLAY lookback filter removals per VOD.
Use this for runs that predate _invalidated.json support."""
import re, sys
from collections import defaultdict

log_files = [
    ('extract_log_all_v25g.txt', 'unicode'),
    ('extract_log_v25g_vod6-9.txt', 'unicode'),
]

# Track which VOD is being processed by looking for job_id patterns
ghost_removals = defaultdict(list)   # vod -> [(killer, victim, t_sec)]
ghost_players = defaultdict(set)     # vod -> {player_names}
replay_removals = defaultdict(list)  # vod -> [(count, from_t, to_t)]

for log_file, encoding_hint in log_files:
    try:
        with open(log_file, encoding='utf-16') as f:
            lines = f.readlines()
    except (FileNotFoundError, UnicodeDecodeError):
        try:
            with open(log_file, encoding='utf-8') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"Skipping {log_file} (not found)")
            continue

    current_vod = None
    for line in lines:
        # Track current VOD from job_id like [crops-vod7-083929]
        m = re.search(r'\[crops-vod(\d+)-\d+\]', line)
        if m:
            current_vod = int(m.group(1))

        # Also track VOD from processing start messages
        m2 = re.search(r'Processing VOD\s*(\d+)|Starting.*vod(\d+)', line, re.IGNORECASE)
        if m2:
            current_vod = int(m2.group(1) or m2.group(2))

        # Ghost player filter: identified
        m = re.search(r'\[crops-vod(\d+)-\d+\] Ghost player filter: identified \d+ potential ghost players: \{(.+?)\}', line)
        if m:
            vod = int(m.group(1))
            names = [n.strip().strip("'\"") for n in m.group(2).split(',')]
            ghost_players[vod].update(names)

        # Ghost filter removed: killer killed victim at t=Xs
        m = re.search(r'\[crops-vod(\d+)-\d+\] Ghost filter removed: (.+?) killed (.+?) at t=(\d+\.\d+)s', line)
        if m:
            vod = int(m.group(1))
            ghost_removals[vod].append((m.group(2), m.group(3), float(m.group(4))))

        # REPLAY lookback filter: removed N kill(s) from t=Xs to t=Ys
        m = re.search(r'REPLAY lookback filter: removed (\d+) kill\(s\) from t=(\d+\.\d+)s to t=(\d+\.\d+)s', line)
        if m:
            count = int(m.group(1))
            from_t = float(m.group(2))
            to_t = float(m.group(3))
            if current_vod is not None:
                replay_removals[current_vod].append((count, from_t, to_t))

# Print summary
all_vods = sorted(set(ghost_removals) | set(replay_removals) | set(range(1, 10)))
total_ghost = 0
total_replay = 0

for vod in all_vods:
    gr = ghost_removals.get(vod, [])
    rr = replay_removals.get(vod, [])
    replay_count = sum(c for c, _, _ in rr)
    total_ghost += len(gr)
    total_replay += replay_count

    print(f'\nVOD {vod}:')
    print(f'  Ghost kills removed: {len(gr)}')
    if ghost_players.get(vod):
        print(f'    Ghost players: {ghost_players[vod]}')
    for killer, victim, t in gr:
        print(f'    t={t:.1f}s  {killer} -> {victim}')
    print(f'  REPLAY kills removed: {replay_count} (across {len(rr)} replay detections)')
    for count, from_t, to_t in rr:
        print(f'    {count} kill(s) between t={from_t:.1f}s - {to_t:.1f}s')
    print(f'  Total invalidated: {len(gr) + replay_count}')

print(f'\n{"="*50}')
print(f'GRAND TOTAL: {total_ghost} ghost + {total_replay} replay = {total_ghost + total_replay} invalidated kills')
print(f'Note: Ghost kills have orphan crops on disk. REPLAY kills may or may not have crops.')
