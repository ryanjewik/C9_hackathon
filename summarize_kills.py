"""Summarize killfeed events per VOD from the latest v25g processing runs.
Includes invalidated kills (ghost player filter + REPLAY lookback filter)."""
import json, os, glob, re
from collections import defaultdict

# Find the LATEST events JSON per VOD (highest timestamp in filename)
output_dir = '/app/outputs'
latest = {}
for f in sorted(glob.glob(os.path.join(output_dir, 'crops-vod*_events.json'))):
    m = re.search(r'crops-vod(\d+)-(\d+)_events', f)
    if m:
        vod = int(m.group(1))
        ts = m.group(2)
        latest[vod] = f  # sorted order means last = latest

for vod in sorted(latest):
    path = latest[vod]
    fname = os.path.basename(path)
    base = path.replace('_events.json', '')
    with open(path) as f:
        events = json.load(f)

    kills = [e for e in events if e.get('type') == 'KILL_EVENT']
    deaths = [e for e in events if e.get('type') == 'DEATH_EVENT']

    # Load invalidated kills if available
    inv_path = base + '_invalidated.json'
    ghost_removed = []
    replay_removed = []
    if os.path.exists(inv_path):
        with open(inv_path) as f:
            inv = json.load(f)
        ghost_removed = [e for e in inv.get('ghost_removed', []) if e.get('type') == 'KILL_EVENT']
        replay_removed = [e for e in inv.get('replay_removed', []) if e.get('type') == 'KILL_EVENT']

    # Per-player kill counts
    killer_counts = defaultdict(int)
    victim_counts = defaultdict(int)
    self_kills = []
    player_teams = {}

    for k in kills:
        p = k['payload']
        killer = p.get('killer_name', '?')
        victim = p.get('victim_name', '?')
        killer_team = p.get('killer_team', '?')
        victim_team = p.get('victim_team', '?')
        is_self = p.get('is_self_kill', False)

        killer_counts[killer] += 1
        victim_counts[victim] += 1
        player_teams[killer] = killer_team
        player_teams[victim] = victim_team

        if is_self:
            self_kills.append((k['t_ms'], killer, victim))

    # Ult badge count from crops on disk
    ult_dir = os.path.join(output_dir, 'crops', 'ult_badge')
    ult_count = 0
    if os.path.isdir(ult_dir):
        ult_count = len([f for f in os.listdir(ult_dir) if f.startswith(f'vod{vod}_')])

    # All unique players
    all_players = set(killer_counts) | set(victim_counts)
    # Sort by team then name
    teal = sorted([p for p in all_players if player_teams.get(p) == 'teal'])
    orange = sorted([p for p in all_players if player_teams.get(p) == 'orange'])
    unknown = sorted([p for p in all_players if player_teams.get(p) not in ('teal', 'orange')])

    print(f'\n{"="*60}')
    print(f'VOD {vod}  ({fname})')
    print(f'{"="*60}')
    print(f'Total KILL_EVENTs: {len(kills)}, DEATH_EVENTs: {len(deaths)}')
    print(f'Self-kills: {len(self_kills)}, Ult badges saved: {ult_count}')

    print(f'\n  {"Player":<20} {"Team":<8} {"Kills":>6} {"Deaths":>7}')
    print(f'  {"-"*20} {"-"*8} {"-"*6} {"-"*7}')
    for group_label, group in [('TEAL', teal), ('ORANGE', orange), ('???', unknown)]:
        if not group:
            continue
        for p in group:
            k = killer_counts.get(p, 0)
            d = victim_counts.get(p, 0)
            print(f'  {p:<20} {player_teams.get(p,"?"):<8} {k:>6} {d:>7}')
        # Team totals
        tk = sum(killer_counts.get(p, 0) for p in group)
        td = sum(victim_counts.get(p, 0) for p in group)
        print(f'  {"["+group_label+" TOTAL]":<20} {"":8} {tk:>6} {td:>7}')
        print()

    if self_kills:
        print(f'  Self-kills:')
        for t, killer, victim in self_kills:
            print(f'    t={t/1000:.1f}s  {killer} -> {victim}')

    # Invalidated kills section
    if ghost_removed or replay_removed:
        print(f'\n  --- INVALIDATED KILLS ---')
        print(f'  Ghost player filter: {len(ghost_removed)} kills removed')
        for e in ghost_removed:
            p = e.get('payload', {})
            print(f'    t={e["t_ms"]/1000:.1f}s  {p.get("killer_name","?")} -> {p.get("victim_name","?")} (ghost victim)')
        print(f'  REPLAY lookback filter: {len(replay_removed)} kills removed')
        for e in replay_removed:
            p = e.get('payload', {})
            print(f'    t={e["t_ms"]/1000:.1f}s  {p.get("killer_name","?")} -> {p.get("victim_name","?")}')
        print(f'  Total invalidated: {len(ghost_removed) + len(replay_removed)} (crops may still exist on disk)')
    else:
        print(f'\n  Invalidated kills: (no _invalidated.json found - re-run to generate)')
