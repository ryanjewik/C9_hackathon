import re

with open(r'E:\cloud9_hackathon\extract_log_vod8_replay2s.txt', 'r', encoding='utf-16') as f:
    lines = f.readlines()

kills = []
self_kills = []
removed = []
round_transitions = []

for line in lines:
    line = line.strip()
    m = re.match(r'\[KILL\] t=([\d.]+)s R(\d+) ROW \d+: (.+?) killed (.+)', line)
    if m:
        kills.append({'t': float(m.group(1)), 'round': int(m.group(2)), 'killer': m.group(3), 'victim': m.group(4)})
        continue
    m2 = re.match(r'\[KILL\] t=([\d.]+)s R(\d+) ROW \d+: SELF-KILL \(fall damage\) (\S+)', line)
    if m2:
        self_kills.append({'t': float(m2.group(1)), 'round': int(m2.group(2)), 'player': m2.group(3)})
        continue
    m3 = re.match(r'\[KILL-REMOVED\] t=([\d.]+)s (.+?) killed (.+?) \(replay filter\)', line)
    if m3:
        removed.append({'t': float(m3.group(1)), 'killer': m3.group(2), 'victim': m3.group(3)})
        continue
    m4 = re.match(r'\[KILL-REMOVED\] t=([\d.]+)s (.+?) SELF-KILL', line)
    if m4:
        removed.append({'t': float(m4.group(1)), 'killer': m4.group(2), 'victim': m4.group(2), 'self_kill': True})
        continue
    m5 = re.match(r'\[ROUND\] .+ at t=([\d.]+)s', line)
    if m5:
        round_transitions.append(float(m5.group(1)))

print(f"Total kills: {len(kills)}, self-kills: {len(self_kills)}, removed by replay: {len(removed)}")
print(f"Round transitions: {len(round_transitions)}")
print()

# Show rounds 1-6
for rnd in range(1, 7):
    rnd_kills = [k for k in kills if k['round'] == rnd]
    rnd_self = [s for s in self_kills if s['round'] == rnd]
    print(f"Round {rnd}:")
    for k in rnd_kills:
        print(f"  {k['killer']} -> {k['victim']}  (t={k['t']:.1f}s)")
    for s in rnd_self:
        print(f"  {s['player']} SELF-KILL  (t={s['t']:.1f}s)")
    if not rnd_kills and not rnd_self:
        print("  (no kills)")
    print()

print("REMOVED kills by replay filter:")
for r in removed:
    tag = " (self-kill)" if r.get('self_kill') else ""
    print(f"  t={r['t']:.1f}s {r['killer']} -> {r['victim']}{tag}")
