import re
from collections import defaultdict

with open(r'E:\cloud9_hackathon\extract_log_vod9_v1.txt', 'r', encoding='utf-16') as f:
    lines = f.readlines()

LEFT_TEAM = "TL"
RIGHT_TEAM = "VIT"
LEFT_PLAYERS = {"kamo", "miniboo", "purp0", "wayne", "nats"}
RIGHT_PLAYERS = {"derke", "chronicle", "jamppi", "profek", "unfake"}

# Parse kills, self-kills, removed kills, round transitions
kills = []  # (t, round, killer, victim, is_self_kill)
removed_set = set()  # (t, killer, victim) for fast lookup
round_transitions = []  # (t, left_score, right_score)

for line in lines:
    line = line.strip()
    
    # Self-kill
    m_self = re.match(r'\[KILL\] t=([\d.]+)s R(\d+) ROW \d+: SELF-KILL \(fall damage\) (\S+)', line)
    if m_self:
        kills.append((float(m_self.group(1)), int(m_self.group(2)), m_self.group(3), m_self.group(3), True))
        continue
    
    # Normal kill
    m_kill = re.match(r'\[KILL\] t=([\d.]+)s R(\d+) ROW \d+: (.+?) killed (.+)', line)
    if m_kill:
        kills.append((float(m_kill.group(1)), int(m_kill.group(2)), m_kill.group(3), m_kill.group(4), False))
        continue
    
    # Removed kill
    m_rem = re.match(r'\[KILL-REMOVED\] t=([\d.]+)s (.+?) killed (.+?) \(replay filter\)', line)
    if m_rem:
        removed_set.add((float(m_rem.group(1)), m_rem.group(2), m_rem.group(3)))
        continue
    m_rem_self = re.match(r'\[KILL-REMOVED\] t=([\d.]+)s (.+?) SELF-KILL', line)
    if m_rem_self:
        removed_set.add((float(m_rem_self.group(1)), m_rem_self.group(2), m_rem_self.group(2)))
        continue
    
    # Round transition
    m_rnd = re.match(r'\[ROUND\] Score: \d+-\d+ -> (\d+)-(\d+) at t=([\d.]+)s', line)
    if m_rnd:
        round_transitions.append((float(m_rnd.group(3)), int(m_rnd.group(1)), int(m_rnd.group(2))))

# Filter out removed kills
filtered_kills = []
for t, rnd, killer, victim, is_self in kills:
    if (t, killer, victim) in removed_set:
        continue
    filtered_kills.append((t, rnd, killer, victim, is_self))

# Group by round
rounds = defaultdict(list)
for t, rnd, killer, victim, is_self in filtered_kills:
    rounds[rnd].append((t, killer, victim, is_self))

# Build score map from transitions
score_after = {}
sorted_transitions = sorted(round_transitions, key=lambda x: x[0])
for i, (t, ls, rs) in enumerate(sorted_transitions):
    rnd_num = i + 1
    score_after[rnd_num] = (ls, rs)

# Determine who won each round
def get_winner(rnd):
    if rnd not in score_after:
        return "?"
    ls, rs = score_after[rnd]
    if rnd == 1:
        if rs > 0:
            return RIGHT_TEAM
        else:
            return LEFT_TEAM
    else:
        prev_ls, prev_rs = score_after.get(rnd - 1, (0, 0))
        if ls > prev_ls:
            return LEFT_TEAM
        elif rs > prev_rs:
            return RIGHT_TEAM
    return "?"

# Player team lookup
def get_team(player):
    if player.lower() in LEFT_PLAYERS:
        return LEFT_TEAM
    elif player.lower() in RIGHT_PLAYERS:
        return RIGHT_TEAM
    return "?"

# Build output
output_lines = []
output_lines.append(f"VOD 9 — Round-by-Round Kill Breakdown (2s Replay Filter)")
output_lines.append(f"{LEFT_TEAM} (left): kamo, MiniBoo, purp0, wayne, nAts")
output_lines.append(f"{RIGHT_TEAM} (right): Derke, Chronicle, Jamppi, PROFEK, UNFAKE")
output_lines.append(f"Replay kills removed: {len(removed_set)}")
output_lines.append("=" * 60)
output_lines.append("")

max_round = max(rounds.keys()) if rounds else 0
kill_totals = defaultdict(int)

for rnd in range(1, max_round + 1):
    rnd_kills = rounds.get(rnd, [])
    ls, rs = score_after.get(rnd, (0, 0))
    winner = get_winner(rnd)
    output_lines.append(f"Round {rnd}  (score after: {ls}-{rs}, won by {winner})")
    output_lines.append("-" * 40)
    for t, killer, victim, is_self in sorted(rnd_kills, key=lambda x: x[0]):
        if is_self:
            output_lines.append(f"  {killer} SELF-KILL (fall damage)")
        else:
            output_lines.append(f"  {killer} -> {victim}")
            kill_totals[killer] += 1
    output_lines.append(f"  ({len(rnd_kills)} kills)")
    output_lines.append("")

# Player totals
output_lines.append("=" * 60)
output_lines.append("PLAYER KILL TOTALS")
output_lines.append("-" * 40)
sorted_players = sorted(kill_totals.items(), key=lambda x: -x[1])
total = 0
for player, count in sorted_players:
    team = get_team(player)
    output_lines.append(f"  {player} ({team}): {count}")
    total += count
output_lines.append(f"  Total: {total}")
output_lines.append(f"  Self-kills: {sum(1 for t, rnd, killer, victim, is_self in filtered_kills if is_self)}")
output_lines.append(f"  Replay-filtered: {len(removed_set)}")

result = "\n".join(output_lines)
with open(r'E:\cloud9_hackathon\vod9_killfeed_breakdown.txt', 'w', encoding='utf-8') as f:
    f.write(result)
print(result)
