"""
Analyze kills per player from VOD test output and generate comparison charts.
"""
import re
from collections import defaultdict
import matplotlib.pyplot as plt

# Parse the test output - paste the kill events here or read from file
test_output = """
[KILL] t=203.9s R1 ROW 1: crashies killed s0m
[KILL] t=208.1s R1 ROW 2: skuba killed Boaster
[KILL] t=208.6s R1 ROW 3: Chronicle killed mada
[KILL] t=208.9s R1 ROW 4: skuba killed kaajak
[KILL] t=211.0s R1 ROW 4: crashies killed skuba
[KILL] t=212.6s R1 ROW 5: brawk killed Alfajer
[KILL] t=236.2s R1 ROW 1: brawk killed crashies
[KILL] t=239.8s R1 ROW 2: brawk killed Chronicle
[ROUND] Score: 0-0 -> 1-0 at t=275.6s
[KILL] t=351.6s R2 ROW 1: skuba killed kaajak
[KILL] t=354.9s R2 ROW 2: s0m killed Alfajer
[KILL] t=379.4s R2 ROW 1: skuba killed Chronicle
[KILL] t=382.0s R2 ROW 2: skuba killed crashies
[KILL] t=385.3s R2 ROW 2: skuba killed Boaster
[ROUND] Score: 1-0 -> 2-0 at t=393.6s
[KILL] t=511.3s R3 ROW 1: mada killed crashies
[KILL] t=513.4s R3 ROW 2: s0m killed Chronicle
[KILL] t=523.7s R3 ROW 1: kaajak killed mada
[ROUND] Score: 2-0 -> 3-0 at t=530.3s
[KILL] t=586.6s R4 ROW 1: Boaster killed s0m
[KILL] t=613.4s R4 ROW 1: Ethan killed kaajak
[KILL] t=613.4s R4 ROW 2: Boaster killed Ethan
[KILL] t=614.7s R4 ROW 3: mada killed Boaster
[KILL] t=625.2s R4 ROW 1: crashies killed mada
[KILL] t=636.2s R4 ROW 1: crashies killed skuba
[ROUND] Score: 3-0 -> 3-1 at t=639.7s
[KILL] t=640.0s R4 ROW 2: Chronicle killed brawk
[KILL] t=694.0s R5 ROW 1: Ethan killed kaajak
[KILL] t=731.3s R5 ROW 1: Boaster killed s0m
[KILL] t=736.3s R5 ROW 2: mada killed Chronicle
[KILL] t=738.9s R5 ROW 2: mada killed Boaster
[KILL] t=753.3s R5 ROW 1: skuba killed crashies
[KILL] t=761.3s R5 ROW 1: skuba killed Alfajer
[ROUND] Score: 3-1 -> 4-1 at t=761.8s
[KILL] t=831.5s R6 ROW 1: mada killed Alfajer
[KILL] t=836.4s R6 ROW 2: s0m killed Chronicle
[KILL] t=836.6s R6 ROW 3: Ethan killed kaajak
[KILL] t=836.8s R6 ROW 3: Ethan killed Boaster
[KILL] t=846.0s R6 ROW 1: mada killed crashies
[ROUND] Score: 4-1 -> 5-1 at t=846.4s
[KILL] t=1011.5s R7 ROW 1: mada killed crashies
[KILL] t=1027.6s R7 ROW 1: Ethan killed kaajak
[KILL] t=1031.2s R7 ROW 2: brawk killed Chronicle
[KILL] t=1031.6s R7 ROW 3: Ethan killed Boaster
[KILL] t=1034.1s R7 ROW 3: brawk killed Alfajer
[ROUND] Score: 5-1 -> 6-1 at t=1034.5s
[KILL] t=1087.5s R8 ROW 1: brawk killed kaajak
[KILL] t=1091.0s R8 ROW 2: Ethan killed Boaster
[KILL] t=1094.4s R8 ROW 2: Chronicle killed Ethan
[KILL] t=1100.3s R8 ROW 1: crashies killed brawk
[KILL] t=1102.8s R8 ROW 2: s0m killed Chronicle
[KILL] t=1105.2s R8 ROW 3: crashies killed s0m
[KILL] t=1146.7s R8 ROW 1: skuba killed Alfajer
[KILL] t=1148.5s R8 ROW 2: mada killed crashies
[ROUND] Score: 6-1 -> 7-1 at t=1257.7s
[KILL] t=1283.0s R9 ROW 1: Alfajer killed s0m
[KILL] t=1300.0s R9 ROW 1: skuba killed Boaster
[KILL] t=1307.0s R9 ROW 1: Ethan killed Chronicle
[KILL] t=1307.8s R9 ROW 2: mada killed Alfajer
[KILL] t=1312.8s R9 ROW 2: skuba killed kaajak
[KILL] t=1317.4s R9 ROW 2: skuba killed crashies
[ROUND] Score: 7-1 -> 8-1 at t=1318.5s
[KILL] t=1398.2s R10 ROW 1: brawk killed Chronicle
[KILL] t=1401.5s R10 ROW 2: brawk killed kaajak
[KILL] t=1401.7s R10 ROW 3: brawk killed Boaster
[KILL] t=1403.1s R10 ROW 4: crashies killed brawk
[KILL] t=1422.1s R10 ROW 1: Ethan killed Alfajer
[KILL] t=1425.4s R10 ROW 2: s0m killed crashies
[ROUND] Score: 8-1 -> 9-1 at t=1461.5s
[KILL] t=1492.3s R11 ROW 1: skuba killed crashies
[KILL] t=1515.5s R11 ROW 1: skuba killed Chronicle
[KILL] t=1525.4s R11 ROW 1: skuba killed kaajak
[KILL] t=1527.1s R11 ROW 2: skuba killed Alfajer
[KILL] t=1528.3s R11 ROW 3: brawk killed Boaster
[ROUND] Score: 9-1 -> 10-1 at t=1528.7s
[KILL] t=1574.6s R12 ROW 1: mada killed Alfajer
[KILL] t=1616.0s R12 ROW 1: brawk killed kaajak
[KILL] t=1634.7s R12 ROW 1: s0m killed Chronicle
[KILL] t=1639.5s R12 ROW 2: Boaster killed s0m
[KILL] t=1652.6s R12 ROW 1: Ethan killed Boaster
[ROUND] Score: 10-1 -> 11-1 at t=1653.8s
[KILL] t=1655.1s R12 ROW 2: Ethan killed crashies
[KILL] t=1878.3s R13 ROW 1: Chronicle killed mada
[KILL] t=1881.2s R13 ROW 2: Alfajer killed Ethan
[KILL] t=1882.5s R13 ROW 3: s0m killed Boaster
[KILL] t=1883.1s R13 ROW 4: Chronicle killed s0m
[KILL] t=1883.6s R13 ROW 5: skuba killed crashies
[KILL] t=1892.7s R13 ROW 1: Chronicle killed skuba
[KILL] t=1896.4s R13 ROW 2: Chronicle killed brawk
[ROUND] Score: 11-1 -> 11-2 at t=1896.6s
[KILL] t=1947.0s R14 ROW 1: Chronicle killed mada
[KILL] t=1955.2s R14 ROW 1: kaajak killed skuba
[KILL] t=2000.0s R14 ROW 1: Alfajer killed Ethan
[KILL] t=2001.3s R14 ROW 2: Alfajer killed s0m
[KILL] t=2003.3s R14 ROW 3: brawk killed Chronicle
[KILL] t=2003.9s R14 ROW 4: crashies killed brawk
[ROUND] Score: 11-2 -> 11-3 at t=2030.4s
[KILL] t=2070.3s R15 ROW 1: kaajak killed mada
[KILL] t=2086.8s R15 ROW 1: Alfajer killed brawk
[KILL] t=2089.8s R15 ROW 2: Ethan killed Chronicle
[KILL] t=2090.1s R15 ROW 3: kaajak killed Ethan
[KILL] t=2091.8s R15 ROW 4: kaajak killed skuba
[KILL] t=2094.8s R15 ROW 4: crashies killed s0m
[KILL] t=2094.9s R15 ROW 3: s0m killed kaajak
[ROUND] Score: 11-3 -> 11-4 at t=2135.3s
[KILL] t=2186.5s R16 ROW 1: mada killed Boaster
[KILL] t=2187.3s R16 ROW 2: kaajak killed s0m
[KILL] t=2187.7s R16 ROW 3: crashies killed mada
[KILL] t=2194.5s R16 ROW 1: Alfajer killed brawk
[KILL] t=2196.1s R16 ROW 2: Alfajer killed Ethan
[KILL] t=2198.6s R16 ROW 3: Chronicle killed skuba
[ROUND] Score: 11-4 -> 11-5 at t=2235.2s
[KILL] t=2267.9s R17 ROW 1: kaajak killed s0m
[KILL] t=2269.9s R17 ROW 2: brawk killed kaajak
[KILL] t=2311.1s R17 ROW 1: Chronicle killed mada
[ROUND] Score: 11-5 -> 11-6 at t=2344.1s
[KILL] t=2481.7s R18 ROW 1: kaajak killed s0m
[KILL] t=2485.8s R18 ROW 2: kaajak killed mada
[KILL] t=2488.5s R18 ROW 2: Ethan killed kaajak
[KILL] t=2513.6s R18 ROW 1: Ethan killed Boaster
[KILL] t=2515.6s R18 ROW 2: brawk killed Alfajer
[KILL] t=2527.7s R18 ROW 1: Chronicle killed brawk
[KILL] t=2529.6s R18 ROW 2: skuba killed crashies
[ROUND] Score: 11-6 -> 12-6 at t=2530.3s
[KILL] t=2532.1s R18 ROW 3: skuba killed Chronicle
[KILL] t=2573.1s R19 ROW 1: kaajak killed brawk
[KILL] t=2623.2s R19 ROW 1: Ethan killed Chronicle
[KILL] t=2627.2s R19 ROW 2: Boaster killed Ethan
[KILL] t=2630.6s R19 ROW 2: Alfajer killed s0m
[KILL] t=2635.0s R19 ROW 2: mada killed Alfajer
[KILL] t=2654.0s R19 ROW 1: Boaster killed skuba
[KILL] t=2657.7s R19 ROW 2: Boaster killed mada
[ROUND] Score: 12-6 -> 12-7 at t=2658.3s
[KILL] t=2706.4s R20 ROW 1: kaajak killed Ethan
[KILL] t=2718.8s R20 ROW 1: kaajak killed mada
[KILL] t=2719.0s R20 ROW 2: Alfajer killed s0m
[KILL] t=2719.5s R20 ROW 3: kaajak killed skuba
[KILL] t=2720.7s R20 ROW 4: brawk killed Alfajer
[KILL] t=2722.1s R20 ROW 5: Chronicle killed brawk
[ROUND] Score: 12-7 -> 12-8 at t=2722.3s
[KILL] t=2849.4s R21 ROW 1: Alfajer killed skuba
[KILL] t=2851.8s R21 ROW 2: crashies killed s0m
[KILL] t=2863.1s R21 ROW 1: kaajak killed mada
[KILL] t=2873.0s R21 ROW 1: crashies killed Ethan
[KILL] t=2876.2s R21 ROW 2: kaajak killed brawk
[ROUND] Score: 12-8 -> 12-9 at t=2908.1s
[KILL] t=2957.5s R22 ROW 1: kaajak killed skuba
[KILL] t=2960.6s R22 ROW 2: s0m killed kaajak
[KILL] t=2998.9s R22 ROW 1: Boaster killed Ethan
[KILL] t=3024.1s R22 ROW 1: Chronicle killed mada
[KILL] t=3030.9s R22 ROW 1: Chronicle killed brawk
[KILL] t=3032.1s R22 ROW 2: Chronicle killed s0m
[ROUND] Score: 12-9 -> 12-10 at t=3070.3s
[KILL] t=3146.6s R23 ROW 1: crashies killed mada
[KILL] t=3165.0s R23 ROW 1: Ethan killed Alfajer
[KILL] t=3165.2s R23 ROW 2: kaajak killed Ethan
[KILL] t=3169.3s R23 ROW 3: s0m killed kaajak
[KILL] t=3169.7s R23 ROW 4: crashies killed s0m
[KILL] t=3170.4s R23 ROW 4: crashies killed brawk
[KILL] t=3175.0s R23 ROW 2: skuba killed Chronicle
[KILL] t=3178.8s R23 ROW 2: crashies killed skuba
[ROUND] Score: 12-10 -> 12-11 at t=3182.9s
[KILL] t=3296.4s R24 ROW 1: skuba killed kaajak
[KILL] t=3297.8s R24 ROW 2: crashies killed Ethan
[KILL] t=3309.3s R24 ROW 1: crashies killed mada
[KILL] t=3312.4s R24 ROW 2: crashies killed skuba
[KILL] t=3313.7s R24 ROW 3: Alfajer killed brawk
[KILL] t=3316.9s R24 ROW 3: s0m killed Alfajer
[KILL] t=3323.9s R24 ROW 1: crashies killed s0m
[ROUND] Score: 12-11 -> 12-12 at t=3375.5s
[KILL] t=3447.3s R25 ROW 1: kaajak killed mada
[KILL] t=3448.4s R25 ROW 2: Ethan killed kaajak
[KILL] t=3455.7s R25 ROW 1: crashies killed skuba
[KILL] t=3457.6s R25 ROW 2: brawk killed crashies
[KILL] t=3469.2s R25 ROW 1: s0m killed Chronicle
[KILL] t=3485.5s R25 ROW 1: Alfajer killed s0m
[KILL] t=3487.1s R25 ROW 2: Boaster killed brawk
[KILL] t=3491.0s R25 ROW 2: Boaster killed Ethan
[ROUND] Score: 12-12 -> 12-13 at t=3491.4s
[KILL] t=3585.0s R26 ROW 1: Alfajer killed s0m
[KILL] t=3591.7s R26 ROW 1: brawk killed kaajak
[KILL] t=3592.0s R26 ROW 2: skuba killed Chronicle
[KILL] t=3592.4s R26 ROW 3: brawk killed Alfajer
[KILL] t=3610.7s R26 ROW 1: Boaster killed Ethan
[KILL] t=3616.5s R26 ROW 1: mada killed Boaster
[ROUND] Score: 12-13 -> 13-13 at t=3617.1s
[KILL] t=3619.1s R26 ROW 2: mada killed crashies
[KILL] t=3736.5s R27 ROW 1: crashies killed Ethan
[KILL] t=3739.9s R27 ROW 2: Alfajer killed s0m
[KILL] t=3739.9s R27 ROW 3: brawk killed Chronicle
[KILL] t=3750.7s R27 ROW 1: kaajak killed skuba
[KILL] t=3763.0s R27 ROW 1: mada killed Boaster
[KILL] t=3765.8s R27 ROW 2: kaajak killed mada
[KILL] t=3772.3s R27 ROW 1: kaajak killed brawk
[ROUND] Score: 13-13 -> 13-14 at t=3772.8s
[KILL] t=3813.7s R28 ROW 1: brawk killed Boaster
[KILL] t=3880.9s R28 ROW 1: Chronicle killed mada
[KILL] t=3881.2s R28 ROW 2: kaajak killed brawk
[KILL] t=3881.7s R28 ROW 3: Chronicle killed skuba
[KILL] t=3882.7s R28 ROW 4: s0m killed Alfajer
[KILL] t=3885.2s R28 ROW 5: kaajak killed Ethan
[KILL] t=3890.7s R28 ROW 1: kaajak killed s0m
"""

def parse_kills(output_text):
    """Parse kill events from test output."""
    kill_pattern = r'\[KILL\] t=[\d.]+s R(\d+) ROW \d+: (\w+) killed (\w+)'
    kills = []
    for match in re.finditer(kill_pattern, output_text):
        round_num = int(match.group(1))
        killer = match.group(2)
        victim = match.group(3)
        kills.append({'round': round_num, 'killer': killer, 'victim': victim})
    return kills

def count_kills_per_player(kills):
    """Count kills and deaths per player."""
    kill_counts = defaultdict(int)
    death_counts = defaultdict(int)
    for k in kills:
        kill_counts[k['killer']] += 1
        death_counts[k['victim']] += 1
    return dict(kill_counts), dict(death_counts)

def plot_kills_chart(kill_counts, death_counts, title, left_team, right_team):
    """Create a bar chart of kills per player."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left team
    left_players = [p for p in kill_counts.keys() if p in left_team]
    left_kills = [kill_counts.get(p, 0) for p in left_players]
    left_deaths = [death_counts.get(p, 0) for p in left_players]
    
    x = range(len(left_players))
    width = 0.35
    ax1.bar([i - width/2 for i in x], left_kills, width, label='Kills', color='green')
    ax1.bar([i + width/2 for i in x], left_deaths, width, label='Deaths', color='red')
    ax1.set_xticks(x)
    ax1.set_xticklabels(left_players, rotation=45, ha='right')
    ax1.set_ylabel('Count')
    ax1.set_title(f'{title} - Left Team')
    ax1.legend()
    
    # Right team
    right_players = [p for p in kill_counts.keys() if p in right_team]
    right_kills = [kill_counts.get(p, 0) for p in right_players]
    right_deaths = [death_counts.get(p, 0) for p in right_players]
    
    x = range(len(right_players))
    ax2.bar([i - width/2 for i in x], right_kills, width, label='Kills', color='green')
    ax2.bar([i + width/2 for i in x], right_deaths, width, label='Deaths', color='red')
    ax2.set_xticks(x)
    ax2.set_xticklabels(right_players, rotation=45, ha='right')
    ax2.set_ylabel('Count')
    ax2.set_title(f'{title} - Right Team')
    ax2.legend()
    
    plt.tight_layout()
    return fig

# VOD 1: NRG vs FNC
vod1_output = test_output
vod1_kills = parse_kills(vod1_output)
vod1_kill_counts, vod1_death_counts = count_kills_per_player(vod1_kills)

nrg_players = ['s0m', 'Ethan', 'mada', 'skuba', 'brawk']
fnc_players = ['crashies', 'Chronicle', 'Boaster', 'Alfajer', 'kaajak']

print("=" * 60)
print("VOD 1: NRG vs FNC")
print("=" * 60)
print(f"Total kills detected: {len(vod1_kills)}")
print("\nNRG Players:")
for p in nrg_players:
    print(f"  {p}: {vod1_kill_counts.get(p, 0)} kills, {vod1_death_counts.get(p, 0)} deaths")
print("\nFNC Players:")
for p in fnc_players:
    print(f"  {p}: {vod1_kill_counts.get(p, 0)} kills, {vod1_death_counts.get(p, 0)} deaths")

fig1 = plot_kills_chart(vod1_kill_counts, vod1_death_counts, "VOD 1: NRG vs FNC", nrg_players, fnc_players)
fig1.savefig('vod1_kills.png', dpi=150)
plt.close(fig1)
print("\nSaved: vod1_kills.png")
