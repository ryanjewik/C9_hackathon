"""Quick stats for VOD 8 strict roster run."""
import json
from collections import Counter

events = json.load(open("/app/outputs/crops-vod8-235935_events.json"))
kills = [e for e in events if e["type"] == "KILL_EVENT"]
self_kills = [e for e in kills if e["payload"].get("is_self_kill")]

print(f"Total kill events: {len(kills)}")
print(f"Self kills: {len(self_kills)}")
for sk in self_kills:
    p = sk["payload"]
    t = sk["t_ms"] / 1000
    print(f"  t={t:.1f}s  {p['killer_name']} -> {p['victim_name']}")

print()
ult = json.load(open("/app/outputs/crops/vod8_ult_diagnostics.json"))
ult_detected = [u for u in ult if u.get("is_ult_badge")]
print(f"Ult badge candidates checked: {len(ult)}")
print(f"Ult badge detections: {len(ult_detected)}")

print()
kc = Counter()
for e in kills:
    kc[e["payload"]["killer_name"]] += 1
print("Kills per player:")
for name, count in kc.most_common():
    print(f"  {name:15s} {count}")
