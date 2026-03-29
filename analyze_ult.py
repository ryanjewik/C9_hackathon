import json

with open(r"E:\cloud9_hackathon\ult_diag_vod9.json") as f:
    data = json.load(f)

print(f"Total entries: {len(data)}")

# Detected = True
detected = [d for d in data if d.get("detected")]
print(f"\n=== DETECTED (detected=True): {len(detected)} ===")
for d in detected:
    c = d["crop"]
    print(f"  crop#{c:3d}  victim={d['victim_pct']:.4f}  killer={d['killer_pct']:.4f}  blob={d['largest_blob']:5d}  bright={d['bright_pct']:.4f}  gap_w={d['gap_w']}")

# Near misses: victim_pct >= 0.10 but not detected
near = [d for d in data if not d.get("detected") and d["victim_pct"] >= 0.10]
print(f"\n=== NEAR-MISSES (victim >= 0.10, not detected): {len(near)} ===")
for d in sorted(near, key=lambda x: -x["victim_pct"]):
    c = d["crop"]
    reason = []
    if d["victim_pct"] < 0.15:
        reason.append(f"victim<0.15")
    if d["largest_blob"] < 150:
        reason.append(f"blob<150")
    if d["bright_pct"] < 0.20:
        reason.append(f"bright<0.20")
    reason_str = ", ".join(reason) if reason else "all pass but detected=False?"
    print(f"  crop#{c:3d}  victim={d['victim_pct']:.4f}  killer={d['killer_pct']:.4f}  blob={d['largest_blob']:5d}  bright={d['bright_pct']:.4f}  gap_w={d['gap_w']}  blocked_by: {reason_str}")

# Distribution summary
print("\n=== VICTIM_PCT DISTRIBUTION (all entries) ===")
buckets = {}
for d in data:
    bucket = round(d["victim_pct"], 2)
    if bucket >= 0.05:
        buckets.setdefault(bucket, []).append(d["crop"])
for k in sorted(buckets.keys()):
    crops = buckets[k]
    print(f"  {k:.2f}: {len(crops)} crops -> {crops[:10]}")
