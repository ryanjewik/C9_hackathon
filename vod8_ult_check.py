"""Check ult badge diagnostics — find candidates closest to triggering."""
import json

ult = json.load(open("/app/outputs/crops/vod8_ult_diagnostics.json"))
print("Total candidates:", len(ult))
detected = [u for u in ult if u.get("detected")]
print("Detected as ult:", len(detected))

by_victim_pct = sorted(ult, key=lambda u: u.get("victim_pct", 0), reverse=True)
print("\nTop 10 by victim_pct (threshold >= 0.15):")
for u in by_victim_pct[:10]:
    c = u.get("crop", "?")
    vp = u.get("victim_pct", 0)
    kp = u.get("killer_pct", 0)
    gw = u.get("gap_w", "?")
    bl = u.get("largest_blob", "?")
    bp = u.get("bright_pct", 0)
    det = u.get("detected", False)
    print("  crop=%3s  vpct=%.3f  kpct=%.3f  gap=%s  blob=%s  bright=%.3f  det=%s" % (c, vp, kp, gw, bl, bp, det))
