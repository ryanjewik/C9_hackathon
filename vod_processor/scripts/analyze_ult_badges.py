"""
Analyze ult badge diagnostics saved by extract_crops.py.

Reads the vod*_ult_diagnostics.json files from the crops directory
and displays a summary table to help tune detection thresholds.

Usage:
    python analyze_ult_badges.py                        # All VODs
    python analyze_ult_badges.py --vod 8                # Specific VOD
    python analyze_ult_badges.py --min-victim 0.10      # Custom threshold preview
"""
import json
import glob
import argparse
import os


def load_diagnostics(crops_dir: str, vod: int = None):
    pattern = os.path.join(crops_dir, f"vod{vod}_ult_diagnostics.json") if vod else \
              os.path.join(crops_dir, "vod*_ult_diagnostics.json")
    entries = []
    for path in sorted(glob.glob(pattern)):
        vod_num = os.path.basename(path).split("_")[0]  # "vod8"
        with open(path) as f:
            data = json.load(f)
        for d in data:
            d["vod"] = vod_num
            entries.append(d)
    return entries


def print_table(entries, min_victim=None):
    if not entries:
        print("No diagnostics found.")
        return

    # Header
    print(f"{'vod':<6} {'crop':>5} {'gap_w':>5} {'teal%':>7} {'red%':>7} "
          f"{'killer%':>8} {'victim%':>8} {'blob':>6} {'det':>4}")
    print("-" * 68)

    for e in sorted(entries, key=lambda x: (x["vod"], x["crop"])):
        victim = e["victim_pct"]
        det_str = "YES" if e["detected"] else ""

        # Highlight rows above the threshold preview
        marker = ""
        if min_victim is not None and victim >= min_victim:
            marker = " <<<"

        print(f"{e['vod']:<6} {e['crop']:>5} {e['gap_w']:>5} "
              f"{e['teal_pct']:>6.1%} {e['red_pct']:>6.1%} "
              f"{e['killer_pct']:>7.1%} {e['victim_pct']:>7.1%} "
              f"{e['largest_blob']:>6} {det_str:>4}{marker}")

    # Summary
    detected = [e for e in entries if e["detected"]]
    print(f"\n{len(detected)} / {len(entries)} detected with current thresholds "
          f"(victim>=15%, blob>=150)")

    if min_victim is not None:
        would_detect = [e for e in entries if e["victim_pct"] >= min_victim and e["largest_blob"] >= 150]
        print(f"{len(would_detect)} / {len(entries)} would fire at victim>={min_victim:.0%}, blob>=150")


def main():
    parser = argparse.ArgumentParser(description="Analyze ult badge diagnostics")
    parser.add_argument("--crops-dir", default="/app/outputs/crops",
                        help="Path to crops directory")
    parser.add_argument("--vod", type=int, default=None, help="Filter to specific VOD")
    parser.add_argument("--min-victim", type=float, default=None,
                        help="Preview: what-if victim threshold (e.g. 0.20)")
    args = parser.parse_args()

    entries = load_diagnostics(args.crops_dir, args.vod)
    print_table(entries, args.min_victim)


if __name__ == "__main__":
    main()
