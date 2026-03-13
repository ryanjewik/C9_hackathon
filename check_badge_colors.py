"""
Analyze color percentages on existing ult badge crop images.

Runs the same HSV teal/red analysis used by _maybe_extract_ult_badge
on each PNG in the given directory so you can compare real badges
vs false positives and tune thresholds.

Usage:
    python check_badge_colors.py                              # default: local_crops_v2/ult_badge
    python check_badge_colors.py local_crops_v2/ult_badge     # explicit path
    python check_badge_colors.py --all                        # also scan diag/ rows
"""
import sys
import os
import glob
import cv2
import numpy as np


def analyze_image(path):
    img = cv2.imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Same masks as _maybe_extract_ult_badge
    teal_mask = cv2.inRange(hsv, np.array([75, 50, 80]), np.array([115, 255, 255]))
    red_mask1 = cv2.inRange(hsv, np.array([0, 120, 140]), np.array([10, 255, 255]))
    red_mask2 = cv2.inRange(hsv, np.array([170, 120, 140]), np.array([179, 255, 255]))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    total = h * w
    teal_pct = cv2.countNonZero(teal_mask) / total
    red_pct = cv2.countNonZero(red_mask) / total

    if teal_pct >= red_pct:
        killer_pct, victim_pct = teal_pct, red_pct
        victim_mask = red_mask
        victim_color = "red"
    else:
        killer_pct, victim_pct = red_pct, teal_pct
        victim_mask = teal_mask
        victim_color = "teal"

    # Largest contiguous blob of victim color
    contours, _ = cv2.findContours(victim_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_blob = max(cv2.contourArea(c) for c in contours) if contours else 0

    # White/bright pixel percentage (weapon icon pixels)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bright = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    bright_pct = cv2.countNonZero(bright) / total

    return {
        "file": os.path.basename(path),
        "size": f"{w}x{h}",
        "teal_pct": teal_pct,
        "red_pct": red_pct,
        "killer_pct": killer_pct,
        "victim_pct": victim_pct,
        "victim_color": victim_color,
        "largest_blob": int(largest_blob),
        "bright_pct": bright_pct,
    }


def main():
    default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "local_crops_v2", "ult_badge")
    target = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else default_dir

    if not os.path.isdir(target):
        print(f"Directory not found: {target}")
        sys.exit(1)

    pngs = sorted(glob.glob(os.path.join(target, "*.png")))
    if not pngs:
        print(f"No PNG files in {target}")
        sys.exit(1)

    print(f"Analyzing {len(pngs)} images in {target}\n")
    print(f"{'file':<40} {'size':>7} {'teal%':>7} {'red%':>7} "
          f"{'killer%':>8} {'victim%':>8} {'v_color':>7} {'blob':>6} {'bright%':>8}")
    print("-" * 110)

    for path in pngs:
        r = analyze_image(path)
        if r is None:
            print(f"{os.path.basename(path):<40} ERROR reading")
            continue
        print(f"{r['file']:<40} {r['size']:>7} "
              f"{r['teal_pct']:>6.1%} {r['red_pct']:>6.1%} "
              f"{r['killer_pct']:>7.1%} {r['victim_pct']:>7.1%} "
              f"{r['victim_color']:>7} {r['largest_blob']:>6} {r['bright_pct']:>7.1%}")

    print(f"\nCurrent thresholds: victim_pct >= 15%, largest_blob >= 150")
    print("Crop 59 is the only real badge — compare its values to the false positives.")


if __name__ == "__main__":
    main()
