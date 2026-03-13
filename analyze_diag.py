"""
Analyze diagnostic row images to extract the annotated line positions
(red=ktr, blue=vtl, green=crop bounds) and measure the gap, crop width, etc.

Usage:
    python analyze_diag.py local_crops_v3    # analyze v3 diagnostics
    python analyze_diag.py local_crops_v4    # analyze v4 diagnostics
    python analyze_diag.py local_crops_v3 --crops 67 68 205 46 154
"""
import sys
import os
import glob
import cv2
import numpy as np
import argparse


def find_line_x(img, target_bgr, tolerance=40):
    """Find x positions of vertical lines drawn in a specific color."""
    h, w = img.shape[:2]
    positions = []
    for x in range(w):
        col = img[:, x, :]
        diff = np.abs(col.astype(int) - np.array(target_bgr, dtype=int))
        match_count = np.sum(np.all(diff < tolerance, axis=1))
        if match_count > h * 0.4:  # line spans at least 40% of height
            positions.append(x)
    # Cluster nearby positions
    if not positions:
        return []
    clusters = []
    cluster_start = positions[0]
    prev = positions[0]
    for p in positions[1:]:
        if p - prev > 3:
            clusters.append((cluster_start + prev) // 2)
            cluster_start = p
        prev = p
    clusters.append((cluster_start + prev) // 2)
    return clusters


def analyze_row(diag_path):
    img = cv2.imread(diag_path)
    if img is None:
        return None
    h, w = img.shape[:2]

    # Red line (BGR: 0,0,255) = ktr (killer_text_right)
    red_lines = find_line_x(img, [0, 0, 255])
    # Blue line (BGR: 255,0,0) = vtl (victim_text_left)
    blue_lines = find_line_x(img, [255, 0, 0])
    # Green lines (BGR: 0,255,0) = crop bounds
    green_lines = find_line_x(img, [0, 255, 0])
    # Yellow lines (BGR: 0,255,255) = search zone
    yellow_lines = find_line_x(img, [0, 255, 255])

    return {
        "file": os.path.basename(diag_path),
        "w": w, "h": h,
        "ktr_red": red_lines,
        "vtl_blue": blue_lines,
        "crop_green": green_lines,
        "search_yellow": yellow_lines,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("crops_dir", default="local_crops_v3", nargs="?")
    parser.add_argument("--crops", nargs="+", type=int, default=None,
                        help="Specific crop numbers to analyze")
    args = parser.parse_args()

    diag_dir = os.path.join(args.crops_dir, "diag")
    if not os.path.isdir(diag_dir):
        print(f"No diag/ folder in {args.crops_dir}")
        sys.exit(1)

    pngs = sorted(glob.glob(os.path.join(diag_dir, "row_*.png")))
    if args.crops:
        crop_set = set(args.crops)
        pngs = [p for p in pngs if any(f"_{c:05d}_" in os.path.basename(p) or
                                        f"row_{c:05d}_" in os.path.basename(p)
                                        for c in crop_set)]

    if not pngs:
        print("No matching diagnostic images found")
        sys.exit(1)

    print(f"{'file':<35} {'w':>4} {'ktr(red)':>10} {'vtl(blue)':>10} "
          f"{'gap':>5} {'crop_L':>7} {'crop_R':>7} {'crop_w':>7} "
          f"{'search_L':>9} {'search_R':>9}")
    print("-" * 120)

    for path in pngs:
        r = analyze_row(path)
        if r is None:
            continue
        ktr = r["ktr_red"][0] if r["ktr_red"] else None
        vtl = r["vtl_blue"][0] if r["vtl_blue"] else None
        gap = (vtl - ktr) if ktr is not None and vtl is not None else None

        crop_l = r["crop_green"][0] if len(r["crop_green"]) >= 1 else None
        crop_r = r["crop_green"][-1] if len(r["crop_green"]) >= 2 else None
        crop_w = (crop_r - crop_l) if crop_l is not None and crop_r is not None else None

        search_l = r["search_yellow"][0] if len(r["search_yellow"]) >= 1 else None
        search_r = r["search_yellow"][-1] if len(r["search_yellow"]) >= 2 else None

        def fmt(v):
            return f"{v:>4}" if v is not None else "   -"

        print(f"{r['file']:<35} {r['w']:>4} "
              f"{fmt(ktr):>10} {fmt(vtl):>10} {fmt(gap):>5} "
              f"{fmt(crop_l):>7} {fmt(crop_r):>7} {fmt(crop_w):>7} "
              f"{fmt(search_l):>9} {fmt(search_r):>9}")


if __name__ == "__main__":
    main()
