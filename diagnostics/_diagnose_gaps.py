"""Diagnose what _extract_weapon_icon gap detection actually sees on real rows."""
import cv2
import numpy as np
import os

TEAM_COLORS = {
    "teal":   {"lower": (75, 25, 60),  "upper": (120, 255, 255)},
    "orange": {"lower": (0, 60, 80),  "upper": (30, 255, 255),
               "lower2": (155, 60, 80), "upper2": (180, 255, 255)},
}

rows_dir = r'e:\cloud9_hackathon\vod_processor\outputs\crops\rows'

# Pick rows from frames that have multiple killfeed entries (real gameplay)
targets = [f for f in sorted(os.listdir(rows_dir))
           if 'frame012655' in f or 'frame012715' in f
           or 'frame014055' in f or 'frame014060' in f
           or 'frame014320' in f or 'frame014330' in f]

if not targets:
    # Fallback to first 10
    targets = sorted(os.listdir(rows_dir))[:10]

for fname in targets:
    fpath = os.path.join(rows_dir, fname)
    row_img = cv2.imread(fpath)
    if row_img is None:
        continue

    h, w = row_img.shape[:2]
    hsv = cv2.cvtColor(row_img, cv2.COLOR_BGR2HSV)

    teal_mask = cv2.inRange(hsv, np.array(TEAM_COLORS['teal']['lower']),
                            np.array(TEAM_COLORS['teal']['upper']))
    orange_mask1 = cv2.inRange(hsv, np.array(TEAM_COLORS['orange']['lower']),
                               np.array(TEAM_COLORS['orange']['upper']))
    orange_mask2 = cv2.inRange(hsv, np.array(TEAM_COLORS['orange']['lower2']),
                               np.array(TEAM_COLORS['orange']['upper2']))
    color_mask_raw = cv2.bitwise_or(teal_mask, cv2.bitwise_or(orange_mask1, orange_mask2))

    # Count raw color pixels
    teal_pct = cv2.countNonZero(teal_mask) / (h * w) * 100
    orange_pct = (cv2.countNonZero(orange_mask1) + cv2.countNonZero(orange_mask2)) / (h * w) * 100

    # Morphological closing
    close_kw = max(3, int(w * 0.04))
    close_kh = max(1, h // 3)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_kw, close_kh))
    color_mask = cv2.morphologyEx(color_mask_raw, cv2.MORPH_CLOSE, close_kernel)

    # Column density
    col_density = np.sum(color_mask > 0, axis=0).astype(np.float64) / max(1, h)
    ks = max(5, int(w * 0.03) | 1)
    col_smooth = np.convolve(col_density, np.ones(ks) / ks, mode='same')

    DENSITY_THRESH = 0.15
    is_name = col_smooth >= DENSITY_THRESH

    runs = []
    in_run = False
    run_start = 0
    for x in range(w):
        if is_name[x] and not in_run:
            in_run = True
            run_start = x
        elif not is_name[x] and in_run:
            in_run = False
            runs.append((run_start, x))
    if in_run:
        runs.append((run_start, w))

    MIN_RUN_W = max(8, int(w * 0.02))
    runs = [(s, e) for s, e in runs if (e - s) >= MIN_RUN_W]

    # Find gaps
    gaps = []
    for i in range(len(runs) - 1):
        gl = runs[i][1]
        gr = runs[i + 1][0]
        gw = gr - gl
        gcx = (gl + gr) / 2.0
        interior = w * 0.15 <= gcx <= w * 0.85
        gaps.append((gl, gr, gw, gcx, interior))

    # Best gap
    best_gap = None
    best_gap_w = 0
    for gl, gr, gw, gcx, interior in gaps:
        if interior and gw > best_gap_w:
            best_gap_w = gw
            best_gap = (gl, gr)

    print(f"\n{'='*80}")
    print(f"FILE: {fname}  ({w}x{h})")
    print(f"  Teal: {teal_pct:.1f}%  Orange: {orange_pct:.1f}%")
    print(f"  Close kernel: {close_kw}x{close_kh}, Smooth ks: {ks}")
    print(f"  Runs ({len(runs)}):")
    for i, (s, e) in enumerate(runs):
        pct_s = s / w * 100
        pct_e = e / w * 100
        print(f"    [{i}] cols {s}-{e} (w={e-s}, {pct_s:.0f}%-{pct_e:.0f}% of row)")
    print(f"  Gaps ({len(gaps)}):")
    for gl, gr, gw, gcx, interior in gaps:
        sel = " <-- SELECTED" if best_gap and gl == best_gap[0] else ""
        print(f"    cols {gl}-{gr} (w={gw}, center={gcx:.0f}={gcx/w*100:.0f}%, interior={interior}){sel}")
    if best_gap:
        pad_x = max(2, int(best_gap_w * 0.08))
        x0 = max(0, best_gap[0] - pad_x)
        x1 = min(w, best_gap[1] + pad_x)
        print(f"  CROP: x={x0}-{x1} (w={x1-x0})")
    else:
        print(f"  CROP: FALLBACK (center 80px)")
