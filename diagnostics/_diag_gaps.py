"""Diagnose why gap detection fails on actual killfeed rows.

Reads saved row images and prints:
- HSV stats across horizontal slices
- Current color mask coverage
- Detected runs and gaps
"""
import cv2
import numpy as np
import os

ROWS_DIR = r'e:\cloud9_hackathon\vod_processor\outputs\crops\rows'
ICONS_DIR = r'e:\cloud9_hackathon\vod_processor\outputs\crops\icons'

# WIDER ranges matching _extract_weapon_icon (not global TEAM_COLORS)
TEAM_COLORS = {
    "teal": {"lower": (75, 25, 60), "upper": (120, 255, 255)},
    "orange": {
        "lower": (0, 60, 80), "upper": (30, 255, 255),
        "lower2": (155, 60, 80), "upper2": (180, 255, 255),
    },
}


def analyze_row(row_path, icon_path=None):
    row_img = cv2.imread(row_path)
    if row_img is None:
        return
    h, w = row_img.shape[:2]
    hsv = cv2.cvtColor(row_img, cv2.COLOR_BGR2HSV)
    fname = os.path.basename(row_path)

    # Icon width for reference
    icon_w = "N/A"
    if icon_path and os.path.exists(icon_path):
        ic = cv2.imread(icon_path)
        if ic is not None:
            icon_w = ic.shape[1]

    print(f"\n{'='*80}")
    print(f"ROW: {fname}  ({w}x{h})  icon_w={icon_w}")
    print(f"{'='*80}")

    # Build color mask (same as _extract_weapon_icon)
    teal_mask = cv2.inRange(hsv,
        np.array(TEAM_COLORS['teal']['lower']),
        np.array(TEAM_COLORS['teal']['upper']))
    orange_mask1 = cv2.inRange(hsv,
        np.array(TEAM_COLORS['orange']['lower']),
        np.array(TEAM_COLORS['orange']['upper']))
    orange_mask2 = cv2.inRange(hsv,
        np.array(TEAM_COLORS['orange']['lower2']),
        np.array(TEAM_COLORS['orange']['upper2']))
    color_mask_raw = cv2.bitwise_or(teal_mask, cv2.bitwise_or(orange_mask1, orange_mask2))

    # Morphological closing
    close_kw = max(3, int(w * 0.04))
    close_kh = max(1, h // 3)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_kw, close_kh))
    color_mask = cv2.morphologyEx(color_mask_raw, cv2.MORPH_CLOSE, close_kernel)

    # Per-region stats: divide row into 10 horizontal slices
    slice_w = w // 10
    for i in range(10):
        x0 = i * slice_w
        x1 = (i + 1) * slice_w if i < 9 else w
        s = hsv[:, x0:x1]
        h_vals = s[:, :, 0].flatten()
        s_vals = s[:, :, 1].flatten()
        v_vals = s[:, :, 2].flatten()
        tpx = cv2.countNonZero(teal_mask[:, x0:x1])
        opx = cv2.countNonZero(cv2.bitwise_or(orange_mask1[:, x0:x1], orange_mask2[:, x0:x1]))
        total = (x1 - x0) * h
        pct = f"t{tpx*100/total:.0f}% o{opx*100/total:.0f}%"
        print(f"  {i*10:3d}-{(i+1)*10:3d}%: H={np.median(h_vals):5.0f}(+-{np.std(h_vals):.0f}) "
              f"S={np.median(s_vals):5.0f}(+-{np.std(s_vals):.0f}) "
              f"V={np.median(v_vals):5.0f}(+-{np.std(v_vals):.0f})  {pct}")

    # Column density + runs (same logic as detector)
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

    print(f"\n  Color mask coverage: raw={cv2.countNonZero(color_mask_raw)*100/(w*h):.1f}%, closed={cv2.countNonZero(color_mask)*100/(w*h):.1f}%")
    print(f"  Runs ({len(runs)}):")
    for i, (s, e) in enumerate(runs):
        print(f"    run{i}: cols {s}-{e}  (w={e-s}, at {s*100/w:.0f}%-{e*100/w:.0f}%)")

    # Gaps
    best_gap = None
    best_gap_w = 0
    for i in range(len(runs) - 1):
        gl = runs[i][1]
        gr = runs[i + 1][0]
        gw = gr - gl
        gcx = (gl + gr) / 2.0
        in_range = "YES" if (gcx >= w * 0.15 and gcx <= w * 0.85) else "NO"
        print(f"    gap {i}: cols {gl}-{gr} (w={gw}, center={gcx:.0f}={gcx*100/w:.0f}%) interior={in_range}")
        if gcx >= w * 0.15 and gcx <= w * 0.85 and gw > best_gap_w:
            best_gap_w = gw
            best_gap = (gl, gr)

    if best_gap:
        print(f"  => SELECTED GAP: cols {best_gap[0]}-{best_gap[1]} (w={best_gap[1]-best_gap[0]})")
    else:
        print(f"  => FALLBACK (no valid interior gap)")


# Pick a mix of rows: some with good crops, some with bad ones
test_rows = [
    "frame012655_t422255ms_row0",   # icon=62px (might be ok)
    "frame012655_t422255ms_row1",   # icon=76px
    "frame014055_t468968ms_row3",   # icon=325px (way too wide)
    "frame014055_t468968ms_row4",   # icon=292px (way too wide)
    "frame014510_t484150ms_row0",   # multi-kill frame
    "frame014510_t484150ms_row1",
    "frame014510_t484150ms_row3",
    "frame015390_t513513ms_row0",   # late in the sample
    "frame015390_t513513ms_row1",
]

for name in test_rows:
    row_p = os.path.join(ROWS_DIR, name + ".png")
    icon_p = os.path.join(ICONS_DIR, name + "_icon.png")
    if os.path.exists(row_p):
        analyze_row(row_p, icon_p)
    else:
        print(f"\n  MISSING: {name}")
