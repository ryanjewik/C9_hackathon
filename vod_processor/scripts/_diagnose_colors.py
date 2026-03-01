"""Deep-dive: what colors are actually in specific row regions."""
import cv2
import numpy as np
import os

TEAM_COLORS = {
    "teal":   {"lower": (75, 50, 80),  "upper": (115, 255, 255)},
    "orange": {"lower": (0, 80, 100),  "upper": (25, 255, 255),
               "lower2": (160, 80, 100), "upper2": (180, 255, 255)},
}

rows_dir = r'e:\cloud9_hackathon\vod_processor\outputs\crops\rows'

# Check a row that has a wide gap problem
targets = ['frame014055_t468968ms_row3.png', 'frame014055_t468968ms_row4.png',
           'frame012655_t422255ms_row0.png']

for fname in targets:
    fpath = os.path.join(rows_dir, fname)
    row_img = cv2.imread(fpath)
    if row_img is None:
        print(f"SKIP {fname}")
        continue

    h, w = row_img.shape[:2]
    hsv = cv2.cvtColor(row_img, cv2.COLOR_BGR2HSV)

    print(f"\n{'='*80}")
    print(f"FILE: {fname}  ({w}x{h})")

    # Sample HSV values at different horizontal positions
    # Check 10 evenly-spaced vertical-strip samples across the row
    for pct in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
        x = int(w * pct / 100)
        strip = hsv[:, max(0,x-2):min(w,x+3)]  # 5px wide strip
        mean_h = np.mean(strip[:,:,0])
        mean_s = np.mean(strip[:,:,1])
        mean_v = np.mean(strip[:,:,2])

        # Check which color it matches
        teal = 75 <= mean_h <= 115 and mean_s >= 50 and mean_v >= 80
        orange = (0 <= mean_h <= 25 or 160 <= mean_h <= 180) and mean_s >= 80 and mean_v >= 100
        dark = mean_v < 60
        label = "TEAL" if teal else "ORANGE" if orange else "DARK" if dark else "other"

        print(f"  x={x:3d} ({pct:2d}%): H={mean_h:5.1f} S={mean_s:5.1f} V={mean_v:5.1f}  -> {label}")

    # Also check: is there a region that's dark/gray (weapon icon area)?
    gray = cv2.cvtColor(row_img, cv2.COLOR_BGR2GRAY)
    print(f"  Gray intensity profile (mean per 10% column slice):")
    for pct in range(0, 100, 10):
        x0 = int(w * pct / 100)
        x1 = int(w * (pct + 10) / 100)
        mean_gray = np.mean(gray[:, x0:x1])
        bar = '#' * int(mean_gray / 10)
        print(f"    {pct:2d}-{pct+10:2d}%: {mean_gray:5.1f} {bar}")
