"""
Compare what pixels look like around ktr for crops 205, 46, and 154.
All three have the same ability icon but OCR treats them differently.
"""
import cv2
import numpy as np
import os
import glob

crops_dir = "local_crops_v3"

targets = {
    46:  {"ktr": 416, "vtl": 434, "gap": 18},
    154: {"ktr": 367, "vtl": 412, "gap": 45},
    205: {"ktr": 394, "vtl": 410, "gap": 16},
}

diag_dir = os.path.join(crops_dir, "diag")

for crop_num, info in sorted(targets.items()):
    pattern = os.path.join(diag_dir, f"row_{crop_num:05d}_*.png")
    matches = glob.glob(pattern)
    if not matches:
        print(f"crop {crop_num}: no diag image found")
        continue
    
    img = cv2.imread(matches[0])
    h, w = img.shape[:2]
    ktr = info["ktr"]
    vtl = info["vtl"]
    gap = info["gap"]
    
    # Get the original row (the diag has annotation lines drawn over it)
    # Use the raw row from the non-annotated version
    # Actually, let's look at the HSV around the ktr region
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    print(f"\n=== Crop {crop_num} (ktr={ktr}, vtl={vtl}, gap={gap}) ===")
    print(f"  Row size: {w}x{h}")
    
    # Look at a 30px window around ktr
    x_start = max(0, ktr - 30)
    x_end = min(w, ktr + 30)
    
    # For each column, compute avg HSV and brightness
    print(f"  Pixels around ktr ({x_start}-{x_end}):")
    print(f"  {'x':>5} {'avgH':>5} {'avgS':>5} {'avgV':>5} {'bright%':>8} {'teal%':>7} {'red%':>6}")
    
    for x in range(x_start, x_end, 2):  # every 2 pixels
        col_hsv = hsv[:, x:x+2, :]
        col_bgr = img[:, x:x+2, :]
        
        avg_h = np.mean(col_hsv[:,:,0])
        avg_s = np.mean(col_hsv[:,:,1])
        avg_v = np.mean(col_hsv[:,:,2])
        
        gray = cv2.cvtColor(col_bgr, cv2.COLOR_BGR2GRAY)
        bright_pct = np.sum(gray >= 170) / gray.size
        
        # Teal check
        teal = cv2.inRange(col_hsv, np.array([75, 50, 80]), np.array([115, 255, 255]))
        teal_pct = np.count_nonzero(teal) / teal.size
        
        # Red check
        red1 = cv2.inRange(col_hsv, np.array([0, 80, 100]), np.array([15, 255, 255]))
        red2 = cv2.inRange(col_hsv, np.array([165, 80, 100]), np.array([179, 255, 255]))
        red_pct = (np.count_nonzero(red1) + np.count_nonzero(red2)) / red1.size
        
        marker = ""
        if x == ktr or x == ktr + 1:
            marker = " <-- ktr"
        elif x == vtl or x == vtl + 1:
            marker = " <-- vtl"
        
        print(f"  {x:>5} {avg_h:>5.1f} {avg_s:>5.1f} {avg_v:>5.1f} "
              f"{bright_pct:>7.0%} {teal_pct:>6.0%} {red_pct:>5.0%}{marker}")
