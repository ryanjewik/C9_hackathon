"""Fine-grained column analysis of a failing row."""
import cv2
import numpy as np

# A row where gap detection fails: frame014055 row3 
img = cv2.imread(r'e:\cloud9_hackathon\vod_processor\outputs\crops\rows\frame014055_t468968ms_row3.png')
h, w = img.shape[:2]
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

print(f"Row: {w}x{h}")
print(f"{'col':>5} {'H_med':>6} {'S_med':>6} {'V_med':>6} {'S_max':>6} {'teal':>6} {'orange':>6}")

# Column-by-column at 10px intervals
for x in range(0, w, 10):
    x1 = min(x + 10, w)
    s = hsv[:, x:x1]
    h_vals = s[:, :, 0].flatten()
    s_vals = s[:, :, 1].flatten()
    v_vals = s[:, :, 2].flatten()
    
    # Check various thresholds
    teal_s50 = np.sum((h_vals >= 75) & (h_vals <= 115) & (s_vals >= 50) & (v_vals >= 80))
    teal_s30 = np.sum((h_vals >= 75) & (h_vals <= 120) & (s_vals >= 30) & (v_vals >= 60))
    orange_s80 = np.sum((h_vals <= 25) & (s_vals >= 80) & (v_vals >= 100))
    orange_s50 = np.sum((h_vals <= 30) & (s_vals >= 50) & (v_vals >= 60))
    total = len(h_vals)
    
    print(f"{x:5d} {np.median(h_vals):6.0f} {np.median(s_vals):6.0f} {np.median(v_vals):6.0f} "
          f"{np.max(s_vals):6.0f}  t50={teal_s50*100/total:4.0f}% t30={teal_s30*100/total:4.0f}%  "
          f"o80={orange_s80*100/total:4.0f}% o50={orange_s50*100/total:4.0f}%")

print("\n--- Also checking frame014510 row0 (one that works better) ---")
img2 = cv2.imread(r'e:\cloud9_hackathon\vod_processor\outputs\crops\rows\frame014510_t484150ms_row0.png')
h2, w2 = img2.shape[:2]
hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
print(f"Row: {w2}x{h2}")
print(f"{'col':>5} {'H_med':>6} {'S_med':>6} {'V_med':>6} {'S_max':>6}")
for x in range(0, w2, 10):
    x1 = min(x + 10, w2)
    s = hsv2[:, x:x1]
    h_vals = s[:, :, 0].flatten()
    s_vals = s[:, :, 1].flatten()
    v_vals = s[:, :, 2].flatten()
    teal_s50 = np.sum((h_vals >= 75) & (h_vals <= 115) & (s_vals >= 50) & (v_vals >= 80))
    teal_s30 = np.sum((h_vals >= 75) & (h_vals <= 120) & (s_vals >= 30) & (v_vals >= 60))
    orange_s80 = np.sum((h_vals <= 25) & (s_vals >= 80) & (v_vals >= 100))
    orange_s50 = np.sum((h_vals <= 30) & (s_vals >= 50) & (v_vals >= 60))
    total = len(h_vals)
    print(f"{x:5d} {np.median(h_vals):6.0f} {np.median(s_vals):6.0f} {np.median(v_vals):6.0f} "
          f"{np.max(s_vals):6.0f}  t50={teal_s50*100/total:4.0f}% t30={teal_s30*100/total:4.0f}%  "
          f"o80={orange_s80*100/total:4.0f}% o50={orange_s50*100/total:4.0f}%")
