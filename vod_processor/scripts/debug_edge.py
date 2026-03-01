"""Debug: run killfeed row analysis on video frames to check team_smooth values."""
import cv2
import numpy as np
import sys
sys.path.insert(0, "/app")

cap = cv2.VideoCapture("/app/uploads/match_vod_3.mp4")
if not cap.isOpened():
    print("Cannot open video")
    sys.exit(1)

cap.set(cv2.CAP_PROP_POS_MSEC, 200000)

found = 0
for _ in range(500):
    ret, frame = cap.read()
    if not ret:
        break
    h_f, w_f = frame.shape[:2]
    kf_region = frame[0:int(h_f * 0.35), int(w_f * 0.45):]
    hsv_kf = cv2.cvtColor(kf_region, cv2.COLOR_BGR2HSV)
    sat = hsv_kf[:, :, 1]
    row_sat = sat.mean(axis=1)

    in_band = False
    band_start = 0
    for y in range(len(row_sat)):
        if row_sat[y] > 30 and not in_band:
            band_start = y
            in_band = True
        elif row_sat[y] < 15 and in_band and y - band_start > 20:
            row_img = kf_region[band_start:y, :]
            rh, rw = row_img.shape[:2]
            if 25 < rh < 60 and rw > 200:
                hsv = cv2.cvtColor(row_img, cv2.COLOR_BGR2HSV)
                H = hsv[:, :, 0]; S = hsv[:, :, 1]; V = hsv[:, :, 2]
                valid = (S >= 15) & (V >= 40)
                n_val = np.maximum(valid.sum(0).astype(float), 1.0)
                orange_px = valid & ((H <= 30) | (H >= 150))
                teal_px = valid & (H >= 70) & (H <= 120)
                o_frac = orange_px.sum(0).astype(float) / n_val
                t_frac = teal_px.sum(0).astype(float) / n_val
                low = valid.sum(0) < max(2, rh * 0.2)
                o_frac[low] = 0.0; t_frac[low] = 0.0
                ks = max(5, int(rw * 0.025) | 1)
                kern = np.ones(ks) / ks
                o_sm = np.convolve(o_frac, kern, "same")
                t_sm = np.convolve(t_frac, kern, "same")
                dom = o_sm - t_sm
                crossings = []
                for x in range(1, rw):
                    if dom[x - 1] * dom[x] < 0:
                        crossings.append(x)
                SWING_W = max(10, int(rw * 0.06))
                bc = None; bs = -1
                for cr in crossings:
                    if cr < rw * 0.15 or cr > rw * 0.85: continue
                    la = float(np.mean(dom[max(0, cr - SWING_W):cr]))
                    ra = float(np.mean(dom[cr:min(rw, cr + SWING_W)]))
                    s = abs(ra - la)
                    if s > bs: bs = s; bc = cr
                if bc is None:
                    in_band = False; continue
                team = np.maximum(o_frac, t_frac)
                ek = max(11, int(rw * 0.06) | 1)
                ekern = np.ones(ek) / ek
                ts = np.convolve(team, ekern, "same")
                print(f"\n=== Row {rw}x{rh} crossing={bc} ===")
                lo = max(0, bc - 120); hi = min(rw, bc + 80)
                for x in range(lo, hi, 4):
                    marker = " <-- CROSS" if abs(x - bc) < 3 else ""
                    above = "*" if ts[x] >= 0.40 else " "
                    print(f"  col {x:3d}: ts={ts[x]:.3f} raw={team[x]:.3f} {above}{marker}")
                found += 1
                if found >= 3: break
            in_band = False
        elif row_sat[y] < 15 and in_band:
            in_band = False
    if found >= 3: break

cap.release()
if found == 0:
    print("No rows found")
