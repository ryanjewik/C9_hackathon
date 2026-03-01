"""
Debug ROI alignment: save crops of score + team tag ROIs at various timestamps.
Outputs to /app/outputs/debug_rois/
"""
import os, sys, cv2
sys.path.insert(0, '/app')

from config.settings import ROI_CONFIG

ROI_KEYS = [
    "top_left_score",
    "top_right_score",
    "top_left_team_tag",
    "top_right_team_tag",
]

# Timestamps to sample (seconds into VOD 4)
TIMESTAMPS = [300, 512, 650, 850, 1100, 1400]

VIDEO = "/app/uploads/match_vod_4.mp4"
OUT_DIR = "/app/outputs/debug_rois"
os.makedirs(OUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video: {w}x{h} @ {fps:.2f}fps")

for t_sec in TIMESTAMPS:
    frame_num = int(t_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if not ret:
        print(f"t={t_sec}s: could not read frame")
        continue

    # Save full frame (downscaled for reference)
    small = cv2.resize(frame, (960, 540))
    cv2.imwrite(os.path.join(OUT_DIR, f"t{t_sec}s_full.jpg"), small)

    for key in ROI_KEYS:
        roi = ROI_CONFIG.get(key)
        if not roi:
            continue
        rx, ry, rw, rh = roi
        x1 = int(rx * w)
        y1 = int(ry * h)
        x2 = int((rx + rw) * w)
        y2 = int((ry + rh) * h)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            print(f"t={t_sec}s {key}: empty crop")
            continue

        # Save raw crop
        fname = f"t{t_sec}s_{key}.png"
        cv2.imwrite(os.path.join(OUT_DIR, fname), crop)

        # Also save an upscaled version for easier viewing
        scale = max(4, 100 // max(crop.shape[:2]))
        big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(OUT_DIR, f"t{t_sec}s_{key}_big.png"), big)

        print(f"t={t_sec}s {key}: crop {crop.shape[1]}x{crop.shape[0]}px  saved")

    # Also save the top-center HUD strip (wider context around scores + tags)
    # x: 30%-70% of width, y: 0-7% of height
    ctx_x1, ctx_y1 = int(0.30 * w), 0
    ctx_x2, ctx_y2 = int(0.70 * w), int(0.07 * h)
    ctx_crop = frame[ctx_y1:ctx_y2, ctx_x1:ctx_x2]
    cv2.imwrite(os.path.join(OUT_DIR, f"t{t_sec}s_hud_strip.png"), ctx_crop)
    print(f"t={t_sec}s hud_strip: {ctx_crop.shape[1]}x{ctx_crop.shape[0]}px")
    print()

cap.release()
print(f"\nAll saved to {OUT_DIR}")
print(f"Files: {sorted(os.listdir(OUT_DIR))}")
