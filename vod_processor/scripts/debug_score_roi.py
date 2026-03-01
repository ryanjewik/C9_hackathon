"""Debug: extract score ROI crops from VOD 4 at timestamps where
round changes should have been detected but weren't."""
import cv2, os, sys, numpy as np

path = "/app/uploads/match_vod_4.mp4"
cap = cv2.VideoCapture(path)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video: {w}x{h} @ {fps:.2f}fps")

# Score ROI coords from settings.py (normalized to full frame)
left_x = int(0.417 * w)
left_y = int(0.009 * h)
score_w = int(0.036 * w)
score_h = int(0.042 * h)
right_x = int(0.555 * w)
right_y = int(0.009 * h)

print(f"Left score ROI: x={left_x}, y={left_y}, w={score_w}, h={score_h}")
print(f"Right score ROI: x={right_x}, y={right_y}, w={score_w}, h={score_h}")

# Also extract a wider region around the scores to see context
ctx_x = int(0.38 * w)
ctx_y = 0
ctx_w = int(0.24 * w)
ctx_h = int(0.08 * h)

out_dir = "/app/outputs/score_debug"
os.makedirs(out_dir, exist_ok=True)

# Timestamps: first round change detected at 512s (3-0), then nothing.
# Check timestamps where kills happen in what should be later rounds
timestamps = [512, 650, 760, 855, 1005, 1100, 1230, 1320, 1440, 1550]

for t_sec in timestamps:
    frame_num = int(t_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if not ret:
        print(f"t={t_sec}s: could not read frame")
        continue
    
    left_roi = frame[left_y:left_y+score_h, left_x:left_x+score_w]
    right_roi = frame[right_y:right_y+score_h, right_x:right_x+score_w]
    context_roi = frame[ctx_y:ctx_y+ctx_h, ctx_x:ctx_x+ctx_w]
    
    # Save crops
    cv2.imwrite(f"{out_dir}/t{t_sec}_left.png", left_roi)
    cv2.imwrite(f"{out_dir}/t{t_sec}_right.png", right_roi)
    cv2.imwrite(f"{out_dir}/t{t_sec}_context.png", context_roi)
    
    # Also try OCR right here
    try:
        import easyocr
        ocr = easyocr.Reader(['en'], gpu=True, verbose=False)
        
        # Scale up 3x like the detector does
        left_scaled = cv2.resize(left_roi, (left_roi.shape[1]*3, left_roi.shape[0]*3), interpolation=cv2.INTER_CUBIC)
        right_scaled = cv2.resize(right_roi, (right_roi.shape[1]*3, right_roi.shape[0]*3), interpolation=cv2.INTER_CUBIC)
        
        lr = ocr.readtext(left_scaled, allowlist='0123456789')
        rr = ocr.readtext(right_scaled, allowlist='0123456789')
        
        lt = lr[0][1] if lr else "?"
        lc = lr[0][2] if lr else 0
        rt = rr[0][1] if rr else "?"
        rc = rr[0][2] if rr else 0
        
        print(f"t={t_sec}s: LEFT={lt}(conf={lc:.2f})  RIGHT={rt}(conf={rc:.2f})")
    except Exception as e:
        print(f"t={t_sec}s: OCR error: {e}")
        # Only init once
        break

cap.release()
print(f"\nScore debug images saved to {out_dir}/")
