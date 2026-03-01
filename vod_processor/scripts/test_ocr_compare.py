#!/usr/bin/env python3
"""Compare threshold vs contrast on normal kills AND self-kills."""
import cv2
import sys
import numpy as np
sys.path.insert(0, '/app')
from vod_processor.app.services.ocr.ocr_engine import OCREngine

ocr = OCREngine()
ocr._lazy_init()

cap = cv2.VideoCapture('/app/uploads/match_vod_6.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)

# Killfeed ROI params
KX, KY, KW, KH = 0.690, 0.092, 0.305, 0.318
ROW_H = 38

# Test timestamps: mix of normal kills and self-kills
# Normal: t=1424s (R10 start), t=1907s (R13 start), t=486s (R1)
# Self-kill: t=2063s (R14 spike)
test_cases = [
    (486, "R1 normal kill"),
    (1424, "R10 normal kill"),
    (1907, "R13 normal kill"),
    (2063, "R14 self-kill (spike)"),
]

for ts, label in test_cases:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(ts * fps))
    ret, frame = cap.read()
    if not ret:
        continue
    fh, fw = frame.shape[:2]
    kf = frame[int(KY*fh):int(KY*fh)+int(KH*fh), int(KX*fw):int(KX*fw)+int(KW*fw)]

    print(f"=== {label} (t={ts}s) ===")
    for row_idx in range(min(3, kf.shape[0] // ROW_H)):
        row_img = kf[row_idx*ROW_H:(row_idx+1)*ROW_H, :]

        res_c = ocr.read_text_multipass(row_img, min_confidence=0.2, strategies=['contrast'])
        names_c = [(r.text, r.bbox[0]) for r in res_c if len(r.text.strip()) >= 3]

        res_t = ocr.read_text_multipass(row_img, min_confidence=0.2, strategies=['threshold'])
        names_t = [(r.text, r.bbox[0]) for r in res_t if len(r.text.strip()) >= 3]

        print(f"  Row {row_idx}:")
        print(f"    contrast : {names_c}")
        print(f"    threshold: {names_t}")
    print()

cap.release()
