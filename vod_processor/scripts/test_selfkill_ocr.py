#!/usr/bin/env python3
"""Quick test: does OCR now find two names on self-kill rows?"""
import cv2
import sys
sys.path.insert(0, '/app')
from vod_processor.app.services.ocr.ocr_engine import OCREngine

ocr = OCREngine()
ocr._lazy_init()

for row_idx in range(4):
    img = cv2.imread(f'/app/outputs/debug_selfkill_row{row_idx}_2063s.png')
    if img is None:
        continue
    print(f'=== Row {row_idx} ===')
    results = ocr.read_text_multipass(img, min_confidence=0.2, strategies=['contrast'])
    for r in results:
        print(f'  text="{r.text}" conf={r.confidence:.2f} bbox={r.bbox}')
    print()
