#!/usr/bin/env python3
"""Test if multiple OCR strategies find both names on self-kill rows."""
import cv2
import sys
sys.path.insert(0, '/app')
from vod_processor.app.services.ocr.ocr_engine import OCREngine

ocr = OCREngine()
ocr._lazy_init()

strategies = ['contrast', 'threshold', 'saturation', 'high_contrast', 'adaptive']

for row_idx in range(4):
    img = cv2.imread(f'/app/outputs/debug_selfkill_row{row_idx}_2063s.png')
    if img is None:
        continue
    print(f'=== Row {row_idx} ===')
    
    # Test each strategy individually first
    for strat in strategies:
        results = ocr.read_text_multipass(img, min_confidence=0.2, strategies=[strat])
        names = [(r.text, r.bbox[0]) for r in results if len(r.text.strip()) >= 3]
        print(f'  {strat:15s}: {names}')
    
    # Test combined (all strategies together)
    results = ocr.read_text_multipass(img, min_confidence=0.2, strategies=strategies)
    names = [(r.text, r.bbox[0], round(r.confidence, 2)) for r in results if len(r.text.strip()) >= 3]
    print(f'  {"ALL COMBINED":15s}: {names}')
    print()
