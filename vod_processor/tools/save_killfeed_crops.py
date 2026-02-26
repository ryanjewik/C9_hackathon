"""
Utility: save_killfeed_crops.py

Usage:
    python tools/save_killfeed_crops.py --video PATH_TO_VIDEO --out ./crops \
        [--sample-rate 5] [--max-crops 200] [--model MODEL_PTH --labels CLASS_NAMES_JSON]

This script:
- Loads the killfeed ROI from settings
- Segments rows using the same logic as KillfeedDetector
- Extracts weapon icon crops with the detector heuristic
- Saves row images and icon crops into the output folder
- Optionally runs the IconClassifier if model+labels provided

Run from the project root (parent of vod_processor) or this file will add the project
parent to `sys.path` automatically so imports work.
"""

import os
import sys
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

# Ensure project parent is on sys.path so we can import package modules
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent  # vod_processor/
VOD_PROC_DIR = str(PROJECT_ROOT)
PARENT = PROJECT_ROOT.parent  # e.g., e:/cloud9_hackathon
# Ensure both the vod_processor package directory and its parent are on sys.path.
# This allows imports that use either `vod_processor.*` or absolute `app.*`.
if VOD_PROC_DIR not in sys.path:
    sys.path.insert(0, VOD_PROC_DIR)
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

# Imports from project
try:
    from vod_processor.app.services.processing import vod_processor as vmod
    from vod_processor.config.settings import ROI_CONFIG, TEAM_COLORS
except Exception as e:
    print("Failed to import project modules:", e)
    print("Make sure you're running this script from the 'vod_processor' folder and that the package imports are resolvable.")
    raise

# Optional classifier
IconClassifier = None


def make_debug_image(row_img: np.ndarray) -> np.ndarray:
    """Create a debug visualization matching the hue-gap contour _extract_weapon_icon algorithm."""
    h, w = row_img.shape[:2]
    hsv = cv2.cvtColor(row_img, cv2.COLOR_BGR2HSV)

    H_ch = hsv[:, :, 0]
    S_ch = hsv[:, :, 1]
    V_ch = hsv[:, :, 2]

    # ── Hue-transition detection (same as detector) ──
    MIN_SAT = 15
    MIN_VAL = 40
    valid = (S_ch >= MIN_SAT) & (V_ch >= MIN_VAL)
    n_valid = valid.sum(axis=0).astype(np.float64)
    n_valid_safe = np.maximum(n_valid, 1.0)

    orange_px = valid & ((H_ch <= 30) | (H_ch >= 150))
    orange_frac = orange_px.sum(axis=0).astype(np.float64) / n_valid_safe

    teal_px = valid & (H_ch >= 70) & (H_ch <= 120)
    teal_frac = teal_px.sum(axis=0).astype(np.float64) / n_valid_safe

    low_valid = n_valid < max(2, h * 0.2)
    orange_frac[low_valid] = 0.0
    teal_frac[low_valid] = 0.0

    ks = max(5, int(w * 0.025) | 1)
    kernel = np.ones(ks) / ks
    orange_sm = np.convolve(orange_frac, kernel, mode='same')
    teal_sm = np.convolve(teal_frac, kernel, mode='same')
    dominance = orange_sm - teal_sm

    # Find zero-crossings
    crossings = []
    for x in range(1, w):
        if dominance[x - 1] * dominance[x] < 0:
            crossings.append(x)
        elif dominance[x - 1] != 0 and dominance[x] == 0:
            crossings.append(x)

    # Pick crossing with the largest dominance swing (same as detector)
    SWING_WINDOW = max(10, int(w * 0.06))
    best_cross = None
    best_swing = -1.0
    for cr in crossings:
        if cr < w * 0.15 or cr > w * 0.85:
            continue
        left_avg = float(np.mean(dominance[max(0, cr - SWING_WINDOW):cr]))
        right_avg = float(np.mean(dominance[cr:min(w, cr + SWING_WINDOW)]))
        swing = abs(right_avg - left_avg)
        if swing > best_swing:
            best_swing = swing
            best_cross = cr

    # ── Dominance-gradient detection (mirrors _extract_weapon_icon step 4) ──
    left_bound = right_bound = best_cross if best_cross is not None else w // 2
    weapon_left_col = None
    weapon_right_col = None

    if best_cross is not None:
        dom_grad = np.abs(np.gradient(dominance))
        gk = 15
        g_kernel = np.ones(gk) / gk
        grad_smooth = np.convolve(dom_grad, g_kernel, mode='same')

        peak_window = 40
        peak_lo = max(0, best_cross - peak_window)
        peak_hi = min(len(grad_smooth), best_cross + peak_window)
        peak_grad = float(grad_smooth[peak_lo:peak_hi].max())
        GRAD_THRESH = max(peak_grad * 0.04, 0.001)
        CONSEC_DROP = 6
        safe_left = int(w * 0.08)
        safe_right = int(w * 0.92)

        # Walk LEFT from crossing
        weapon_left_col = safe_left
        consec = 0
        for x in range(best_cross - 1, safe_left - 1, -1):
            if grad_smooth[x] < GRAD_THRESH:
                consec += 1
                if consec >= CONSEC_DROP:
                    weapon_left_col = x + CONSEC_DROP
                    break
            else:
                consec = 0

        # Walk RIGHT from crossing
        weapon_right_col = safe_right
        consec = 0
        for x in range(best_cross + 1, safe_right + 1):
            if grad_smooth[x] < GRAD_THRESH:
                consec += 1
                if consec >= CONSEC_DROP:
                    weapon_right_col = x - CONSEC_DROP
                    break
            else:
                consec = 0

        # Extend gradient zone using proportional extension
        grad_gap = weapon_right_col - weapon_left_col
        extend_left = max(30, int(grad_gap * 1.0))
        extend_right = max(25, int(grad_gap * 0.8))
        ext_left = max(safe_left, weapon_left_col - extend_left)
        ext_right = min(safe_right, weapon_right_col + extend_right)

        left_bound = ext_left
        right_bound = ext_right

        crop_w = right_bound - left_bound
        MIN_CROP = 50
        MAX_CROP = int(w * 0.28)
        if crop_w < MIN_CROP:
            mid = best_cross
            left_bound = max(0, mid - MIN_CROP // 2)
            right_bound = min(w, left_bound + MIN_CROP)
        elif crop_w > MAX_CROP:
            mid = (left_bound + right_bound) // 2
            left_bound = max(0, mid - MAX_CROP // 2)
            right_bound = min(w, left_bound + MAX_CROP)

        pad = max(6, int((right_bound - left_bound) * 0.10))
        left_bound = max(0, left_bound - pad)
        right_bound = min(w, right_bound + pad)

    # ── Build debug canvas: [original+overlay] [dominance chart] [hue map] ──
    # Row 1: original with crop overlay
    overlay = row_img.copy()
    if best_cross is not None:
        cv2.rectangle(overlay, (left_bound, 0), (right_bound, h), (0, 0, 255), 2)
        cv2.line(overlay, (best_cross, 0), (best_cross, h), (0, 255, 255), 1)
        cv2.putText(overlay, "EDGE", (left_bound + 2, h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
    else:
        cv2.putText(overlay, "NO CROSSING", (w // 2 - 30, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

    # Row 2: dominance signal chart
    chart_h = max(40, h)
    chart = np.zeros((chart_h, w, 3), dtype=np.uint8)
    mid_y = chart_h // 2
    cv2.line(chart, (0, mid_y), (w - 1, mid_y), (80, 80, 80), 1)
    max_abs = max(np.abs(dominance).max(), 0.01)
    for x in range(w - 1):
        y1 = int(mid_y - (dominance[x] / max_abs) * (mid_y - 2))
        y2 = int(mid_y - (dominance[x + 1] / max_abs) * (mid_y - 2))
        color = (0, 128, 255) if dominance[x] > 0 else (255, 200, 0)
        cv2.line(chart, (x, y1), (x + 1, y2), color, 1)
    for cr in crossings:
        cv2.line(chart, (cr, 0), (cr, chart_h - 1), (0, 255, 255), 1)
    if best_cross is not None:
        cv2.rectangle(chart, (left_bound, 0), (right_bound, chart_h - 1), (0, 0, 255), 1)

    # Row 3: hue classification map with edge boundary markers
    hue_map = np.zeros((h, w, 3), dtype=np.uint8)
    hue_map[orange_px.astype(bool)] = (0, 128, 255)  # BGR: orange
    hue_map[teal_px.astype(bool)] = (255, 200, 0)    # BGR: cyan/teal
    # Draw detected edge boundaries as green vertical lines
    if weapon_left_col is not None:
        cv2.line(hue_map, (weapon_left_col, 0), (weapon_left_col, h), (0, 255, 0), 1)
    if weapon_right_col is not None:
        cv2.line(hue_map, (weapon_right_col, 0), (weapon_right_col, h), (0, 255, 0), 1)

    # Resize all to same height
    target_h = max(h, 40)
    overlay_r = cv2.resize(overlay, (w, target_h))
    chart_r = cv2.resize(chart, (w, target_h))
    hue_r = cv2.resize(hue_map, (w, target_h))

    cv2.putText(overlay_r, "row+crop", (2, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    cv2.putText(chart_r, f"dominance (crossings={len(crossings)})", (2, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    cv2.putText(hue_r, "hue pixels (orange/teal)", (2, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    debug_img = np.vstack([overlay_r, chart_r, hue_r])
    return debug_img


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, help="Path to VOD video file")
    p.add_argument("--out", default="./crops", help="Output folder for saved crops")
    p.add_argument("--sample-rate", type=int, default=5, help="Sample every N frames")
    p.add_argument("--max-crops", type=int, default=500, help="Stop after saving this many crops (0 = no limit)")
    p.add_argument("--model", default=None, help="Optional path to model .pth for IconClassifier")
    p.add_argument("--labels", default=None, help="JSON file with class names (list)")
    p.add_argument("--device", default="cpu", help="torch device (cpu or cuda)")
    p.add_argument("--debug", action="store_true", help="Save debug visualizations of gap detection")
    p.add_argument("--skip-seconds", type=float, default=0, help="Skip this many seconds from the start of the video")
    return p.parse_args()


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _mask_from_range(hsv, lo, hi):
    lo = np.array(lo, dtype=np.uint8)
    hi = np.array(hi, dtype=np.uint8)
    return cv2.inRange(hsv, lo, hi)


def is_kill_event(row_img: np.ndarray) -> bool:
    """Heuristic: return True if row_img looks like a killfeed event.

    Checks for presence of team-colored regions (teal/orange/red/white) in
    the left/right areas of the row. This reduces false saves when rows are
    empty or part of background.
    """
    if row_img is None or row_img.size == 0:
        return False

    h, w = row_img.shape[:2]
    hsv = cv2.cvtColor(row_img, cv2.COLOR_BGR2HSV)

    # Check left and right quarters for team color presence
    left_slice = hsv[:, : max(1, w // 4)]
    right_slice = hsv[:, max(0, w - w // 4):]

    def slice_has_color(slc) -> bool:
        # Teal
        teal = TEAM_COLORS.get('teal')
        if teal:
            m = _mask_from_range(slc, teal['lower'], teal['upper'])
            if int(cv2.countNonZero(m)) > (slc.shape[0] * slc.shape[1]) * 0.005:
                return True
        # Orange (may have dual ranges)
        orange = TEAM_COLORS.get('orange')
        if orange:
            m1 = _mask_from_range(slc, orange['lower'], orange['upper'])
            count = int(cv2.countNonZero(m1))
            if 'lower2' in orange:
                m2 = _mask_from_range(slc, orange['lower2'], orange['upper2'])
                count += int(cv2.countNonZero(m2))
            if count > (slc.shape[0] * slc.shape[1]) * 0.005:
                return True
        # Red
        red = TEAM_COLORS.get('red')
        if red:
            c = 0
            if 'lower1' in red:
                c += int(cv2.countNonZero(_mask_from_range(slc, red['lower1'], red['upper1'])))
            if 'lower2' in red:
                c += int(cv2.countNonZero(_mask_from_range(slc, red['lower2'], red['upper2'])))
            if c > (slc.shape[0] * slc.shape[1]) * 0.005:
                return True
        # White - name text / separators
        white = TEAM_COLORS.get('white')
        if white:
            m = _mask_from_range(slc, white['lower'], white['upper'])
            if int(cv2.countNonZero(m)) > (slc.shape[0] * slc.shape[1]) * 0.008:
                return True
        return False

    left_has = slice_has_color(left_slice)
    right_has = slice_has_color(right_slice)

    # If either side shows team color, it's likely a killfeed event
    return left_has or right_has


def main():
    args = parse_args()
    video_path = args.video
    out_dir = Path(args.out).resolve()
    sample_rate = max(1, args.sample_rate)
    max_crops = args.max_crops

    ensure_dir(out_dir)
    rows_dir = out_dir / "rows"
    icons_dir = out_dir / "icons"
    debug_dir = out_dir / "debug"
    ensure_dir(rows_dir)
    ensure_dir(icons_dir)
    if args.debug:
        ensure_dir(debug_dir)

    # Optionally load classifier
    classifier = None
    if args.model and args.labels:
        try:
            from vod_processor.app.services.processing.weapon_classifier import IconClassifier
            classifier = IconClassifier(args.model, args.labels, device=args.device)
            print("Loaded IconClassifier", args.model)
        except Exception as e:
            print("Failed to load IconClassifier:", e)
            classifier = None

    # Create detector instance (we'll reuse segmentation logic)
    detector = vmod.KillfeedDetector("killfeed", target_fps=10.0)
    if classifier is not None:
        detector.set_weapon_classifier(classifier)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Compute killfeed ROI in pixels
    killfeed_roi_norm = ROI_CONFIG.get("killfeed")
    if not killfeed_roi_norm:
        raise RuntimeError("killfeed ROI not found in ROI_CONFIG")
    x, y, w, h = vmod.roi_to_px(frame_w, frame_h, killfeed_roi_norm)

    print(f"Video {frame_w}x{frame_h} @ {fps:.2f}fps. Killfeed ROI px: {x},{y},{w},{h}")

    # Skip ahead in video if requested
    skip_seconds = args.skip_seconds
    if skip_seconds > 0:
        skip_frame = int(skip_seconds * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, skip_frame)
        print(f"Skipping to {skip_seconds:.0f}s (frame {skip_frame})")
        frame_idx = skip_frame
    else:
        frame_idx = 0

    saved = 0
    saved_meta = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_rate != 0:
            frame_idx += 1
            continue

        t_ms = (frame_idx / fps) * 1000.0
        roi_frame = vmod.crop(frame, (x, y, w, h))
        if roi_frame.size == 0:
            frame_idx += 1
            continue

        # Segment rows using detector logic (fixed segmentation)
        rows = detector._segment_rows_fixed(roi_frame)
        for actual_row_idx, y_start, y_end, row_img in rows:
            # Only save rows/icons that look like a killfeed event
            if not is_kill_event(row_img):
                continue

            # Save row image for inspection
            ts = int(t_ms)
            row_filename = f"frame{frame_idx:06d}_t{ts}ms_row{actual_row_idx}.png"
            row_path = rows_dir / row_filename
            cv2.imwrite(str(row_path), row_img)

            # Save debug visualization if requested
            if args.debug:
                dbg_img = make_debug_image(row_img)
                dbg_path = debug_dir / f"frame{frame_idx:06d}_t{ts}ms_row{actual_row_idx}_debug.png"
                cv2.imwrite(str(dbg_path), dbg_img)

            # Extract icon — use OCR text boundaries when available
            entry = None
            try:
                entry = detector._parse_row(row_img)
            except Exception:
                pass

            ktr = entry.get("killer_text_right") if entry else None
            vtl = entry.get("victim_text_left") if entry else None
            icon = detector._extract_weapon_icon(row_img, killer_text_right=ktr, victim_text_left=vtl)
            icon_label = "unknown"
            if icon is not None:
                icon_filename = f"frame{frame_idx:06d}_t{ts}ms_row{actual_row_idx}_icon.png"
                icon_path = icons_dir / icon_filename
                cv2.imwrite(str(icon_path), icon)

                # Optionally classify
                if classifier is not None:
                    try:
                        icon_label = classifier.classify(icon)
                    except Exception:
                        icon_label = "error"
                    # save label text file
                    lbl_path = icons_dir / (icon_filename + ".label.txt")
                    lbl_path.write_text(str(icon_label))
            else:
                icon_filename = None

            saved += 1
            saved_meta.append({
                "frame": int(frame_idx),
                "t_ms": int(t_ms),
                "row_idx": int(actual_row_idx),
                "row_path": str(row_path),
                "icon_path": str(icon_path) if icon is not None else None,
                "label": icon_label,
            })

            if max_crops and saved >= max_crops:
                break
        if max_crops and saved >= max_crops:
            break

        frame_idx += 1

    cap.release()

    # Save metadata
    import json
    meta_path = out_dir / "crops_meta.json"
    meta_path.write_text(json.dumps(saved_meta, indent=2))

    print(f"Saved {saved} crops to {icons_dir}, rows to {rows_dir}")
    print(f"Metadata: {meta_path}")


if __name__ == '__main__':
    main()
