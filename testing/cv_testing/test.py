import cv2
import numpy as np
import json
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

# -----------------------------
# Config: ROIs (normalized coords)
# -----------------------------
# Use normalized coordinates (x, y, w, h) in [0..1] so it scales with resolution.
# Replace these with the correct values for your HUD style.
ROI_NORM: Dict[str, Tuple[float, float, float, float]] = {
    # Adjusted to match screenshots: (x, y, w, h) normalized
    "minimap":     (0.02, 0.02, 0.18, 0.26),
    "left_pips":   (0.02, 0.20, 0.18, 0.60),
    "top_hud":     (0.33, 0.00, 0.34, 0.12),
    "killfeed":    (0.62, 0.04, 0.36, 0.12),
    "right_pips":  (0.78, 0.20, 0.18, 0.60),
    "bottom_hud":  (0.28, 0.70, 0.44, 0.20),
}

# Per-detector "effective FPS" (not video FPS)
DET_FPS: Dict[str, float] = {
    "killfeed":  20.0,
    "left_pips": 15.0,
    "right_pips":15.0,
    "top_hud":   10.0,
    "bottom_hud":12.0,
    "minimap":   12.0,
}

# Debug output controls
WRITE_DEBUG_VIDEO = True
DEBUG_VIDEO_PATH = "debug_cv.mp4"

# Event output
EVENTS_PATH = "events.json"


# -----------------------------
# Utility helpers
# -----------------------------
def roi_to_px(frame_w: int, frame_h: int, roi_norm: Tuple[float, float, float, float]) -> Tuple[int, int, int, int]:
    x, y, w, h = roi_norm
    px = int(x * frame_w)
    py = int(y * frame_h)
    pw = int(w * frame_w)
    ph = int(h * frame_h)
    return px, py, pw, ph


def crop(frame, roi_px: Tuple[int, int, int, int]):
    x, y, w, h = roi_px
    return frame[y:y+h, x:x+w]


def mse_gray(a, b) -> float:
    # Mean squared error between two grayscale images
    if a is None or b is None:
        return 1e9
    if a.shape != b.shape:
        return 1e9
    diff = (a.astype("float32") - b.astype("float32"))
    return float((diff * diff).mean())


def _find_contours_gray(img_gray, blur=5, thresh=30):
    g = cv2.GaussianBlur(img_gray, (blur, blur), 0)
    _, th = cv2.threshold(g, thresh, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def count_roundish_contours(roi_bgr) -> int:
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    contours = _find_contours_gray(gray, blur=5, thresh=50)
    count = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < 30 or area > 2000:
            continue
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / float(h + 1e-6)
        if 0.6 <= aspect <= 1.6:
            count += 1
    return count


def count_color_blobs(roi_bgr, min_area=50) -> int:
    # heuristically count colored blobs on minimap (player icons)
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    # broad mask for bright colored player icons (red/blue/green/yellow)
    masks = []
    # red range (two ranges)
    masks.append(cv2.inRange(hsv, (0, 100, 100), (10, 255, 255)))
    masks.append(cv2.inRange(hsv, (160, 100, 100), (179, 255, 255)))
    # blue
    masks.append(cv2.inRange(hsv, (90, 80, 80), (140, 255, 255)))
    # green/yellow
    masks.append(cv2.inRange(hsv, (20, 80, 80), (60, 255, 255)))

    m = np.zeros_like(masks[0])
    for mm in masks:
        m = cv2.bitwise_or(m, mm)

    # cleanup
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = 0
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            cnt += 1
    return cnt


@dataclass
class Event:
    t_ms: float
    type: str
    roi: str
    payload: Dict[str, Any]


class RateGate:
    """
    Run a detector at a target effective FPS, using timestamps from the video.
    """
    def __init__(self, target_fps: float):
        self.target_fps = target_fps
        self.period_ms = 1000.0 / target_fps if target_fps > 0 else 0.0
        self.next_due_ms = 0.0

    def due(self, t_ms: float) -> bool:
        if self.period_ms <= 0:
            return True
        if t_ms >= self.next_due_ms:
            self.next_due_ms = t_ms + self.period_ms
            return True
        return False


# -----------------------------
# Detector base + skeleton detectors
# -----------------------------
class Detector:
    name: str

    def __init__(self, name: str, target_fps: float, change_mse_threshold: Optional[float] = None):
        self.name = name
        self.gate = RateGate(target_fps)
        self.change_mse_threshold = change_mse_threshold
        self._prev_gray = None  # for change gating
        self._last_fire_ms = -1e9

    def should_run(self, t_ms: float, roi_bgr) -> bool:
        if not self.gate.due(t_ms):
            return False

        if self.change_mse_threshold is None:
            return True

        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        score = mse_gray(gray, self._prev_gray)
        self._prev_gray = gray
        return score >= self.change_mse_threshold

    def process(self, t_ms: float, roi_bgr) -> List[Event]:
        raise NotImplementedError


class KillfeedDetector(Detector):
    def __init__(self, target_fps: float):
        # Killfeed changes quickly; gate by change detection
        super().__init__("killfeed", target_fps, change_mse_threshold=60.0)

    def process(self, t_ms: float, roi_bgr) -> List[Event]:
        # Simple heuristic: count large rectangular changes in RHS killfeed area
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        contours = _find_contours_gray(gray, blur=7, thresh=40)
        large_rects = 0
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w > roi_bgr.shape[1] * 0.25 and h > roi_bgr.shape[0] * 0.15:
                large_rects += 1

        evs: List[Event] = []
        if large_rects > 0:
            evs.append(Event(t_ms=t_ms, type="KILLFEED_ROWS", roi=self.name, payload={"rows": large_rects}))
        return evs


class PipsDetector(Detector):
    def __init__(self, name: str, target_fps: float):
        # Pips are stable; gate by change detection to avoid running constantly
        super().__init__(name, target_fps, change_mse_threshold=40.0)
        self.last_count: Optional[int] = None
        self.last_event_ms = -1e9

    def _count_pips_placeholder(self, roi_bgr) -> int:
        # Rough circular blob count for pips/ability indicators
        try:
            return count_roundish_contours(roi_bgr)
        except Exception:
            return 0

    def process(self, t_ms: float, roi_bgr) -> List[Event]:
        count = self._count_pips_placeholder(roi_bgr)
        evs: List[Event] = []

        if self.last_count is None:
            self.last_count = count
            return evs

        # Example transition logic: if pips decreased
        if count < self.last_count and (t_ms - self.last_event_ms) > 600:
            evs.append(Event(
                t_ms=t_ms,
                type="UTILITY_PIP_DROP",
                roi=self.name,
                payload={"prev": self.last_count, "curr": count}
            ))
            self.last_event_ms = t_ms

        self.last_count = count
        return evs


class TopHudDetector(Detector):
    def __init__(self, target_fps: float):
        super().__init__("top_hud", target_fps, change_mse_threshold=35.0)

    def process(self, t_ms: float, roi_bgr) -> List[Event]:
        # Detect significant brightness/structure changes at top HUD (timer/round icon changes)
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        mean = float(gray.mean())
        evs: List[Event] = []
        if hasattr(self, "_last_mean"):
            if abs(mean - self._last_mean) > 12:
                evs.append(Event(t_ms=t_ms, type="TOPHUD_CHANGE", roi=self.name, payload={"mean": mean}))
        self._last_mean = mean
        return evs


class BottomHudDetector(Detector):
    def __init__(self, target_fps: float):
        super().__init__("bottom_hud", target_fps, change_mse_threshold=35.0)

    def process(self, t_ms: float, roi_bgr) -> List[Event]:
        # Heuristic: detect overlay changes (player portrait/health) via MSE
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        evs: List[Event] = []
        if self._prev_gray is not None:
            score = mse_gray(gray, self._prev_gray)
            if score > 200.0:
                evs.append(Event(t_ms=t_ms, type="BOTTOMHUD_SPIKE", roi=self.name, payload={"mse": score}))
        self._prev_gray = gray
        return evs


class MinimapDetector(Detector):
    def __init__(self, target_fps: float):
        super().__init__("minimap", target_fps, change_mse_threshold=30.0)

    def process(self, t_ms: float, roi_bgr) -> List[Event]:
        # Count bright colored blobs (player icons) on minimap; emit event if count changes
        evs: List[Event] = []
        try:
            cnt = count_color_blobs(roi_bgr, min_area=40)
        except Exception:
            cnt = 0

        if hasattr(self, "last_cnt"):
            if cnt != self.last_cnt:
                evs.append(Event(t_ms=t_ms, type="MINIMAP_BLIPS", roi=self.name, payload={"count": cnt, "prev": self.last_cnt}))
        self.last_cnt = cnt
        return evs


# -----------------------------
# Main processing loop
# -----------------------------
def main():
    video_path = "round_8.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {video_path} ({w}x{h}) fps={fps:.2f}")

    # Build detectors
    detectors: List[Detector] = [
        KillfeedDetector(DET_FPS["killfeed"]),
        PipsDetector("left_pips", DET_FPS["left_pips"]),
        PipsDetector("right_pips", DET_FPS["right_pips"]),
        TopHudDetector(DET_FPS["top_hud"]),
        BottomHudDetector(DET_FPS["bottom_hud"]),
        MinimapDetector(DET_FPS["minimap"]),
    ]

    # Debug video writer
    writer = None
    if WRITE_DEBUG_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(DEBUG_VIDEO_PATH, fourcc, fps, (w, h))
        if not writer.isOpened():
            print("WARN: Could not open debug video writer; disabling.")
            writer = None

    events: List[Event] = []

    frame_idx = 0
    t_start = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

        # For each detector: crop ROI and optionally run
        for det in detectors:
            roi_norm = ROI_NORM[det.name]
            roi_px = roi_to_px(w, h, roi_norm)
            roi_img = crop(frame, roi_px)

            if roi_img.size == 0:
                continue

            if det.should_run(t_ms, roi_img):
                evs = det.process(t_ms, roi_img)
                events.extend(evs)

        # Debug overlay: draw ROIs
        if writer is not None:
            dbg = frame.copy()
            for name, roi_norm in ROI_NORM.items():
                x, y, rw, rh = roi_to_px(w, h, roi_norm)
                cv2.rectangle(dbg, (x, y), (x + rw, y + rh), (0, 255, 0), 2)
                cv2.putText(dbg, name, (x + 5, y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            writer.write(dbg)

        frame_idx += 1
        if frame_idx % int(max(1, fps)) == 0:
            elapsed = time.time() - t_start
            print(f"Processed ~{frame_idx} frames in {elapsed:.1f}s (x{(frame_idx/fps)/elapsed:.2f} realtime)")

    cap.release()
    if writer is not None:
        writer.release()

    # Save events
    out = [
        {"t_ms": e.t_ms, "type": e.type, "roi": e.roi, "payload": e.payload}
        for e in events
    ]
    with open(EVENTS_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"\nDone. Wrote {len(events)} events to {EVENTS_PATH}")
    if WRITE_DEBUG_VIDEO:
        print(f"Wrote debug video to {DEBUG_VIDEO_PATH}")


if __name__ == "__main__":
    main()
