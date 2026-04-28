"""
IconCNNClassifier — wraps the trained weapon/ability icon CNN.

The model lives at /app/valorant_resources/icon_model.pth (in-container path)
and is mounted read-only from the host's valorant_resources/ directory.

Exposes `classify(bgr_img)` which takes a BGR numpy array (as returned by
OpenCV) and returns a string label such as 'Vandal' or 'Hunters_Fury'.
"""

from __future__ import annotations

import json
import os
from typing import Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Default model location — relative to /app inside the Docker container.
# Can be overridden by the MODEL_DIR env-var or the constructor argument.
# ---------------------------------------------------------------------------
def _resolve_model_dir() -> str:
    # Try relative path first (works when loaded via /app/vod_processor/app/...)
    candidate = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "../../../../valorant_resources")
    )
    if os.path.isdir(candidate):
        return candidate
    # Fallback for when loaded via /app/app/... mount (3 levels up lands at /app)
    return "/app/valorant_resources"

_DEFAULT_MODEL_DIR = os.environ.get("ICON_MODEL_DIR", _resolve_model_dir())

IMG_W = 128
IMG_H = 32


# Canonical team background colours (BGR) from synthetic_data.py.
# Used to normalise the letterbox padding colour so real crops with
# non-standard backgrounds (pink, magenta, …) still match training data.
_TEAM_COLORS_BGR = np.array([
    [159, 189, 112],   # teal  ~ RGB(112, 189, 159)
    [ 86,  73, 210],   # red   ~ RGB(210,  73,  86)
], dtype=np.float32)


def _snap_to_team_color_bgr(color: np.ndarray) -> np.ndarray:
    """Return the canonical team BGR colour closest to `color`."""
    dists = np.linalg.norm(_TEAM_COLORS_BGR - color.astype(np.float32), axis=1)
    return _TEAM_COLORS_BGR[int(np.argmin(dists))].astype(np.uint8)


def _dominant_team_color_bgr(bgr_img: np.ndarray, sat_threshold: int = 40) -> np.ndarray:
    """Detect background team colour using majority vote among saturated pixels
    across the entire image.

    Each pixel whose HSV saturation exceeds sat_threshold votes for its nearest
    canonical team colour (teal or red).  The large flat background region
    dominates by pixel count over any thin contaminated border strip or icon
    content, so this is robust to kill-feed row bleed (purple, magenta, etc.).

    Falls back to a snapped edge-median if no saturated pixels are found.
    """
    import cv2 as _cv2
    hsv = _cv2.cvtColor(bgr_img, _cv2.COLOR_BGR2HSV)
    sat_mask = hsv[:, :, 1] > sat_threshold
    if sat_mask.any():
        pixels = bgr_img[sat_mask].astype(np.float32)          # (N, 3)
        dists = np.linalg.norm(
            pixels[:, None, :] - _TEAM_COLORS_BGR[None, :, :], axis=2
        )                                                        # (N, 2)
        winner = int(np.bincount(np.argmin(dists, axis=1), minlength=len(_TEAM_COLORS_BGR)).argmax())
        return _TEAM_COLORS_BGR[winner].astype(np.uint8)
    # Fallback — no saturated pixels (e.g. greyscale / very washed-out crop)
    edges = np.concatenate(
        [bgr_img[0, :], bgr_img[-1, :], bgr_img[:, 0], bgr_img[:, -1]], axis=0
    )
    return _snap_to_team_color_bgr(np.median(edges, axis=0))


def _letterbox_bgr(bgr_img: np.ndarray, target_w: int, target_h: int,
                   pad_color: np.ndarray = None) -> np.ndarray:
    """
    Scale bgr_img to fit inside target_w×target_h preserving aspect ratio, then
    pad with the median edge-pixel colour (the team background colour).  This
    matches the letterbox compositing used by synthetic_data.py so inference
    input looks like training data instead of a distorted square-to-rectangle
    stretch.

    pad_color: optional BGR uint8 array to use as padding instead of the
    auto-detected edge median (used by _content_zoom_bgr to enforce a
    canonical team colour after cropping).
    """
    from PIL import Image as _PIL_Image

    h, w = bgr_img.shape[:2]
    scale = min(target_w / max(w, 1), target_h / max(h, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    # Resize preserving aspect ratio (via PIL to stay dependency-free from cv2 here)
    rgb = bgr_img[:, :, ::-1].copy()
    pil = _PIL_Image.fromarray(rgb.astype(np.uint8))
    resample = _PIL_Image.LANCZOS if scale < 1 else _PIL_Image.BILINEAR
    pil_resized = pil.resize((new_w, new_h), resample)
    resized_bgr = np.array(pil_resized)[:, :, ::-1].copy()

    if pad_color is None:
        pad_color = _dominant_team_color_bgr(bgr_img)

    canvas = np.full((target_h, target_w, 3), pad_color, dtype=np.uint8)
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized_bgr
    return canvas


def _content_zoom_bgr(bgr_img: np.ndarray, min_fill: float = 0.6) -> np.ndarray:
    """
    After letterboxing, if the icon occupies less than min_fill of the canvas
    width (e.g. a ~32×32 ability badge in a 128×32 canvas), detect the
    non-background content bounding box, crop to it with a small margin, then
    resize back to the full canvas size.

    This ensures small ability icons (Paint_Shells, NULL-cmd, Showstopper…)
    fill the CNN input rather than appearing as a tiny central blob surrounded
    by ~75% background padding.

    Background colour is re-estimated from the outer border of the already-
    letterboxed image (same median-edge approach used by _letterbox_bgr).
    """
    h, w = bgr_img.shape[:2]

    # Majority vote among saturated pixels → canonical team colour, robust to
    # kill-feed row bleed on the crop edges.
    bg_color = _dominant_team_color_bgr(bgr_img).astype(np.float64)

    # Content mask: exclude any pixel that is close to EITHER canonical team
    # colour (teal or red).  This removes both the primary background AND any
    # bleed from the adjacent kill-feed row (e.g. the victim's team colour on
    # the right edge) — neither colour ever appears in the icon itself.
    dists_to_canonical = np.linalg.norm(
        bgr_img.astype(np.float32)[:, :, np.newaxis, :] - _TEAM_COLORS_BGR[np.newaxis, np.newaxis, :, :],
        axis=3,
    ).min(axis=2)  # shape (h, w) — distance to nearest canonical colour
    mask = dists_to_canonical > 40

    cols = np.any(mask, axis=0)
    if not cols.any():
        return bgr_img  # no content detected, fall back

    x1, x2 = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])
    content_w = x2 - x1 + 1

    if content_w >= min_fill * w:
        return bgr_img  # icon already fills enough of the canvas

    rows = np.any(mask, axis=1)
    y1 = int(np.where(rows)[0][0])  if rows.any() else 0
    y2 = int(np.where(rows)[0][-1]) if rows.any() else h - 1

    # Add 15% margin so we don't clip icon edges
    mx = max(2, int(0.15 * content_w))
    my = max(2, int(0.15 * (y2 - y1 + 1)))
    x1 = max(0, x1 - mx);  x2 = min(w - 1, x2 + mx)
    y1 = max(0, y1 - my);  y2 = min(h - 1, y2 + my)

    cropped = bgr_img[y1:y2 + 1, x1:x2 + 1]
    if cropped.size == 0:
        return bgr_img

    return _letterbox_bgr(cropped, w, h, pad_color=bg_color.astype(np.uint8))


def _unsharp_mask_bgr(bgr_img: np.ndarray, amount: float = 0.8, kernel_size: int = 3, sigma: float = 1.0) -> np.ndarray:
    """Sharpens icon edges via unsharp mask — improves contour detection for
    small ability icons blurred by broadcast compression.  Must match the
    SharpenTransform(amount=0.8) applied to val data in train_cnn.py."""
    import cv2
    arr = bgr_img.astype(np.float32)
    blurred = cv2.GaussianBlur(arr, (kernel_size, kernel_size), sigma)
    return np.clip(arr * (1.0 + amount) - blurred * amount, 0, 255).astype(np.uint8)


def _build_model(num_classes: int):
    """Construct the IconCNN architecture (must match train_cnn.py exactly)."""
    import torch.nn as nn

    class IconCNN(nn.Module):
        def __init__(self, num_classes: int):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(1, 2)),    # width-only pool -> 16x8
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 8 * 16, 256),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(256, num_classes),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    return IconCNN(num_classes)


class IconCNNClassifier:
    """
    Thin wrapper around the trained IconCNN.

    Parameters
    ----------
    model_dir : str, optional
        Directory containing ``icon_model.pth`` and ``class_names.json``.
        Defaults to the mounted ``valorant_resources/`` path.
    confidence_threshold : float
        Predictions below this confidence are returned as 'unknown'.
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        confidence_threshold: float = 0.25,
    ):
        import torch
        from torchvision import transforms

        self._confidence_threshold = confidence_threshold
        model_dir = model_dir or _DEFAULT_MODEL_DIR

        weights_path = os.path.join(model_dir, "icon_model.pth")
        labels_path = os.path.join(model_dir, "class_names.json")

        if not os.path.isfile(weights_path):
            raise FileNotFoundError(
                f"IconCNNClassifier: model weights not found at {weights_path}"
            )
        if not os.path.isfile(labels_path):
            raise FileNotFoundError(
                f"IconCNNClassifier: class names not found at {labels_path}"
            )

        with open(labels_path, "r") as f:
            self._class_names = json.load(f)

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = _build_model(len(self._class_names)).to(self._device)
        self._model.load_state_dict(
            torch.load(weights_path, map_location=self._device)
        )
        self._model.eval()

        self._transform = transforms.Compose([
            transforms.Resize((IMG_H, IMG_W)),
            transforms.ToTensor(),
        ])

        print(
            f"[IconCNNClassifier] Loaded {len(self._class_names)}-class model "
            f"from {weights_path} (device={self._device})"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, bgr_img: np.ndarray) -> str:
        """
        Classify a weapon/ability icon crop.

        Parameters
        ----------
        bgr_img : np.ndarray
            BGR uint8 image (OpenCV format).

        Returns
        -------
        str
            Class label, e.g. 'Vandal', or 'unknown' if confidence is below
            the threshold or the image is invalid.
        """
        label, _ = self.classify_with_confidence(bgr_img)
        return label

    def classify_with_confidence(self, bgr_img: np.ndarray) -> Tuple[str, float]:
        """Like :meth:`classify` but also returns the confidence score."""
        import torch
        from PIL import Image

        if bgr_img is None or bgr_img.size == 0:
            return "unknown", 0.0

        try:
            # Letterbox to match synthetic training data: preserve aspect ratio
            # and pad with team background colour.  Direct Resize distorts square
            # ability icon crops into a 4:1 rectangle, causing misclassification.
            bgr_img = _letterbox_bgr(bgr_img, IMG_W, IMG_H)

            # Content-aware zoom: if the icon is small (e.g. a ~32×32 ability
            # badge in the 128×32 canvas) detect the non-background bbox and
            # resize it to fill the canvas so the CNN sees more icon pixels.
            bgr_img = _content_zoom_bgr(bgr_img)

            # Unsharp mask — sharpens icon edges blurred by broadcast compression.
            # Must match SharpenTransform(amount=0.8) applied to val data in training.
            bgr_img = _unsharp_mask_bgr(bgr_img)

            # OpenCV BGR → PIL RGB
            rgb = bgr_img[:, :, ::-1].copy()
            pil_img = Image.fromarray(rgb.astype(np.uint8))
            tensor = self._transform(pil_img).unsqueeze(0).to(self._device)

            with torch.no_grad():
                logits = self._model(tensor)
                probs = torch.softmax(logits, dim=1)
                conf, idx = torch.max(probs, dim=1)

            confidence = conf.item()
            label = self._class_names[idx.item()]

            if confidence < self._confidence_threshold:
                return "unknown", confidence

            return label, confidence

        except Exception as exc:
            print(f"[IconCNNClassifier] classify failed: {exc}")
            return "unknown", 0.0
