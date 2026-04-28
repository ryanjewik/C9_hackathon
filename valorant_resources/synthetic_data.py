import os
import cv2
import numpy as np
import random
from pathlib import Path

# ---------------- CONFIG ----------------

INPUT_DIR = "abilities"   # weapons removed — enough real samples exist

OUTPUT_DIR = "dataset"
TRAIN_SPLIT = 0.85
SAMPLES_PER_ICON = 600

# Output canvas size (W x H).  Killfeed crops are ~4:1 landscape.
OUT_W = 128
OUT_H = 32

# Team background color ranges measured from real killfeed crops (BGR).
# Each entry is (center, half-range) — a random value is sampled within
# [center-half_range, center+half_range] per channel to simulate the
# variation caused by broadcast JPEG compression.
# Teal  ~ RGB(0, 200, 170)  -> BGR(170, 200, 0)  broadcast-compressed median BGR(159,189,112)
# Red   ~ RGB(190, 35, 55)  -> BGR(55,  35, 190) broadcast-compressed median BGR(86, 73, 210)
TEAM_COLORS_BGR = [
    ((159, 189, 112), 20),   # teal
    ((86,  73,  210), 20),   # red
]

# ----------------------------------------


def random_team_bg(w, h):
    """Fill a canvas with a randomly chosen team background color,
    with per-pixel noise and a per-sample color shift to simulate
    broadcast compression variation."""
    center, half_range = random.choice(TEAM_COLORS_BGR)
    # Per-sample color shift (simulates different video compression levels)
    shift = np.array([random.randint(-half_range, half_range) for _ in range(3)], dtype=np.int16)
    base = np.clip(np.array(center, dtype=np.int16) + shift, 0, 255).astype(np.uint8)
    canvas = np.full((h, w, 3), base, dtype=np.uint8)
    # Fine per-pixel noise on top
    noise = np.random.randint(-8, 9, (h, w, 3), dtype=np.int16)
    canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return canvas


def place_icon_on_bg(icon_rgb, alpha, out_w, out_h, max_fill_w=None):
    """Alpha-composite icon onto a random team-colored background,
    letterboxing to preserve aspect ratio.

    max_fill_w: if set, caps the icon's rendered width to this many pixels
    (simulates the ~70-90% canvas fill seen in real killfeed crops for weapons).
    """
    bg = random_team_bg(out_w, out_h)

    h, w = icon_rgb.shape[:2]
    scale = min(out_w / w, out_h / h)
    if max_fill_w is not None:
        scale = min(scale, max_fill_w / w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    icon_resized = cv2.resize(icon_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    alpha_resized = cv2.resize(alpha, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Random small shift (simulate imperfect crop alignment)
    max_shift_x = max(0, (out_w - new_w) // 2)
    max_shift_y = max(0, (out_h - new_h) // 2)
    x0 = (out_w - new_w) // 2 + random.randint(-max_shift_x // 2, max_shift_x // 2)
    y0 = (out_h - new_h) // 2 + random.randint(-max_shift_y // 2, max_shift_y // 2)
    x0 = max(0, min(x0, out_w - new_w))
    y0 = max(0, min(y0, out_h - new_h))

    a = alpha_resized[:, :, np.newaxis].astype(np.float32) / 255.0
    region = bg[y0:y0+new_h, x0:x0+new_w].astype(np.float32)
    blended = a * icon_resized.astype(np.float32) + (1 - a) * region
    bg[y0:y0+new_h, x0:x0+new_w] = np.clip(blended, 0, 255).astype(np.uint8)
    return bg


def augment(img):
    # Random brightness
    factor = random.uniform(0.75, 1.25)
    img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    # Slight blur
    if random.random() > 0.6:
        img = cv2.GaussianBlur(img, (3, 3), 0)

    # Random noise
    if random.random() > 0.7:
        noise = np.random.normal(0, 6, img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Simulate broadcast JPEG compression artifacts
    if random.random() > 0.5:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(55, 90)]
        _, encimg = cv2.imencode('.jpg', img, encode_param)
        img = cv2.imdecode(encimg, 1)

    # Unsharp mask sharpening (counterpart to blurring above) — teaches model
    # to handle both soft and crisp icon edges.  Particularly important for
    # ability icons where circular contours are the primary distinguishing feature.
    if random.random() > 0.5:
        amount = random.uniform(0.5, 1.5)
        blurred = cv2.GaussianBlur(img.astype(np.float32), (3, 3), 1.0)
        img = np.clip(img.astype(np.float32) * (1 + amount) - blurred * amount, 0, 255).astype(np.uint8)

    return img


def process_icon(icon_path, class_name, flip=False, ability_icon=False):
    icon = cv2.imread(str(icon_path), cv2.IMREAD_UNCHANGED)
    if icon is None:
        print(f"  [SKIP] Could not read {icon_path}")
        return

    if icon.ndim == 2:
        icon = cv2.cvtColor(icon, cv2.COLOR_GRAY2BGRA)
    elif icon.shape[2] == 3:
        # No alpha — assume white icon on black; derive alpha from brightness
        gray = cv2.cvtColor(icon, cv2.COLOR_BGR2GRAY)
        alpha = gray
        icon = np.dstack([icon, alpha])

    icon_rgb = icon[:, :, :3]
    alpha = icon[:, :, 3]

    # Weapon icons in source PNGs face the opposite direction from how they
    # appear in the killfeed.  Mirror horizontally to match inference crops.
    if flip:
        icon_rgb = cv2.flip(icon_rgb, 1)
        alpha = cv2.flip(alpha, 1)

    train_dir = Path(OUTPUT_DIR) / "train" / class_name
    val_dir   = Path(OUTPUT_DIR) / "val"   / class_name
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    base_name = icon_path.stem

    for i in range(SAMPLES_PER_ICON):
        # Random scale variation before placement
        scale_factor = random.uniform(0.75, 1.25)
        scaled_rgb = cv2.resize(icon_rgb, None, fx=scale_factor, fy=scale_factor)
        scaled_alpha = cv2.resize(alpha, None, fx=scale_factor, fy=scale_factor)

        # Weapons: cap canvas fill to match real killfeed crop width (~65-90% of OUT_W).
        # Real crops are ~80-95px extracted from a killfeed row, then letterboxed into
        # the 128px canvas — so the icon fills roughly 65-90% of canvas width.
        # For thin landscape weapon assets (aspect ratio ≈ 4:1) the normal scale fills
        # 100% of the canvas, creating a training/inference distribution mismatch.
        if flip:
            max_fill = int(OUT_W * random.uniform(0.65, 0.92))
        elif ability_icon:
            # Ability icons in real crops appear smaller than the full canvas —
            # cap fill to 30-55% of canvas width to better match real-world crops.
            max_fill = int(OUT_W * random.uniform(0.30, 0.55))
        else:
            max_fill = None

        # For ability icons (webp): add pre-composite noise directly onto icon
        # pixels to simulate broadcast compression degradation on the icon itself.
        if ability_icon:
            noise_sigma = random.uniform(8, 22)
            icon_noise = np.random.normal(0, noise_sigma, scaled_rgb.shape)
            icon_mask = scaled_alpha[:, :, np.newaxis] > 10
            scaled_rgb = np.clip(
                scaled_rgb.astype(np.float32) + icon_noise * icon_mask, 0, 255
            ).astype(np.uint8)

        img = place_icon_on_bg(scaled_rgb, scaled_alpha, OUT_W, OUT_H, max_fill_w=max_fill)
        img = augment(img)

        split_dir = train_dir if random.random() < TRAIN_SPLIT else val_dir
        cv2.imwrite(str(split_dir / f"{base_name}_{i}.png"), img)

    print(f"  Generated {SAMPLES_PER_ICON} samples for '{class_name}'")


def main():
    # ── Abilities ──────────────────────────────────────────────────────────
    print(f"Scanning abilities in 'abilities'...")
    for file in sorted(os.listdir("abilities")):
        if not file.lower().endswith((".png", ".webp", ".jpg", ".jpeg")):
            continue
        icon_path = Path("abilities") / file
        class_name = icon_path.stem
        print(f"Processing ability: {class_name}")
        # Apply extra realism effects (smaller fill + icon noise) for webp ability icons.
        # Fall.png and Spike.png are kept as-is since they are overlay graphics.
        # Webp ability icons are horizontally mirrored vs in-game orientation, so flip them.
        is_webp = icon_path.suffix.lower() == ".webp"
        process_icon(icon_path, class_name, flip=is_webp, ability_icon=is_webp)

    # ── Weapons (horizontally flipped to match killfeed orientation) ─────────
    # Source PNGs face right but the killfeed shows them facing left,
    # so flip every weapon icon to match the real training data orientation.
    weapons_dir = "weapons"
    if os.path.isdir(weapons_dir):
        print(f"\nScanning weapons in '{weapons_dir}' (with horizontal flip)...")
        for file in sorted(os.listdir(weapons_dir)):
            if not file.lower().endswith((".png", ".webp", ".jpg", ".jpeg")):
                continue
            icon_path = Path(weapons_dir) / file
            class_name = icon_path.stem
            print(f"Processing weapon: {class_name}")
            process_icon(icon_path, class_name, flip=True)
    else:
        print(f"\n[WARN] Weapons directory '{weapons_dir}' not found — skipping weapons")

    print("\nSynthetic dataset generation complete.")


if __name__ == "__main__":
    main()
