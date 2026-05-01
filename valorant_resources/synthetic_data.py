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
SAMPLES_PER_ABILITY = 1200  # abilities are rarer in-game and harder to learn — extra coverage helps

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
    ((159, 189, 112), 12),   # teal — tighter range to stay closer to real broadcast colours
    ((86,  73,  210), 12),   # red
]

# ----------------------------------------


def random_team_bg(w, h):
    """Fill a canvas with a randomly chosen team background color,
    with per-pixel noise and a per-sample color shift to simulate
    broadcast compression variation.
    Returns (canvas, base_color_bgr) where base_color_bgr is the canonical
    team colour chosen (before noise/desaturation) so callers can derive
    the opposite team colour for killfeed bleed simulation."""
    center, half_range = random.choice(TEAM_COLORS_BGR)
    # Per-sample color shift (simulates different video compression levels)
    shift = np.array([random.randint(-half_range, half_range) for _ in range(3)], dtype=np.int16)
    base = np.clip(np.array(center, dtype=np.int16) + shift, 0, 255).astype(np.uint8)
    canvas = np.full((h, w, 3), base, dtype=np.uint8)
    # Fine per-pixel noise on top — toned down so backgrounds stay recognisable
    noise = np.random.randint(-5, 6, (h, w, 3), dtype=np.int16)
    canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # Simulate broadcast-compression desaturation (modest range — real crops are
    # muted but not radically so).
    hsv = cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.78, 0.92), 0, 255)
    canvas = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return canvas, np.array(center, dtype=np.uint8)


def add_killfeed_bleed(canvas, killer_bg_color_bgr, icon_left_x=None, icon_right_x=None):
    """Simulate real killfeed crop boundary artifacts.

    Layout observed in real crops:

      [killer bg] [line_L] [icon] [line_M] [victim strip] [line_R] [killer bg]
                  [──── top black bar ────────────────────]
                  [─── bottom black bar ──────────────────]

      - line_L  : 1px black, left edge of icon
      - line_M  : 1px black, right edge of icon / left edge of victim strip
      - victim strip: 3-10px of victim team colour (solid, not faded)
      - line_R  : 1px black, right edge of victim strip
      - top/bottom bars: 1px black spanning from line_L to line_R

    icon_left_x : first column of the icon (inclusive).
    icon_right_x: first column after the icon (exclusive).
    Both default to 60-80% estimates if not supplied.
    """
    h, w = canvas.shape[:2]

    # ── Victim team colour (opposite of killer) ──
    teal_bgr = np.array([159, 189, 112], dtype=np.uint8)
    red_bgr  = np.array([ 86,  73, 210], dtype=np.uint8)
    if np.linalg.norm(killer_bg_color_bgr.astype(np.float32) - teal_bgr.astype(np.float32)) \
       < np.linalg.norm(killer_bg_color_bgr.astype(np.float32) - red_bgr.astype(np.float32)):
        victim_color = red_bgr
    else:
        victim_color = teal_bgr

    # Broadcast-compression saturation reduction.
    victim_hsv = cv2.cvtColor(victim_color.reshape(1, 1, 3), cv2.COLOR_BGR2HSV).astype(np.float32)
    victim_hsv[0, 0, 1] = np.clip(victim_hsv[0, 0, 1] * random.uniform(0.60, 0.90), 0, 255)
    victim_color = cv2.cvtColor(victim_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).reshape(3)

    # Fallback positions if not supplied.
    if icon_right_x is None:
        icon_right_x = int(w * random.uniform(0.60, 0.80))
    if icon_left_x is None:
        icon_left_x = max(0, icon_right_x - int(w * 0.50))

    # ── Padding between icon edges and the boundary lines ──
    # Gives the icon breathing room within its killer-bg colour region.
    pad_x = random.randint(3, 6)

    # ── Narrow victim-colour strip (3-10px) right of icon ──
    bleed_w = random.randint(3, 10)
    line_M     = min(icon_right_x + pad_x, w - 1)  # grey line after icon padding
    strip_start = line_M + 1
    strip_end   = min(strip_start + bleed_w, w - 1)  # -1 for line_R

    if strip_end > strip_start:
        noise = np.random.randint(-10, 11, (h, strip_end - strip_start, 3), dtype=np.int16)
        victim_strip = np.clip(victim_color.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        canvas[:, strip_start:strip_end] = victim_strip

    # ── Light-grey vertical lines: left of icon, right of icon+pad, right of strip ──
    line_L = max(0, icon_left_x - pad_x)   # grey line before icon padding
    line_R = strip_end                      # grey line right of victim strip
    grey = random.randint(150, 210)
    for lx in (line_L, line_M, line_R):
        if 0 <= lx < w:
            canvas[:, lx] = grey

    # ── Bottom thin bar spanning line_L to line_R ──
    # (Top black bar removed — real crops show it is absent.)
    BAR_H = random.randint(1, 2)
    bar_x0 = line_L
    bar_x1 = min(line_R + 1, w)
    if bar_x1 > bar_x0:
        if random.random() < 0.25:
            # Lime/yellow bar replacing the black bottom bar
            b = random.randint(10, 50)
            g = random.randint(190, 235)
            r = random.randint(120, 175)
            canvas[h - BAR_H:h, bar_x0:bar_x1] = np.array([b, g, r], dtype=np.uint8)
        else:
            canvas[h - BAR_H:h, bar_x0:bar_x1] = 0

    return canvas


def place_icon_on_bg(icon_rgb, alpha, out_w, out_h, max_fill_w=None, ability_icon=False):
    """Alpha-composite icon onto a random team-colored background,
    letterboxing to preserve aspect ratio.

    max_fill_w: if set, caps the icon's rendered width to this many pixels
    (simulates the ~70-90% canvas fill seen in real killfeed crops for weapons).
    ability_icon: if True, applies killfeed boundary artifacts (black bars,
    grey vertical lines, victim-colour strip).  Only set for webp ability icons.
    """
    # random_team_bg returns (canvas, canonical_color_bgr) so we know which
    # team colour was used and can derive the opposite for bleed simulation.
    bg, killer_bg_color = random_team_bg(out_w, out_h)

    h, w = icon_rgb.shape[:2]

    if ability_icon:
        # Reserve space for bars + padding so the icon has breathing room on all sides.
        # BAR_H: black bar thickness (max of range in add_killfeed_bleed).
        # PAD_X: horizontal gap between icon and grey vertical lines.
        # PAD_Y: vertical gap between icon and black bars (team colour visible here).
        BAR_H = 2
        PAD_X = 6
        PAD_Y = 4  # px of team-colour gap between icon and black bars
        avail_h = out_h - 2 * (BAR_H + PAD_Y)
        avail_w = out_w - 2 * PAD_X
        scale = min(avail_w / w, avail_h / h)
        if max_fill_w is not None:
            scale = min(scale, max_fill_w / w)
        new_w = int(w * scale)
        new_h = int(h * scale)

        icon_resized = cv2.resize(icon_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        alpha_resized = cv2.resize(alpha, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Center icon within the padded band.
        # Tight x-jitter (±4px): inference preprocessing always centers the icon
        # via letterbox+content-zoom, so large x drift creates a train/infer mismatch.
        max_shift_y = max(0, (avail_h - new_h) // 2)
        x0 = PAD_X + (avail_w - new_w) // 2 + random.randint(-4, 4)
        y0 = (BAR_H + PAD_Y) + (avail_h - new_h) // 2 + random.randint(-max_shift_y // 2, max_shift_y // 2)
        x0 = max(PAD_X, min(x0, out_w - PAD_X - new_w))
        y0 = max(BAR_H + PAD_Y, min(y0, out_h - BAR_H - PAD_Y - new_h))

        a = alpha_resized[:, :, np.newaxis].astype(np.float32) / 255.0
        region = bg[y0:y0+new_h, x0:x0+new_w].astype(np.float32)
        blended = a * icon_resized.astype(np.float32) + (1 - a) * region
        bg[y0:y0+new_h, x0:x0+new_w] = np.clip(blended, 0, 255).astype(np.uint8)

        # Simulate killfeed row boundary artifacts (bars, lines, victim strip).
        bg = add_killfeed_bleed(bg, killer_bg_color, icon_left_x=x0, icon_right_x=x0 + new_w)
    else:
        # Weapons (png): simple centered layout, no boundary artifacts.
        scale = min(out_w / w, out_h / h)
        if max_fill_w is not None:
            scale = min(scale, max_fill_w / w)
        new_w = int(w * scale)
        new_h = int(h * scale)

        icon_resized = cv2.resize(icon_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        alpha_resized = cv2.resize(alpha, (new_w, new_h), interpolation=cv2.INTER_AREA)

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

    n_samples = SAMPLES_PER_ABILITY if ability_icon else SAMPLES_PER_ICON
    for i in range(n_samples):
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
            # Square webp ability icons are always height-limited in a 128×32 canvas
            # (scale = min(128/w, 32/h) = 32/h for square icons), so a width cap is
            # a no-op.  Drive scale from a target height (70-95% of OUT_H) instead
            # so the icon fills 22-30px, matching what real killfeed crops look like.
            h_icon = max(scaled_rgb.shape[0], 1)
            target_h = int(OUT_H * random.uniform(0.70, 0.95))
            max_fill = int(scaled_rgb.shape[1] * (target_h / h_icon))
        else:
            max_fill = None

        # For ability icons (webp): pre-composite degradation to simulate broadcast
        # compression artifacts before compositing onto the background.
        if ability_icon:
            # Toned-down noise sigma (10-22 vs old 15-35) — enough degradation to
            # close the synthetic/real gap without smearing distinguishing features.
            noise_sigma = random.uniform(10, 22)
            icon_noise = np.random.normal(0, noise_sigma, scaled_rgb.shape)
            icon_mask = scaled_alpha[:, :, np.newaxis] > 10
            scaled_rgb = np.clip(
                scaled_rgb.astype(np.float32) + icon_noise * icon_mask, 0, 255
            ).astype(np.uint8)
            # Pre-composite blur: toned down sigma range (0.3-1.0 vs old 0.5-1.5).
            if random.random() > 0.35:
                sigma = random.uniform(0.3, 1.0)
                scaled_rgb = cv2.GaussianBlur(scaled_rgb, (3, 3), sigma)

        img = place_icon_on_bg(scaled_rgb, scaled_alpha, OUT_W, OUT_H, max_fill_w=max_fill, ability_icon=ability_icon)
        img = augment(img)

        # Multi-pass JPEG for ability icons: simulates game→broadcast→VOD platform
        # double-encoding that crushes thin crosshair ring lines and contour edges.
        if ability_icon:
            for _ in range(random.randint(1, 2)):
                q = random.randint(35, 65)
                _, enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q])
                img = cv2.imdecode(enc, 1)

        split_dir = train_dir if random.random() < TRAIN_SPLIT else val_dir
        cv2.imwrite(str(split_dir / f"{base_name}_{i}.png"), img)

    print(f"  Generated {n_samples} samples for '{class_name}'")


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
