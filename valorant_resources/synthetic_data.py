import os
import cv2
import numpy as np
import random
from pathlib import Path

# ---------------- CONFIG ----------------

INPUT_DIRS = [
    "weapons",
    "abilities"
]

OUTPUT_DIR = "dataset"
TRAIN_SPLIT = 0.85
SAMPLES_PER_ICON = 400
IMAGE_SIZE = 64

# ----------------------------------------

def resize_with_padding(img, size=64):
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    img = cv2.resize(img, (new_w, new_h))

    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y_offset = (size - new_h) // 2
    x_offset = (size - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img

    return canvas

def apply_team_tint(img):
    team = random.choice(["blue", "red", None])
    if team is None:
        return img

    img = img.astype(np.float32)

    if team == "blue":
        img[:, :, 0] *= 1.15  # boost blue channel
    else:
        img[:, :, 2] *= 1.15  # boost red channel

    return np.clip(img, 0, 255).astype(np.uint8)



def augment_icon(icon):
    img = icon.copy()

    # Random brightness
    brightness = random.uniform(0.75, 1.25)
    img = np.clip(img * brightness, 0, 255).astype(np.uint8)

    # Team tint
    img = apply_team_tint(img)

    # Slight blur
    if random.random() > 0.6:
        img = cv2.GaussianBlur(img, (3, 3), 0)

    # Add slight noise
    if random.random() > 0.7:
        noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)
    
    # Simulate broadcast compression
    if random.random() > 0.5:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(60, 90)]
        _, encimg = cv2.imencode('.jpg', img, encode_param)
        img = cv2.imdecode(encimg, 1)


    return img

def place_on_background(img):
    # Realistic Valorant killfeed dark UI background
    bg = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 28, dtype=np.uint8)

    # Slight random shift (simulate imperfect crop)
    shift_x = random.randint(-4, 4)
    shift_y = random.randint(-4, 4)

    canvas = bg.copy()

    h, w = img.shape[:2]

    x1 = max(0, shift_x)
    y1 = max(0, shift_y)

    x2 = min(IMAGE_SIZE, shift_x + w)
    y2 = min(IMAGE_SIZE, shift_y + h)

    img_x1 = max(0, -shift_x)
    img_y1 = max(0, -shift_y)

    img_x2 = img_x1 + (x2 - x1)
    img_y2 = img_y1 + (y2 - y1)

    canvas[y1:y2, x1:x2] = img[img_y1:img_y2, img_x1:img_x2]

    return canvas


def process_icon(icon_path, class_name):
    icon = cv2.imread(str(icon_path), cv2.IMREAD_UNCHANGED)
    if icon is None:
        return

    # Handle alpha channel
    if icon.shape[2] == 4:
        alpha = icon[:, :, 3] / 255.0
        rgb = icon[:, :, :3]
        bg = np.zeros_like(rgb)
        for c in range(3):
            bg[:, :, c] = alpha * rgb[:, :, c]
        icon = bg
    else:
        icon = icon[:, :, :3]

    train_dir = Path(OUTPUT_DIR) / "train" / class_name
    val_dir = Path(OUTPUT_DIR) / "val" / class_name
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    base_name = icon_path.stem

    for i in range(SAMPLES_PER_ICON):

        # Slight random scale variation
        scale_factor = random.uniform(0.7, 1.3)
        scaled = cv2.resize(icon, None, fx=scale_factor, fy=scale_factor)
        img = resize_with_padding(scaled, IMAGE_SIZE)
        img = augment_icon(img)
        img = place_on_background(img)

        if random.random() < TRAIN_SPLIT:
            out_path = train_dir / f"{base_name}_{i}.png"
        else:
            out_path = val_dir / f"{base_name}_{i}.png"

        cv2.imwrite(str(out_path), img)

def main():
    for base_dir in INPUT_DIRS:
        for file in os.listdir(base_dir):

            if not file.lower().endswith((".png", ".webp", ".jpg", ".jpeg")):
                continue

            icon_path = Path(base_dir) / file

            # Class name = filename without extension
            class_name = icon_path.stem

            process_icon(icon_path, class_name)

    print("Synthetic dataset generation complete.")


if __name__ == "__main__":
    main()
