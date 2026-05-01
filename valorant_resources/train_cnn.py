import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler, Dataset
import os
import json
import shutil
import numpy as np
import cv2
from pathlib import Path
from collections import Counter
from PIL import Image as PILImage
import random


# Canonical team background colours in BGR — must match synthetic_data.py.
_TEAM_COLORS_BGR = np.array([
    [159, 189, 112],   # teal
    [ 86,  73, 210],   # red
], dtype=np.float32)


def _snap_to_team_color_bgr(color: np.ndarray) -> np.ndarray:
    """Return the canonical team BGR colour closest to `color`."""
    dists = np.linalg.norm(_TEAM_COLORS_BGR - color.astype(np.float32), axis=1)
    return _TEAM_COLORS_BGR[int(np.argmin(dists))].astype(np.uint8)


def _dominant_team_color_bgr(bgr_img: np.ndarray, sat_threshold: int = 40) -> np.ndarray:
    """Majority vote among saturated pixels across the full image → canonical team colour.
    The large flat background region dominates by pixel count over any thin
    contaminated border strip, so this is robust to kill-feed row bleed."""
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    sat_mask = hsv[:, :, 1] > sat_threshold
    if sat_mask.any():
        pixels = bgr_img[sat_mask].astype(np.float32)
        dists = np.linalg.norm(
            pixels[:, None, :] - _TEAM_COLORS_BGR[None, :, :], axis=2
        )
        winner = int(np.bincount(np.argmin(dists, axis=1), minlength=len(_TEAM_COLORS_BGR)).argmax())
        return _TEAM_COLORS_BGR[winner].astype(np.uint8)
    edges = np.concatenate(
        [bgr_img[0, :], bgr_img[-1, :], bgr_img[:, 0], bgr_img[:, -1]], axis=0
    )
    return _snap_to_team_color_bgr(np.median(edges, axis=0))


def _letterbox_pil(pil_img, out_w, out_h, pad_color_bgr=None):
    """Letterbox a PIL image to out_w x out_h, padding with the median
    edge-pixel color — identical to _letterbox_bgr used at inference time.
    pad_color_bgr: optional BGR uint8 array to override auto-detected padding."""
    bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    scale = min(out_w / w, out_h / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    if pad_color_bgr is None:
        pad_color_bgr = _dominant_team_color_bgr(bgr)
    canvas = np.full((out_h, out_w, 3), pad_color_bgr, dtype=np.uint8)
    x0 = (out_w - nw) // 2
    y0 = (out_h - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return PILImage.fromarray(rgb)


class LetterboxTransform:
    """Applied as the first step for real crops so they match inference preprocessing."""
    def __init__(self, out_w, out_h):
        self.out_w = out_w
        self.out_h = out_h

    def __call__(self, pil_img):
        return _letterbox_pil(pil_img, self.out_w, self.out_h)


def _content_zoom_pil(pil_img: PILImage.Image, min_fill: float = 0.6) -> PILImage.Image:
    """
    PIL-space equivalent of _content_zoom_bgr used at inference time.
    After letterboxing, if the icon occupies less than min_fill of the canvas
    width, detect the non-background content bbox, crop to it with a 15%
    margin, then resize back to the original canvas size.
    Must be applied after LetterboxTransform so the image is already at
    IMG_W × IMG_H and the background is a flat team colour.
    """
    arr = np.array(pil_img.convert("RGB"))
    h, w = arr.shape[:2]

    # Majority vote among saturated pixels → canonical team colour (BGR),
    # then convert to RGB for diff against the PIL (RGB) array.
    bg_bgr = _dominant_team_color_bgr(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))

    # Content mask: exclude any pixel close to EITHER canonical team colour.
    # This strips the primary background AND victim-team bleed on the right edge.
    # _TEAM_COLORS_BGR in RGB space: reverse channel order.
    team_colors_rgb = _TEAM_COLORS_BGR[:, ::-1]  # (2, 3) RGB
    dists_per_team = np.linalg.norm(
        arr.astype(np.float32)[:, :, np.newaxis, :] - team_colors_rgb[np.newaxis, np.newaxis, :, :],
        axis=3,
    )  # shape (h, w, 2) — per-team distances kept for bleed detection
    dists_to_canonical = dists_per_team.min(axis=2)  # shape (h, w)
    mask = dists_to_canonical > 28

    cols = np.any(mask, axis=0)
    if not cols.any():
        return pil_img

    x1, x2 = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])
    if (x2 - x1 + 1) >= min_fill * w:
        return pil_img

    rows = np.any(mask, axis=1)
    y1 = int(np.where(rows)[0][0])  if rows.any() else 0
    y2 = int(np.where(rows)[0][-1]) if rows.any() else h - 1

    # Cap right margin before victim-team bleed zone so the 15% padding
    # doesn't pull bleed pixels into the re-letterboxed crop.
    # bg_bgr is BGR; _TEAM_COLORS_BGR is also BGR — compare directly.
    killer_idx = int(np.argmin(np.linalg.norm(_TEAM_COLORS_BGR - bg_bgr.astype(np.float32), axis=1)))
    victim_idx = 1 - killer_idx
    # dists_per_team is in RGB space but distances are symmetric — victim_idx
    # still correctly selects the opposite team's distance channel.
    victim_col_frac = (dists_per_team[:, :, victim_idx] < 45).mean(axis=0)  # (w,)
    right_victim = np.where(victim_col_frac[x2 + 1:] > 0.25)[0]
    bleed_start = (x2 + 1 + int(right_victim[0])) if len(right_victim) > 0 else w

    content_w = x2 - x1 + 1
    mx = max(2, int(0.15 * content_w))
    my = max(2, int(0.15 * (y2 - y1 + 1)))
    x1 = max(0, x1 - mx)
    x2 = max(x1 + 1, min(bleed_start - 1, x2 + mx))  # stop before bleed zone
    y1 = max(0, y1 - my);  y2 = min(h - 1, y2 + my)

    cropped = arr[y1:y2 + 1, x1:x2 + 1]
    if cropped.size == 0:
        return pil_img
    return _letterbox_pil(PILImage.fromarray(cropped), w, h, pad_color_bgr=bg_bgr)


class ContentZoomTransform:
    """After letterboxing, zoom into non-background content bbox if the icon
    fills less than min_fill of canvas width.  Ensures small ability icons
    (Paint_Shells, NULL-cmd, Showstopper, …) fill the CNN input instead of
    appearing as a tiny central badge in a sea of background padding.
    Applied to both synthetic and real crops so training matches inference."""
    def __init__(self, min_fill: float = 0.6):
        self.min_fill = min_fill

    def __call__(self, pil_img: PILImage.Image) -> PILImage.Image:
        return _content_zoom_pil(pil_img, self.min_fill)


class SharpenTransform:
    """Unsharp mask sharpening — enhances icon edges blurred by broadcast
    compression.  Particularly important for small (~32×32) ability icons
    in the 128×32 canvas where edge detail is critical for classification."""
    def __init__(self, amount: float = 0.8, kernel_size: int = 3, sigma: float = 1.0):
        self.amount = amount
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, pil_img: PILImage.Image) -> PILImage.Image:
        arr = np.array(pil_img.convert("RGB")).astype(np.float32)
        blurred = cv2.GaussianBlur(arr, (self.kernel_size, self.kernel_size), self.sigma)
        sharpened = np.clip(
            arr * (1.0 + self.amount) - blurred * self.amount, 0, 255
        ).astype(np.uint8)
        return PILImage.fromarray(sharpened)


class CLAHETransform:
    """CLAHE (Contrast-Limited Adaptive Histogram Equalization) in LAB space.
    Recovers local contrast crushed by broadcast JPEG encoding — particularly
    critical for ability icon ring edges and inner symbol detail in real crops.
    Must be applied consistently at training and inference."""
    def __init__(self, clip_limit: float = 2.0, tile_grid: tuple = (4, 4)):
        self.clip_limit = clip_limit
        self.tile_grid = tile_grid

    def __call__(self, pil_img: PILImage.Image) -> PILImage.Image:
        arr = np.array(pil_img.convert("RGB"))
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return PILImage.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


# ---------------- CONFIG ----------------

DATA_DIR = "dataset"
BATCH_SIZE = 64
EPOCHS = 30
LR = 0.001
# Landscape crop dimensions matching real killfeed crops (~4:1 ratio)
IMG_W = 128
IMG_H = 32
# Max real samples per class — caps dominant classes (Phantom/Vandal) so they
# don't drown out rare ones.  Set to None to use all real samples.
MAX_REAL_PER_CLASS = 60
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------------------

train_transform = transforms.Compose([
    transforms.Resize((IMG_H, IMG_W)),
    # Zoom small ability icons (already-sized synthetic crops) to fill canvas
    ContentZoomTransform(min_fill=0.6),
    # No horizontal flip — weapons face a specific direction in the killfeed
    # Random sharpening: teaches model to handle both blurry (broadcast compression)
    # and sharp versions of icons — critical for small ability icons.
    transforms.RandomApply([SharpenTransform(amount=1.2)], p=0.5),
    transforms.RandomAffine(
        degrees=3,                        # small rotation helps with real crop tilt
        translate=(0.06, 0.06),
        scale=(0.85, 1.15)
    ),
    transforms.ColorJitter(brightness=0.3, contrast=0.25, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    # Randomly mask small regions — prevents the model from keying on a single
    # salient shape (solves knife/pistol and horizontal-stripe/rifle ambiguity)
    transforms.RandomErasing(p=0.35, scale=(0.02, 0.15), ratio=(0.3, 3.0), value=0),
])

# Real crops are raw killfeed extracts at variable sizes — letterbox first
# (same as _letterbox_bgr at inference) so aspect ratio is preserved, then
# zoom to content, then apply the same augmentations as synthetic.
real_train_transform = transforms.Compose([
    LetterboxTransform(IMG_W, IMG_H),
    ContentZoomTransform(min_fill=0.6),
    CLAHETransform(),
    transforms.RandomApply([SharpenTransform(amount=1.2)], p=0.5),
    transforms.RandomAffine(degrees=3, translate=(0.06, 0.06), scale=(0.85, 1.15)),
    transforms.ColorJitter(brightness=0.3, contrast=0.25, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.35, scale=(0.02, 0.15), ratio=(0.3, 3.0), value=0),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_H, IMG_W)),
    # Zoom small ability icons — must match inference pipeline
    ContentZoomTransform(min_fill=0.6),
    # CLAHE: recover local contrast crushed by broadcast JPEG — must match inference.
    CLAHETransform(),
    # Deterministic sharpening — matches what icon_classifier.py applies at inference.
    SharpenTransform(amount=1.2),
    transforms.ToTensor(),
])

# Real val: letterbox + zoom + CLAHE + sharpen (matching inference) then to tensor — no augmentation
real_val_transform = transforms.Compose([
    LetterboxTransform(IMG_W, IMG_H),
    ContentZoomTransform(min_fill=0.6),
    CLAHETransform(),
    SharpenTransform(amount=1.2),
    transforms.ToTensor(),
])

# ---- Remap dataset labels to a global class index ----

class LabelRemapDataset(Dataset):
    """Wraps an ImageFolder and remaps its local label indices to a global
    class list so that ConcatDataset produces consistent labels."""
    def __init__(self, dataset, label_map):
        self.dataset = dataset
        self.label_map = label_map

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return img, self.label_map[label]

# ---- Split real data into train/val, then cap train portion ----
# Hold out REAL_VAL_PER_CLASS samples per class for a genuine real-world val set.
# The remaining samples are capped at MAX_REAL_PER_CLASS for training.

REAL_VAL_PER_CLASS = 15   # held-out real samples per class for honest val metrics

real_dir         = Path(DATA_DIR) / "real"
capped_real_dir  = Path(DATA_DIR) / "real_capped"
real_val_dir     = Path(DATA_DIR) / "real_val"

for out_dir in [capped_real_dir, real_val_dir]:
    if out_dir.exists():
        shutil.rmtree(out_dir)

for cls_dir in sorted(real_dir.iterdir()):
    if not cls_dir.is_dir():
        continue
    files = sorted(cls_dir.glob("*"))
    n = len(files)
    # For classes with fewer samples than REAL_VAL_PER_CLASS, take at most
    # 20% for val (min 1) so the class still has training exposure.
    # Without this, Tour_de_force (10 samples) gets 0 training samples.
    if n >= REAL_VAL_PER_CLASS:
        n_val = REAL_VAL_PER_CLASS
    else:
        n_val = max(1, n // 5)   # 20% val, at least 1
    val_files   = files[-n_val:]
    train_files = files[:-n_val] if n_val < n else []
    if MAX_REAL_PER_CLASS is not None:
        train_files = train_files[:MAX_REAL_PER_CLASS]

    for f in train_files:
        dst = capped_real_dir / cls_dir.name
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy(f, dst / f.name)
    for f in val_files:
        dst = real_val_dir / cls_dir.name
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy(f, dst / f.name)

real_source = str(capped_real_dir)

synthetic_dataset  = datasets.ImageFolder(
    os.path.join(DATA_DIR, "train"),
    transform=train_transform
)
real_dataset       = datasets.ImageFolder(real_source,        transform=real_train_transform)
syn_val_raw        = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_transform)
real_val_raw       = datasets.ImageFolder(str(real_val_dir),  transform=real_val_transform)

# Build unified sorted class list across all splits
all_classes = sorted(set(synthetic_dataset.classes)
                     | set(real_dataset.classes)
                     | set(syn_val_raw.classes)
                     | set(real_val_raw.classes))
num_classes = len(all_classes)
global_idx = {cls: i for i, cls in enumerate(all_classes)}

syn_map      = {i: global_idx[c] for i, c in enumerate(synthetic_dataset.classes)}
real_map     = {i: global_idx[c] for i, c in enumerate(real_dataset.classes)}
syn_val_map  = {i: global_idx[c] for i, c in enumerate(syn_val_raw.classes)}
real_val_map = {i: global_idx[c] for i, c in enumerate(real_val_raw.classes)}

syn_remapped      = LabelRemapDataset(synthetic_dataset, syn_map)
real_remapped     = LabelRemapDataset(real_dataset,      real_map)
syn_val_remapped  = LabelRemapDataset(syn_val_raw,       syn_val_map)
real_val_remapped = LabelRemapDataset(real_val_raw,      real_val_map)

train_dataset = ConcatDataset([syn_remapped, real_remapped])
# Val = synthetic val + held-out real samples for honest real-world accuracy
val_dataset   = ConcatDataset([syn_val_remapped, real_val_remapped])

# ---- Weighted sampler for class balance ----
# Use remapped (global) labels for weight calculation
all_labels = (
    [syn_map[label]  for _, label in synthetic_dataset.samples]
    + [real_map[label] for _, label in real_dataset.samples]
)
class_counts = Counter(all_labels)
weights = [1.0 / class_counts[label] for label in all_labels]
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE)

real_val_count = len(real_val_raw)
syn_val_count  = len(syn_val_raw)
print(f"Classes ({num_classes}):", all_classes)
print(f"Train samples: {len(train_dataset)}  |  Val samples: {len(val_dataset)} "
      f"(synthetic={syn_val_count}, real={real_val_count})")

# ---------------- MODEL ----------------
# Pool schedule: MaxPool2d(2)×2 then MaxPool2d((1,2)) (width-only).
# W: 128->64->32->16   H: 32->16->8->8  =>  128 * 8 * 16 = 16384
# Halving height only twice (not three times) preserves 8 rows for small
# square ability icons (32×32 in canvas) vs the 4 rows in the old 3×MaxPool2d(2).

class IconCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                       # 64x16

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                       # 32x8

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),       # 32x8 -> 16x8 (width-only pool)
            # Ability icons (32×32 in canvas) now occupy 4×8 in feature map
            # instead of 4×4 — doubles spatial info for contour detection.
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


model = IconCNN(num_classes).to(DEVICE)

# label_smoothing=0.05 penalises overconfident predictions slightly — the model was
# reaching 91-96% confidence on wrong classes (Overdrive, Resurrection)
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ---------------- TRAIN LOOP ----------------

best_val_acc = 0.0

for epoch in range(EPOCHS):

    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    scheduler.step()
    train_acc = 100. * correct / total

    # -------- VALIDATION --------

    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            val_correct += predicted.eq(labels).sum().item()
            val_total += labels.size(0)

    val_acc = 100. * val_correct / val_total

    print(f"Epoch {epoch+1}/{EPOCHS} | LR {scheduler.get_last_lr()[0]:.5f} | "
          f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    # Save best checkpoint so we don't end up with an overfit final epoch
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "icon_model_best.pth")
        print(f"  -> New best val acc {val_acc:.2f}% — checkpoint saved")

# Replace final model with best checkpoint
import shutil
shutil.copy("icon_model_best.pth", "icon_model.pth")
print(f"\nBest val acc: {best_val_acc:.2f}% — saved as icon_model.pth")

with open("class_names.json", "w") as f:
    json.dump(all_classes, f)
print("Class names saved as class_names.json")

# ---- Per-class val accuracy (best model) ----
model.eval()
cls_correct: Counter = Counter()
cls_total:   Counter = Counter()
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        pred = model(images).argmax(dim=1)
        for p, t in zip(pred.cpu().tolist(), labels.cpu().tolist()):
            cls_total[t] += 1
            if p == t:
                cls_correct[t] += 1
failing = [(all_classes[i], 100.0 * cls_correct[i] / cls_total[i], cls_correct[i], cls_total[i])
           for i in range(num_classes) if cls_total[i] > 0 and cls_correct[i] < cls_total[i]]
if failing:
    print("\nClasses below 100% val accuracy:")
    for cls, acc, c, tot in sorted(failing, key=lambda x: x[1]):
        print(f"  {cls:25s}  {acc:5.1f}%  ({c}/{tot})")
else:
    print("\nAll classes at 100% val accuracy.")

