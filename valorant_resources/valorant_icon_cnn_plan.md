# Valorant Killfeed Icon Recognition — System Plan

## 🎯 Goal
Add weapon and ability recognition to an existing killfeed CV pipeline using a lightweight CNN classifier.

---

# 🧩 SYSTEM ARCHITECTURE

```
Game Frame Capture
        ↓
Killfeed Detection (existing CV)
        ↓
Killfeed Row Extraction
        ↓
Row Split → [killer] [ICON] [victim]
        ↓
Icon Crop (32–64 px region)
        ↓
CNN Classifier (weapon/ability)
        ↓
Event Builder (killer + victim + weapon)
        ↓
Database / Analytics
```

The ML model only classifies the icon — all game logic stays outside ML.

---

# 🧠 CNN MODEL ARCHITECTURE

**Input:** 64×64 RGB image (cropped icon)

| Layer | Output Size |
|------|-------------|
| Conv2D (32, 3×3) + ReLU | 64×64×32 |
| MaxPool (2×2) | 32×32×32 |
| Conv2D (64, 3×3) + ReLU | 32×32×64 |
| MaxPool | 16×16×64 |
| Conv2D (128, 3×3) + ReLU | 16×16×128 |
| MaxPool | 8×8×128 |
| Flatten | 8192 |
| Fully Connected (256) + ReLU | 256 |
| Output Layer (N classes) | N |
| Softmax | Probabilities |

**Why this works:**
- Icons are flat UI graphics
- No perspective or 3D deformation
- Small network = fast + low data requirement

---

# 📦 DATASET PLAN (CLASSIFICATION)

We use **image classification**, not object detection.
Each image contains only **one cropped icon**.

```
dataset/
  train/
    vandal/
    phantom/
    operator/
    sheriff/
    ghost/
    spectre/
    odin/
    judge/
    classic/
    knife/
    jett_ult/
    sova_arrow/
    raze_ult/
  val/
    same folders...
```

Folder name = class label.

---

# 🔢 DATA REQUIREMENTS

| Class Type | Samples per Class |
|------------|-------------------|
| Common weapons | 200–400 |
| Rare weapons | 100–200 |
| Abilities | 150–300 |

Because icons are simple, this is enough for strong accuracy.

---

# 🎨 SYNTHETIC DATA GENERATION

To reduce manual labeling, generate augmented images from clean icon assets.

### Simulated variations:
- Brightness shifts (HUD lighting)
- Blur (motion/compression)
- Random backgrounds
- Slight scaling differences

Workflow:
```
Clean PNG icon → Augment → Paste on random background → Save to class folder
```

This can generate thousands of training samples automatically.

---

# 🔄 DATA FLOW DURING INFERENCE

```
Killfeed row → Crop icon → Resize to 64×64 → Tensor → CNN → Class name
```

---

# 🔢 MODEL QUANTIZATION (FOR SPEED)

After training, convert model to INT8 for faster CPU inference.

Benefits:
- ~4× smaller model
- Lower latency
- Minimal accuracy loss

Quantization is done after training and does not change pipeline logic.

---

# ⚙️ WHY CLASSIFICATION INSTEAD OF YOLO

| Detection Model | Our CNN |
|-----------------|--------|
| Finds object location | Location already known |
| Needs bounding boxes | Just folders of images |
| Heavier model | Lightweight |
| More data | Less data |

---

# 🧠 FINAL DESIGN PHILOSOPHY

This system separates responsibilities:

| Component | Responsibility |
|-----------|---------------|
| CV | Find killfeed rows + icon region |
| CNN | Identify which icon it is |
| Logic Layer | Interpret game event |
| Analytics | Stats and tracking |

ML is a plug‑in perception module, not the whole system.

---

# 🧪 IMPLEMENTATION — CNN (PyTorch)

Below is a minimal, production‑ready PyTorch implementation.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class IconCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

---

## 🏋️ Training Script

```python
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
])

train_dataset = torchvision.datasets.ImageFolder("dataset/train", transform=transform)
val_dataset = torchvision.datasets.ImageFolder("dataset/val", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

model = IconCNN(num_classes=len(train_dataset.classes))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "icon_cnn.pth")
```

---

# 🎨 SYNTHETIC DATA GENERATION SCRIPT

This script generates augmented samples from clean PNG icons.

```python
import os
import cv2
import numpy as np
import random

INPUT_DIR = "valorant_resources"
OUTPUT_DIR = "dataset/train"
SAMPLES_PER_ICON = 300

os.makedirs(OUTPUT_DIR, exist_ok=True)

for root, dirs, files in os.walk(INPUT_DIR):
    for file in files:
        if file.endswith(".png"):
            class_name = os.path.basename(root)
            os.makedirs(os.path.join(OUTPUT_DIR, class_name), exist_ok=True)

            icon = cv2.imread(os.path.join(root, file), cv2.IMREAD_UNCHANGED)

            for i in range(SAMPLES_PER_ICON):
                img = icon.copy()

                # Random scale
                scale = random.uniform(0.9, 1.1)
                img = cv2.resize(img, None, fx=scale, fy=scale)

                # Random brightness
                brightness = random.uniform(0.7, 1.3)
                img = np.clip(img * brightness, 0, 255).astype(np.uint8)

                # Add slight blur
                if random.random() > 0.5:
                    img = cv2.GaussianBlur(img, (3, 3), 0)

                # Place on dark background
                bg = np.zeros((64, 64, 3), dtype=np.uint8)
                h, w = img.shape[:2]
                x_offset = (64 - w) // 2
                y_offset = (64 - h) // 2

                if img.shape[2] == 4:
                    alpha = img[:, :, 3] / 255.0
                    for c in range(3):
                        bg[y_offset:y_offset+h, x_offset:x_offset+w, c] = \
                            (alpha * img[:, :, c] + (1 - alpha) * bg[y_offset:y_offset+h, x_offset:x_offset+w, c])
                else:
                    bg[y_offset:y_offset+h, x_offset:x_offset+w] = img

                output_path = os.path.join(OUTPUT_DIR, class_name, f"{i}.png")
                cv2.imwrite(output_path, bg)
```

---

# 🚀 RESULT
You now have:

- A defined CNN architecture
- A working PyTorch training loop
- An automated synthetic dataset generator
- A scalable pipeline ready for quantization + deployment

This moves the project from planning → implementation stage.

