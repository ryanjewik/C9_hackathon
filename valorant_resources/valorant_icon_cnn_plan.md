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

# 🚀 RESULT
You now have a real‑time, lightweight, scalable weapon & ability recognition system integrated into your existing killfeed pipeline.

