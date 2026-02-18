import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import json
import sys

# -------- CONFIG --------

MODEL_PATH = "icon_model.pth"
CLASS_PATH = "class_names.json"
IMAGE_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------- MODEL DEFINITION --------

class IconCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 55)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# -------- LOAD MODEL --------

with open(CLASS_PATH, "r") as f:
    class_names = json.load(f)

model = IconCNN(len(class_names)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# -------- TRANSFORM --------

transform = transforms.Compose([
    transforms.ToTensor(),
])

# -------- PREPROCESS --------

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

def preprocess_real_crop(path):
    img = cv2.imread(path)

    img = resize_with_padding(img, IMAGE_SIZE)

    return img


# -------- INFERENCE --------

def predict(path):
    img = preprocess_real_crop(path)

    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        top5 = torch.topk(probs, 5)

    print("\nTop 5 Predictions:")
    for i in range(5):
        idx = top5.indices[0][i].item()
        conf = top5.values[0][i].item()
        print(f"{class_names[idx]}: {conf:.4f}")

if __name__ == "__main__":
    path = sys.argv[1]
    predict(path)
