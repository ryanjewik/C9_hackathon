import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import sys
import json


# -------- CONFIG --------

MODEL_PATH = "icon_model.pth"
IMAGE_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------- MODEL --------

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
            nn.Linear(256, 55)  # <-- set manually for now
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = IconCNN(55).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

with open("class_names.json", "r") as f:
    class_names = json.load(f)


transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# -------- INFER --------

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    return class_names[pred.item()], conf.item()


if __name__ == "__main__":
    path = sys.argv[1]
    pred, conf = predict(path)
    print("Predicted class:", pred)
    print("Confidence:", conf)

