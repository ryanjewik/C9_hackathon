import json
from typing import List

import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
except Exception:
    torch = None
    nn = None
    T = None


if torch is not None and nn is not None:
    class IconCNN(nn.Module):
        def __init__(self, num_classes: int):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(128 * 8 * 8, 256)
            self.fc2 = nn.Linear(256, num_classes)

        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = self.pool(torch.relu(self.conv3(x)))
            x = x.view(x.size(0), -1)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x
else:
    IconCNN = None


class IconClassifier:
    """Simple wrapper to load a PyTorch icon classifier and classify BGR numpy images.

    Usage:
        clf = IconClassifier("/path/to/icon_model.pth", "/path/to/class_names.json", device='cpu')
        label = clf.classify(bgr_image)

    The classifier expects small square crops (the wrapper will resize to 64x64).
    """

    def __init__(self, model_path: str, labels_path: str, device: str = 'cpu'):
        if torch is None or IconCNN is None:
            # Graceful fallback: classifier not available in this environment
            raise RuntimeError("PyTorch not available; IconClassifier disabled in this image")
        self.device = torch.device(device)
        with open(labels_path, 'r', encoding='utf-8') as f:
            self.labels: List[str] = json.load(f)
        num_classes = len(self.labels)
        self.model = IconCNN(num_classes)
        state = torch.load(model_path, map_location=self.device)
        # Support either state_dict or full model saved
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        try:
            self.model.load_state_dict(state)
        except Exception:
            # try direct load (when saved via torch.save(model.state_dict()))
            self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        # transforms: BGR->RGB, resize, to tensor, normalize
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _prepare(self, bgr_img: np.ndarray):
        # Convert BGR->RGB
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)
        return tensor

    def classify(self, bgr_img: np.ndarray) -> str:
        """Return the top predicted label (string)."""
        if bgr_img is None:
            return "unknown"
        try:
            x = self._prepare(bgr_img)
            with torch.no_grad():
                logits = self.model(x)
                probs = torch.nn.functional.softmax(logits, dim=1)
                top = torch.argmax(probs, dim=1).item()
            return self.labels[top]
        except Exception:
            return "unknown"

    # provide alternative names for compatibility
    def predict(self, bgr_img: np.ndarray) -> str:
        return self.classify(bgr_img)

    def infer(self, bgr_img: np.ndarray) -> str:
        return self.classify(bgr_img)
