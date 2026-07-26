import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
from efficientnet_pytorch import EfficientNet
import os
from tqdm import tqdm

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model (binary classification)
model = EfficientNet.from_name("efficientnet-b0")
model._fc = nn.Linear(model._fc.in_features, 1)
model.load_state_dict(torch.load("model/drowsiness_detector_model.pth", map_location=device))
model.to(device)
model.eval()

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Dataset & DataLoader
dataset = datasets.ImageFolder("dataset", transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

# Evaluate
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in tqdm(loader, desc="Evaluating"):
        images = images.to(device)
        outputs = model(images)
        probs = torch.sigmoid(outputs).cpu().numpy().flatten()
        preds = (probs > 0.5).astype(int)
        y_pred.extend(preds)
        y_true.extend(labels.numpy())

# Save results
os.makedirs("outputs", exist_ok=True)
np.save("outputs/y_true.npy", np.array(y_true))
np.save("outputs/y_pred.npy", np.array(y_pred))

print("✅ Evaluation complete. Saved: outputs/y_true.npy and outputs/y_pred.npy")
