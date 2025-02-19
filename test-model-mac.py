import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from tqdm import tqdm
import matplotlib.pyplot as plt

# -----------------------------
# Configuration
# -----------------------------
PROCESSED_DIR = "processed_data"
PROCESSED_IMAGES_DIR = os.path.join(PROCESSED_DIR, "images")
PROCESSED_LABELS_FILE = os.path.join(PROCESSED_DIR, "labels.csv")
BATCH_SIZE = 32
NUM_EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Dataset Class
# -----------------------------
class ProcessedFireSmokeDataset(Dataset):
    def __init__(self, labels_csv, processed_dir):
        self.annotations = pd.read_csv(labels_csv)
        self.processed_dir = processed_dir

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.processed_dir, self.annotations.iloc[idx]['processed_image_path'])
        fused_image = np.load(img_path)
        fused_image = torch.from_numpy(fused_image.transpose((2, 0, 1))).float()
        labels = torch.tensor([self.annotations.iloc[idx]['fire_label'], self.annotations.iloc[idx]['smoke_label']]).float()
        return fused_image, labels

# -----------------------------
# Model Definition
# -----------------------------
class FireSmokeModel(nn.Module):
    def __init__(self, pretrained=True):
        super(FireSmokeModel, self).__init__()
        self.model = models.mobilenet_v3_small(pretrained=pretrained)
        self.model.features[0][0] = nn.Conv2d(4, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.model.classifier[3] = nn.Linear(in_features=1024, out_features=2)

    def forward(self, x):
        return self.model(x)

# -----------------------------
# Grad-CAM Implementation
# -----------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.device = next(model.parameters()).device
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_image, target_class=None):
        self.model.eval()
        input_image = input_image.unsqueeze(0).to(self.device)
        output = self.model(input_image)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        loss = output[0, target_class]
        self.model.zero_grad()
        loss.backward()
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze()
        cam = F.relu(cam).cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        cam = cv2.resize(cam, (254, 254))
        return cam

# -----------------------------
# Visualization Function
# -----------------------------
def visualize_gradcam(image, cam, save_path=None):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    overlay = heatmap + np.float32(image)
    overlay = overlay / np.max(overlay)
    plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    plt.axis('off')
    if save_path:
        plt.savefig(save_path)
    plt.show()

# -----------------------------
# Data Preparation
# -----------------------------
dataset = ProcessedFireSmokeDataset(PROCESSED_LABELS_FILE, PROCESSED_IMAGES_DIR)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# -----------------------------
# Model Training & Grad-CAM
# -----------------------------
model = FireSmokeModel().to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
gradcam = GradCAM(model, model.model.features[-1])

for epoch in range(NUM_EPOCHS):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {loss.item():.4f}")

    # Generate Grad-CAM for a sample image
    sample_image, _ = next(iter(val_loader))
    sample_image = sample_image[0].to(DEVICE)
    cam = gradcam.generate_cam(sample_image, target_class=0)
    visualize_gradcam(sample_image[:3].permute(1, 2, 0).cpu().numpy(), cam, save_path=f"gradcam_epoch_{epoch+1}.png")

print("Training complete with Grad-CAM visualization!")
import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from tqdm import tqdm
import matplotlib.pyplot as plt

# -----------------------------
# Configuration
# -----------------------------
PROCESSED_DIR = "processed_data"
PROCESSED_IMAGES_DIR = os.path.join(PROCESSED_DIR, "images")
PROCESSED_LABELS_FILE = os.path.join(PROCESSED_DIR, "labels.csv")
BATCH_SIZE = 32
NUM_EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Dataset Class
# -----------------------------
class ProcessedFireSmokeDataset(Dataset):
    def __init__(self, labels_csv, processed_dir):
        self.annotations = pd.read_csv(labels_csv)
        self.processed_dir = processed_dir

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.processed_dir, self.annotations.iloc[idx]['processed_image_path'])
        fused_image = np.load(img_path)
        fused_image = torch.from_numpy(fused_image.transpose((2, 0, 1))).float()
        labels = torch.tensor([self.annotations.iloc[idx]['fire_label'], self.annotations.iloc[idx]['smoke_label']]).float()
        return fused_image, labels

# -----------------------------
# Model Definition
# -----------------------------
class FireSmokeModel(nn.Module):
    def __init__(self, pretrained=True):
        super(FireSmokeModel, self).__init__()
        self.model = models.mobilenet_v3_small(pretrained=pretrained)
        self.model.features[0][0] = nn.Conv2d(4, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.model.classifier[3] = nn.Linear(in_features=1024, out_features=2)

    def forward(self, x):
        return self.model(x)

# -----------------------------
# Grad-CAM Implementation
# -----------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.device = next(model.parameters()).device
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_image, target_class=None):
        self.model.eval()
        input_image = input_image.unsqueeze(0).to(self.device)
        output = self.model(input_image)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        loss = output[0, target_class]
        self.model.zero_grad()
        loss.backward()
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze()
        cam = F.relu(cam).cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        cam = cv2.resize(cam, (254, 254))
        return cam

# -----------------------------
# Visualization Function
# -----------------------------
def visualize_gradcam(image, cam, save_path=None):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    overlay = heatmap + np.float32(image)
    overlay = overlay / np.max(overlay)
    plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    plt.axis('off')
    if save_path:
        plt.savefig(save_path)
    plt.show()

# -----------------------------
# Data Preparation
# -----------------------------
dataset = ProcessedFireSmokeDataset(PROCESSED_LABELS_FILE, PROCESSED_IMAGES_DIR)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# -----------------------------
# Model Training & Grad-CAM
# -----------------------------
model = FireSmokeModel().to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
gradcam = GradCAM(model, model.model.features[-1])

for epoch in range(NUM_EPOCHS):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {loss.item():.4f}")

    # Generate Grad-CAM for a sample image
    sample_image, _ = next(iter(val_loader))
    sample_image = sample_image[0].to(DEVICE)
    cam = gradcam.generate_cam(sample_image, target_class=0)
    visualize_gradcam(sample_image[:3].permute(1, 2, 0).cpu().numpy(), cam, save_path=f"gradcam_epoch_{epoch+1}.png")

print("Training complete with Grad-CAM visualization!")
