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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = "/Users/karanchanana/Bhavya/fire-detection/processed_data"
PROCESSED_IMAGES_DIR = os.path.join(PROCESSED_DIR, "images")
PROCESSED_LABELS_FILE = os.path.join(PROCESSED_DIR, "labels.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Training parameters
BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 0.001

# Device configuration for Apple Silicon
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

# -----------------------------
# Dataset Class
# -----------------------------
class FireSmokeDataset(Dataset):
    def __init__(self, labels_csv, processed_dir):
        self.annotations = pd.read_csv(labels_csv)
        self.processed_dir = processed_dir
        # Device configuration for Apple Silicon
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.processed_dir, self.annotations.iloc[idx]['processed_image_path'])
        fused_image = np.load(img_path)
        fused_image = torch.from_numpy(fused_image.transpose((2, 0, 1))).float()
        labels = torch.tensor([
            self.annotations.iloc[idx]['fire_label'],
            self.annotations.iloc[idx]['smoke_label']
        ]).float()
        return fused_image, labels

# -----------------------------
# Model Definition
# -----------------------------
class FireSmokeModel(nn.Module):
    def __init__(self, pretrained=True):
        super(FireSmokeModel, self).__init__()
        self.model = models.mobilenet_v3_small(pretrained=pretrained)
        # Modify first conv layer to accept 4 channels (RGB + Thermal)
        self.model.features[0][0] = nn.Conv2d(4, 16, kernel_size=3, stride=2, padding=1, bias=False)
        # Modify classifier for binary classification (fire and smoke)
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
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_image, target_class=None):
        # Generate class activation map
        self.model.eval()
        input_image = input_image.unsqueeze(0)
        
        # Forward pass
        output = self.model(input_image)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        # Backward pass
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # Generate CAM
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        cam = F.relu(cam)
        cam = cam.cpu().numpy()
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
        return cv2.resize(cam, (254, 254))

# -----------------------------
# Training Function
# -----------------------------
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for images, labels in train_bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(RESULTS_DIR, 'best_model.pth'))
            
        # Generate and save Grad-CAM visualization
        if epoch % 2 == 0:  # Generate every 2 epochs
            sample_image, _ = next(iter(val_loader))
            sample_image = sample_image[0].to(DEVICE)
            generate_and_save_gradcam(model, sample_image, epoch)
    
    return train_losses, val_losses

# -----------------------------
# Grad-CAM Visualization
# -----------------------------
def generate_and_save_gradcam(model, image, epoch):
    gradcam = GradCAM(model, model.model.features[-1])
    cam = gradcam.generate_cam(image)
    
    # Convert image for visualization (use first 3 channels for RGB)
    vis_image = image[:3].cpu().numpy().transpose(1, 2, 0)
    vis_image = (vis_image - vis_image.min()) / (vis_image.max() - vis_image.min())
    
    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    overlay = heatmap * 0.7 + vis_image * 0.3
    
    # Save visualization
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 3, 1)
    plt.imshow(vis_image)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(cam, cmap='jet')
    plt.title('Grad-CAM')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title('Overlay')
    plt.axis('off')
    
    plt.savefig(os.path.join(RESULTS_DIR, f'gradcam_epoch_{epoch}.png'))
    plt.close()

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Load dataset
    dataset = FireSmokeDataset(PROCESSED_LABELS_FILE, PROCESSED_IMAGES_DIR)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Initialize model and training components
    model = FireSmokeModel().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train model
    print("Starting training...")
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS
    )
    
    # Save training history
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, 'training_history.png'))
    plt.close()
    
    print("Training complete! Results saved in:", RESULTS_DIR)