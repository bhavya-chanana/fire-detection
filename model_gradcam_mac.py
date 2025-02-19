import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import ssl

# Disable SSL verification for downloading pretrained models
ssl._create_default_https_context = ssl._create_unverified_context

# -----------------------------
# Configuration
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = "/Users/karanchanana/Bhavya/fire-detection/processed_data"
PROCESSED_IMAGES_DIR = os.path.join(PROCESSED_DIR, "images")
PROCESSED_LABELS_FILE = os.path.join(PROCESSED_DIR, "labels.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results-50")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics-50")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# Training parameters
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
SEED = 42

# Device configuration for Apple Silicon
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

# Set random seeds for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

# -----------------------------
# Dataset Class
# -----------------------------
class FireSmokeDataset(Dataset):
    def __init__(self, labels_csv, processed_dir):
        self.annotations = pd.read_csv(labels_csv)
        self.processed_dir = processed_dir

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
        
        # Validation phase with metrics
        metrics = calculate_metrics(model, val_loader, criterion)
        val_losses.append(metrics['avg_loss'])
        
        # Plot and save metrics
        plot_metrics(metrics, epoch)
        
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {metrics["avg_loss"]:.4f}')
        
        # Save best model
        if metrics['avg_loss'] < best_val_loss:
            best_val_loss = metrics['avg_loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
                'train_losses': train_losses,
                'val_losses': val_losses
            }, os.path.join(RESULTS_DIR, 'best_model.pth'))
        
        # Generate and save Grad-CAM visualization
        if epoch % 5 == 0:  # Generate every 5 epochs
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
# Metrics Calculation and Plotting
# -----------------------------
def calculate_metrics(model, data_loader, criterion):
    """Calculate performance metrics"""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Calculating metrics"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            preds = torch.sigmoid(outputs) >= 0.5
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    metrics = {}
    classes = ['Fire', 'Smoke']
    
    for i, class_name in enumerate(classes):
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels[:, i], all_preds[:, i], average='binary'
        )
        accuracy = accuracy_score(all_labels[:, i], all_preds[:, i])
        
        metrics[class_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        cm = confusion_matrix(all_labels[:, i], all_preds[:, i])
        metrics[f'{class_name}_cm'] = cm
    
    metrics['avg_loss'] = total_loss / len(data_loader)
    return metrics

def plot_metrics(metrics, epoch):
    """Plot and save metrics visualization"""
    # Plot confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    classes = ['Fire', 'Smoke']
    
    for i, class_name in enumerate(classes):
        cm = metrics[f'{class_name}_cm']
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Blues')
        axes[i].set_title(f'{class_name} Confusion Matrix')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig(os.path.join(METRICS_DIR, f'confusion_matrix_epoch_{epoch}.png'))
    plt.close()
    
    # Save metrics to text file
    with open(os.path.join(METRICS_DIR, f'metrics_epoch_{epoch}.txt'), 'w') as f:
        for class_name in classes:
            f.write(f"\n{class_name} Metrics:\n")
            for metric, value in metrics[class_name].items():
                f.write(f"{metric}: {value:.4f}\n")
        f.write(f"\nAverage Loss: {metrics['avg_loss']:.4f}\n")

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    print("\nStarting Fire and Smoke Detection Training")
    print(f"Using device: {DEVICE}")
    print(f"Results will be saved in: {RESULTS_DIR}")
    
    # Load and split dataset
    dataset = FireSmokeDataset(PROCESSED_LABELS_FILE, PROCESSED_IMAGES_DIR)
    
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    # Split dataset
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )
    
    print(f"\nDataset split:")
    print(f"Total samples: {total_size}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4
    )
    
    # Initialize model and training components
    model = FireSmokeModel(pretrained=True).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train model
    print("\nStarting training...")
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS
    )
    
    # Save final training history
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, 'training_history.png'))
    plt.close()
    
    print("\nTraining complete!")
    print(f"Results saved in: {RESULTS_DIR}")
    print(f"Metrics saved in: {METRICS_DIR}")