import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from albumentations import (
    Compose, HorizontalFlip, VerticalFlip, Rotate, 
    Normalize, RandomBrightnessContrast
)

class FireSmokeDataset(Dataset):
    def __init__(self, csv_file, rgb_dir, thermal_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations
            rgb_dir (string): Directory with all RGB images
            thermal_dir (string): Directory with all thermal images
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.annotations = pd.read_csv(csv_file)
        self.rgb_dir = rgb_dir
        self.thermal_dir = thermal_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get image name and labels
        img_name = self.annotations.iloc[idx, 0]
        fire_label = self.annotations.iloc[idx, 1]
        smoke_label = self.annotations.iloc[idx, 2]

        # Load RGB image
        rgb_path = os.path.join(self.rgb_dir, img_name)
        rgb_image = cv2.imread(rgb_path)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

        # Load thermal image
        thermal_path = os.path.join(self.thermal_dir, img_name)
        thermal_image = cv2.imread(thermal_path, cv2.IMREAD_GRAYSCALE)
        
        # Ensure both images are 254x254
        rgb_image = cv2.resize(rgb_image, (254, 254))
        thermal_image = cv2.resize(thermal_image, (254, 254))

        # Stack images - Early fusion
        thermal_image = np.expand_dims(thermal_image, axis=2)
        fused_image = np.concatenate([rgb_image, thermal_image], axis=2)

        # Apply transformations if any
        if self.transform:
            transformed = self.transform(image=fused_image)
            fused_image = transformed['image']

        # Convert to tensor
        fused_image = torch.from_numpy(fused_image.transpose((2, 0, 1))).float()
        labels = torch.tensor([fire_label, smoke_label]).float()

        return fused_image, labels

# Define transformations
def get_transforms(is_training=True):
    if is_training:
        return Compose([
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            Rotate(limit=30, p=0.5),
            RandomBrightnessContrast(p=0.2),
            Normalize(
                mean=[0.485, 0.456, 0.406, 0.0],  # Adjusted for 4 channels
                std=[0.229, 0.224, 0.225, 1.0]
            )
        ])
    else:
        return Compose([
            Normalize(
                mean=[0.485, 0.456, 0.406, 0.0],
                std=[0.229, 0.224, 0.225, 1.0]
            )
        ])

def create_dataloaders(csv_path, rgb_dir, thermal_dir, batch_size=32):
    """
    Create training and validation dataloaders
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Split into train and validation (90-10 split)
    train_df = df.sample(frac=0.9, random_state=42)
    val_df = df.drop(train_df.index)
    
    # Create temporary CSV files for train and validation
    train_csv = 'train_temp.csv'
    val_csv = 'val_temp.csv'
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    
    # Create datasets
    train_dataset = FireSmokeDataset(
        train_csv, rgb_dir, thermal_dir, 
        transform=get_transforms(is_training=True)
    )
    val_dataset = FireSmokeDataset(
        val_csv, rgb_dir, thermal_dir, 
        transform=get_transforms(is_training=False)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, 
        shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, 
        shuffle=False, num_workers=4
    )
    
    # Clean up temporary files
    os.remove(train_csv)
    os.remove(val_csv)
    
    return train_loader, val_loader

if __name__ == "__main__":
    # Configuration
    CSV_PATH = "flame_dataset.csv"
    RGB_DIR = "rgb_frames"  # Directory containing RGB frames
    THERMAL_DIR = "thermal_frames"  # Directory containing thermal frames
    BATCH_SIZE = 32

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        CSV_PATH, RGB_DIR, THERMAL_DIR, BATCH_SIZE
    )

    # Example of iterating through the dataloader
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"Image shape: {images.shape}")  # Should be [batch_size, 4, 254, 254]
        print(f"Labels shape: {labels.shape}")  # Should be [batch_size, 2]
        
        # Break after first batch for testing
        break