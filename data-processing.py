import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from albumentations import Compose, HorizontalFlip, VerticalFlip, Rotate, Normalize
from tqdm import tqdm

# Add these constants at the top
PROCESSED_DIR = "processed_data"
PROCESSED_IMAGES_DIR = os.path.join(PROCESSED_DIR, "images")
PROCESSED_LABELS_FILE = os.path.join(PROCESSED_DIR, "labels.csv")

# Add device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

def normalize_image(image):
    """Normalize pixel values to [-1, 1] range"""
    return (image / 127.5) - 1.0

def preprocess_and_save_data(csv_path, rgb_dir, thermal_dir):
    """Preprocess images and save them along with labels"""
    
    # Create directories if they don't exist
    os.makedirs(PROCESSED_IMAGES_DIR, exist_ok=True)
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Initialize lists for new labels
    processed_data = []
    
    # Process each image
    print("Processing images...")
    for idx in tqdm(range(len(df))):
        img_name = df.iloc[idx, 0]
        fire_label = df.iloc[idx, 1]
        smoke_label = df.iloc[idx, 2]
        
        # Load and process RGB image
        rgb_path = os.path.join(rgb_dir, img_name)
        rgb_image = cv2.imread(rgb_path)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        rgb_image = cv2.resize(rgb_image, (254, 254))
        
        # Load and process thermal image
        thermal_path = os.path.join(thermal_dir, img_name)
        thermal_image = cv2.imread(thermal_path, cv2.IMREAD_GRAYSCALE)
        thermal_image = cv2.resize(thermal_image, (254, 254))
        
        # Stack images
        thermal_image = np.expand_dims(thermal_image, axis=2)
        fused_image = np.concatenate([rgb_image, thermal_image], axis=2)
        
        # Normalize to [-1, 1]
        fused_image = normalize_image(fused_image.astype(np.float32))
        
        # Save processed image
        processed_filename = f"processed_{os.path.splitext(img_name)[0]}.npy"
        save_path = os.path.join(PROCESSED_IMAGES_DIR, processed_filename)
        np.save(save_path, fused_image)
        
        # Add to processed data list
        processed_data.append({
            'processed_image_path': processed_filename,
            'original_image': img_name,
            'fire_label': fire_label,
            'smoke_label': smoke_label
        })
    
    # Save labels CSV
    processed_df = pd.DataFrame(processed_data)
    processed_df.to_csv(PROCESSED_LABELS_FILE, index=False)
    
    print(f"Preprocessing complete!")
    print(f"Processed images saved in: {PROCESSED_IMAGES_DIR}")
    print(f"Labels saved in: {PROCESSED_LABELS_FILE}")
    
    return processed_df

class ProcessedFireSmokeDataset(Dataset):
    def __init__(self, labels_csv, processed_dir, transform=None):
        self.annotations = pd.read_csv(labels_csv)
        self.processed_dir = processed_dir
        self.transform = transform
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load processed image
        img_path = os.path.join(self.processed_dir, self.annotations.iloc[idx]['processed_image_path'])
        fused_image = np.load(img_path)
        
        # Get labels
        fire_label = self.annotations.iloc[idx]['fire_label']
        smoke_label = self.annotations.iloc[idx]['smoke_label']
        
        # Apply transforms if any
        if self.transform:
            transformed = self.transform(image=fused_image)
            fused_image = transformed['image']
        
        # Convert to tensor and move to device
        fused_image = torch.from_numpy(fused_image.transpose((2, 0, 1))).float().to(self.device)
        labels = torch.tensor([fire_label, smoke_label]).float().to(self.device)

        return fused_image, labels

if __name__ == "__main__":
    # Print CUDA information
    print("\nCUDA Information:")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
        print(f"Device memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Configuration
    CSV_PATH = "flame_dataset.csv"
    RGB_DIR = "D:/VIT/8TH SEM/Capstone/FLAME2-dataset/254p Frame Pairs/254p RGB Images"
    THERMAL_DIR = "D:\VIT\8TH SEM\Capstone\FLAME2-dataset\\254p Frame Pairs\\254p Thermal Images"
    
    # Preprocess and save data
    processed_df = preprocess_and_save_data(CSV_PATH, RGB_DIR, THERMAL_DIR)
    
    print("\nSample of processed data:")
    print(processed_df.head())
    
    # Verify data loading
    sample_dataset = ProcessedFireSmokeDataset(PROCESSED_LABELS_FILE, PROCESSED_IMAGES_DIR)
    sample_image, sample_labels = sample_dataset[0]
    
    print(f"\nVerification:")
    print(f"Loaded image shape: {sample_image.shape}")
    print(f"Loaded labels: {sample_labels}")
    print(f"Image value range: [{sample_image.min():.2f}, {sample_image.max():.2f}]")