import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Constants with macOS style paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)  # Go up one level to fire-detection directory
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")
PROCESSED_IMAGES_DIR = os.path.join(PROCESSED_DIR, "images")
PROCESSED_LABELS_FILE = os.path.join(PROCESSED_DIR, "labels.csv")

# Device configuration for Apple Silicon
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

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
        if rgb_image is None:
            print(f"Warning: Could not load RGB image at {rgb_path}")
            continue
            
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        rgb_image = cv2.resize(rgb_image, (254, 254))
        
        # Load and process thermal image
        thermal_path = os.path.join(thermal_dir, img_name)
        thermal_image = cv2.imread(thermal_path, cv2.IMREAD_GRAYSCALE)
        if thermal_image is None:
            print(f"Warning: Could not load thermal image at {thermal_path}")
            continue
            
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
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load processed image
        img_path = os.path.join(self.processed_dir, self.annotations.iloc[idx]['processed_image_path'])
        try:
            fused_image = np.load(img_path)
        except Exception as e:
            print(f"Error loading image at {img_path}: {str(e)}")
            raise
        
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
    # Print device information
    print("\nDevice Information:")
    print(f"MPS (Apple Silicon) available: {torch.backends.mps.is_available()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Using device: {device}")
    
    # Updated Configuration with correct macOS style paths
    CSV_PATH = os.path.join(BASE_DIR, "flame_dataset.csv")
    RGB_DIR = os.path.join(PARENT_DIR, "254p RGB Images")
    THERMAL_DIR = os.path.join(PARENT_DIR, "254p Thermal Images")
    
    # Create processed_data directory if it doesn't exist
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(PROCESSED_IMAGES_DIR, exist_ok=True)
    
    # Verify paths exist
    for path in [CSV_PATH, RGB_DIR, THERMAL_DIR]:
        if not os.path.exists(path):
            print(f"Error: Path does not exist: {path}")
            print(f"Looking for: {path}")
            print("Please ensure your data directories are correctly set up")
            exit(1)
    
    try:
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
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")