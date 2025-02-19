import rasterio
import numpy as np
import matplotlib.pyplot as plt

def visualize_bands(tif_path):
    """Visualizes Sentinel-2 bands relevant to fire detection."""
    
    with rasterio.open(tif_path) as src:
        bands = {
            "B2 - Blue": src.read(1),   # Smoke detection
            "B3 - Green": src.read(2),  # Vegetation
            "B4 - Red": src.read(3),    # Burned areas
            "B8 - NIR": src.read(4),    # Healthy vegetation
            "B11 - SWIR1": src.read(5), # Fire & burned land
            "B12 - SWIR2": src.read(6)  # Active fire detection
        }

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        plt.suptitle("Sentinel-2 Bands Visualization", fontsize=16)

        for ax, (name, band) in zip(axes.flat, bands.items()):
            band = np.clip(band, np.percentile(band, 2), np.percentile(band, 98))  # Normalize
            im = ax.imshow(band, cmap='viridis')
            ax.set_title(name)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(f"{tif_path}_visualization.png")  # Save as PNG
        plt.show()

# Example Usage:
visualize_bands("sentinel2_data\sentinel2_2024-01-23.tif")
