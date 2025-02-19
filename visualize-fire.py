import rasterio
import numpy as np
import matplotlib.pyplot as plt

def visualize_bands(tif_path):
    """Visualizes Sentinel-2 bands relevant to fire detection, including false-color composites and NBR."""
    
    with rasterio.open(tif_path) as src:
        # Read relevant bands
        B2 = src.read(1)  # Blue
        B3 = src.read(2)  # Green
        B4 = src.read(3)  # Red
        B8 = src.read(4)  # Near-Infrared (NIR)
        B11 = src.read(5) # Short-Wave Infrared 1 (SWIR1)
        B12 = src.read(6) # Short-Wave Infrared 2 (SWIR2)

        # Normalize bands for visualization
        def normalize(band):
            return np.clip(band, np.percentile(band, 2), np.percentile(band, 98))

        B2, B3, B4, B8, B11, B12 = map(normalize, [B2, B3, B4, B8, B11, B12])

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        plt.suptitle("Sentinel-2 Fire Detection Bands", fontsize=16)

        # Plot individual bands
        band_dict = {
            "B2 - Blue (Smoke)": B2,
            "B3 - Green (Vegetation)": B3,
            "B4 - Red (Burned Areas)": B4,
            "B8 - NIR (Healthy Vegetation)": B8,
            "B11 - SWIR1 (Burned Areas & Fire)": B11,
            "B12 - SWIR2 (Active Fire)": B12
        }

        for ax, (name, band) in zip(axes.flat, band_dict.items()):
            im = ax.imshow(band, cmap='viridis')
            ax.set_title(name)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(f"{tif_path}_band_visualization.png")  # Save band visualizations
        plt.show()

        # 🔥 **False-Color Fire Visualization (SWIR2, SWIR1, Red)**
        false_color_fire = np.dstack((B12, B11, B4))  # (SWIR2, SWIR1, Red)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(false_color_fire)
        plt.title("🔥 False-Color Fire Detection (SWIR2, SWIR1, Red)")
        plt.axis("off")
        plt.savefig(f"{tif_path}_false_color_fire.png")
        plt.show()

        # 🔥 **Normalized Burn Ratio (NBR) Calculation**
        nbr = (B8 - B12) / (B8 + B12)  # NBR = (NIR - SWIR2) / (NIR + SWIR2)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(nbr, cmap='RdYlBu_r', vmin=-1, vmax=1)
        plt.colorbar(label="Normalized Burn Ratio (NBR)")
        plt.title("🔥 Burn Severity (NBR)")
        plt.axis("off")
        plt.savefig(f"{tif_path}_nbr.png")
        plt.show()

# Example Usage:
visualize_bands("sentinel2_data\sentinel2_2024-01-08.tif")
