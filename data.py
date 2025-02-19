import ee  # Google Earth Engine
import cdsapi  # Copernicus API for ERA5
import rasterio  # LANDFIRE processing
import geopandas as gpd
import numpy as np
import os
from tqdm import tqdm  # Progress bar
import matplotlib.pyplot as plt
from rasterio.plot import show

# Authenticate Google Earth Engine
try:
    ee.Initialize()
except Exception as e:
    ee.Authenticate()
    ee.Initialize()

# Define point of interest and create a buffer around it
point = ee.Geometry.Point([-118.5149, 34.0240])  # Los Angeles area
roi = point.buffer(5000)  # 5km buffer around the point

def visualize_bands(tif_path):
    """Visualize Sentinel-2 bands in a single image."""
    with rasterio.open(tif_path) as src:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        band_names = ['Blue (B2)', 'Green (B3)', 'Red (B4)', 
                     'NIR (B8)', 'SWIR1 (B11)', 'SWIR2 (B12)']
        
        for idx, (ax, name) in enumerate(zip(axes.flat, band_names)):
            band = src.read(idx + 1)
            im = ax.imshow(band, cmap='viridis')
            ax.set_title(name)
            plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.savefig(f"{os.path.splitext(tif_path)[0]}_visualization.png")
        plt.close()

def export_sentinel2(start_date, end_date, roi):
    """Export Sentinel-2 imagery to Google Drive for download."""
    collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
        .filterDate(start_date, end_date) \
        .filterBounds(roi) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)) \
        .select(["B2", "B3", "B4", "B8", "B11", "B12"])
    
    image = collection.median().clip(roi)  # Get median composite
    
    task = ee.batch.Export.image.toDrive(
        image=image,
        description="Sentinel2_Fire_Detection",
        folder="EarthEngineData",
        fileNamePrefix="sentinel2_fire",
        scale=10,
        region=roi.getInfo()['coordinates'],
        fileFormat="GeoTIFF"
    )
    
    task.start()
    print("Sentinel-2 export started! Check Google Drive for the file.")

# Execute download with visualization
print("Starting Sentinel-2 export...")
export_sentinel2("2024-01-01", "2024-01-29", roi)
print("\nProcess completed! Download the image from Google Drive and visualize it using `visualize_bands()`.")
