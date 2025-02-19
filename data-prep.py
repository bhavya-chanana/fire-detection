import ee  # Google Earth Engine
import cdsapi  # Copernicus API for ERA5\import rasterio  # LANDFIRE processing
import geopandas as gpd
import numpy as np
import os
from tqdm import tqdm  # Add this import for progress bar
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show
import requests  # Add requests import

# Authenticate Google Earth Engine
try:
    ee.Initialize()
except Exception as e:
    ee.Authenticate()
    ee.Initialize()

# Define point of interest and create a buffer around it
point = ee.Geometry.Point([-118.655524, 34.085020])  # Los Angeles area
roi = point.buffer(5000)  # 5km buffer around the point

def split_region(roi, max_size=8000):  # Reduced from 32000 to 8000
    """Split large region into smaller tiles."""
    try:
        bounds = roi.bounds().getInfo()
        coords = bounds['coordinates'][0]
        # Extract coordinates correctly from the bounds
        west = min(c[0] for c in coords)
        east = max(c[0] for c in coords)
        south = min(c[1] for c in coords)
        north = max(c[1] for c in coords)
        
        width = abs(east - west)
        height = abs(north - south)
        
        # Calculate number of tiles needed
        cols = max(1, int(np.ceil(width * 111320 / (10 * max_size))))
        rows = max(1, int(np.ceil(height * 111320 / (10 * max_size))))
        
        tiles = []
        for i in range(rows):
            for j in range(cols):
                w = west + j * (width / cols)
                e = west + (j + 1) * (width / cols)
                s = south + i * (height / rows)
                n = south + (i + 1) * (height / rows)
                tile = ee.Geometry.Rectangle([w, s, e, n])
                tiles.append(tile)
        return tiles
    except Exception as e:
        print(f"Error splitting region: {str(e)}")
        # Return a single tile if splitting fails
        return [roi]

def visualize_bands(tif_path):
    """Visualize all bands in the downloaded image."""
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

def download_file(url, output_file):
    """Download file using requests with progress bar."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_file, 'wb') as f, tqdm(
            desc=f"Downloading {os.path.basename(output_file)}",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=8192):
                size = f.write(data)
                pbar.update(size)
        return True
    except Exception as e:
        print(f"Download error: {str(e)}")
        return False

def download_sentinel2(start_date, end_date, roi, output_dir="sentinel2_data"):
    """Download and visualize Sentinel-2 imagery."""
    print("Initiating Sentinel-2 download...")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
            .filterDate(start_date, end_date) \
            .filterBounds(roi) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)) \
            .select(["B2", "B3", "B4", "B8", "B11", "B12"])
        
        # Get list of unique dates
        dates = collection.aggregate_array('system:time_start').distinct().getInfo()
        print(f"Found {len(dates)} dates with available imagery")
        
        for date in dates:
            date_str = ee.Date(date).format('YYYY-MM-dd').getInfo()
            print(f"Processing date: {date_str}")
            
            image = collection \
                .filter(ee.Filter.date(date_str, ee.Date(date_str).advance(1, 'day'))) \
                .first()
            
            url = image.getDownloadURL({
                "scale": 10,  # 10m resolution for detailed view
                "crs": "EPSG:4326",
                "format": "GeoTIFF",
                "region": roi
            })
            
            output_file = f"{output_dir}/sentinel2_{date_str}.tif"
            
            if download_file(url, output_file):
                print(f"Downloaded {date_str} successfully!")
                print(f"Generating visualization for {date_str}...")
                visualize_bands(output_file)
            else:
                print(f"Failed to download {date_str}")
                
    except Exception as e:
        print(f"Error: {str(e)}")

def download_era5(start_date, end_date, variables, output_dir="era5_data"):
    """Download ERA5 weather data from Copernicus API."""
    c = cdsapi.Client()
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": variables,
            "year": start_date[:4],
            "month": start_date[5:7],
            "day": start_date[8:10],
            "time": "12:00",
            "format": "netcdf",
        },
        f"{output_dir}/era5.nc",
    )

def download_landfire(dataset, output_dir="landfire_data"):
    """Download LANDFIRE terrain and vegetation data."""
    urls = {
        "elevation": "https://landfire.gov/download/elevation.tif",
        "fuel_type": "https://landfire.gov/download/fueltype.tif",
        "canopy_cover": "https://landfire.gov/download/canopycover.tif",
    }
    os.system(f"wget -O {output_dir}/{dataset}.tif '{urls[dataset]}'")

# Execute download with visualization
print("Starting data download process...")
download_sentinel2("2024-01-01", "2024-01-29", roi)
print("\nProcess completed! Check the output directory for TIF files and visualizations.")

# download_era5("2024-01-01", "2024-01-31", ["2m_temperature", "10m_u_component_of_wind", "relative_humidity"]) 
# download_landfire("elevation")
# download_landfire("fuel_type")
# download_landfire("canopy_cover")

print("All datasets downloaded successfully!")
