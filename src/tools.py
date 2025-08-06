from pathlib import Path
import geopandas as gpd
import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling


def convert_geojson_to_wgs84(filename, input_path=None, output_path=None):
    """
    Convert a GeoJSON file to WGS84 coordinate reference system (EPSG:4326).
    """
    current_dir = Path(__file__).resolve().parent
    data_dir = current_dir.parent / 'data'
    
    # Load your geojson (with unknown CRS)
    gdf = gpd.read_file(data_dir / filename)

    if gdf.crs is None:
        raise ValueError("No CRS found in the file.")

    # Convert to WGS84 (EPSG:4326)
    gdf_wgs84 = gdf.to_crs(epsg=4326)

    if output_path is None:
        base, ext = os.path.splitext(data_dir / filename)
        output_path = f"{base}_wgs84{ext}"

    # Save the new geojson file
    gdf_wgs84.to_file(str(output_path), driver="GeoJSON")

    print("Conversion complete, new file saved as:", output_path)

def get_geojson_path(filename):
    current_dir = Path(__file__).resolve().parent
    data_file_path = current_dir.parent / 'data' / filename
    
    if not data_file_path.exists():
        raise FileNotFoundError(f"GeoJSON file '{filename}' not found at: {data_file_path}")
   
    print(f"GeoJSON file path: {data_file_path}")
    return str(data_file_path)

def get_tif_path(filename=None, ndvi=True):
    """
    Get the path to a .tif file in the 'data' directory.
    - If filename is provided, return that specific file path.
    - If ndvi is True, return the NDVI tif file path.
    - If exclude_ndvi is True, return a .tif file without 'ndvi' in its name.
    """
    current_dir = Path(__file__).resolve().parent
    data_dir = current_dir.parent / 'data'

    # If a specific filename is provided, return its path
    if filename:
        return str(data_dir / filename)

    # If 'exclude_ndvi' is True, return a .tif file without 'ndvi' in its name
    if not ndvi:
        for file in data_dir.iterdir():
            if file.suffix.lower() == '.tif' and 'ndvi' not in file.name.lower():
                return str(file)
        raise FileNotFoundError("No .tif file without 'ndvi' in the name was found.")
    
    # If ndvi=True, return a .tif file that contains 'ndvi' in its name
    for file in data_dir.iterdir():
        if file.suffix.lower() == '.tif' and 'ndvi' in file.name.lower():
            return str(file)
    
    # If no 'ndvi' .tif file is found, raise an error
    raise FileNotFoundError("No .tif file with 'ndvi' in the name was found.")



def detect_visual_changes_proper(path1, path2):
    """
    Better approach using proper geospatial alignment
    """
    with rasterio.open(path1) as src1, rasterio.open(path2) as src2:
        # Check if they have the same CRS and extent
        if src1.crs != src2.crs or src1.bounds != src2.bounds or src1.shape != src2.shape:
            print("Reprojecting to align images properly...")
            
            # Create array to hold reprojected data
            aligned_data = np.empty((src2.count, src1.height, src1.width), dtype=src2.dtypes[0])
            
            # Reproject src2 to match src1's grid
            reproject(
                source=rasterio.band(src2, [1, 2, 3]),  # RGB bands
                destination=aligned_data,
                src_transform=src2.transform,
                src_crs=src2.crs,
                dst_transform=src1.transform,
                dst_crs=src1.crs,
                resampling=Resampling.bilinear
            )
            
            # Now both images are properly aligned
            red1 = src1.read(3).astype(np.float32)
            green1 = src1.read(2).astype(np.float32)
            blue1 = src1.read(1).astype(np.float32)
            
            red2 = aligned_data[2]  # Band 3 (red)
            green2 = aligned_data[1]  # Band 2 (green)
            blue2 = aligned_data[0]  # Band 1 (blue)
        else:
             # Images already aligned
            red1 = src1.read(3).astype(np.float32)
            green1 = src1.read(2).astype(np.float32)
            blue1 = src1.read(1).astype(np.float32)
            
            red2 = src2.read(3).astype(np.float32)
            green2 = src2.read(2).astype(np.float32)
            blue2 = src2.read(1).astype(np.float32)

        # Normalize and calculate changes
        red1_norm = normalize(red1)
        green1_norm = normalize(green1)
        blue1_norm = normalize(blue1)
        
        red2_norm = normalize(red2)
        green2_norm = normalize(green2)
        blue2_norm = normalize(blue2)
        
        # Calculate differences
        red_diff = red2_norm - red1_norm
        green_diff = green2_norm - green1_norm
        blue_diff = blue2_norm - blue1_norm
        
        # Overall change magnitude
        change_magnitude = np.sqrt(red_diff**2 + green_diff**2 + blue_diff**2)
        
        # Return everything needed for plotting
        return {
            'red1_norm': red1_norm,
            'green1_norm': green1_norm,
            'blue1_norm': blue1_norm,
            'red2_norm': red2_norm,
            'green2_norm': green2_norm,
            'blue2_norm': blue2_norm,
            'change_magnitude': change_magnitude
        }

# Simple normalization
def normalize(band):
    return (band - band.min()) / (band.max() - band.min() + 1e-10)