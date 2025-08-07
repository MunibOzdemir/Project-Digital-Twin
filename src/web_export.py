# Add these imports to your existing notebook
import json
from pathlib import Path  # Fixed: was "from zipfile import Path"
from PIL import Image
import os
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import shutil

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

def reproject_tiff(input_path, output_path, target_crs='EPSG:4326'):
    """Reproject a TIFF file using rasterio instead of gdalwarp"""
    with rasterio.open(input_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': target_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        with rasterio.open(output_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest)

def export_for_web(tif_path, geojson_path=None, tif_path_2=None):
    """
    Export satellite data and indices as web-compatible images with bounds
    If tif_path_2 is provided, also generate change detection layer
    """

    current_dir = Path(__file__).resolve().parent
    web_folder = current_dir.parent / 'web'

    # Ensure output folder exists
    os.makedirs(web_folder, exist_ok=True)

    # Step 1: Reproject primary TIFF to WGS84 using rasterio instead of gdalwarp
    tif_wgs84_path = os.path.join(web_folder, "reprojected.tif")
    
    try:
        reproject_tiff(tif_path, tif_wgs84_path, target_crs='EPSG:4326')
        print("✅ rasterio: Reprojection successful.")
    except Exception as e:
        print(f"❌ Error running rasterio reprojection: {e}")
        return

    # Read GeoJSON if provided
    geojson_bounds = None
    if geojson_path and os.path.exists(geojson_path):
        with open(geojson_path, 'r') as f:
            geojson_data = json.load(f)
        
        # Extract coordinates from GeoJSON to get bounds
        coordinates = geojson_data['features'][0]['geometry']['coordinates'][0]
        lons = [coord[0] for coord in coordinates]
        lats = [coord[1] for coord in coordinates]
        geojson_bounds = [[min(lats), min(lons)], [max(lats), max(lons)]]
    
    with rasterio.open(tif_wgs84_path) as src:
        # Get bounds in WGS84
        bounds = src.bounds
        transform = src.transform
        
        # Read bands
        red = src.read(3).astype(np.float32)
        green = src.read(2).astype(np.float32) 
        blue = src.read(1).astype(np.float32)
        nir = src.read(4).astype(np.float32)
        
        # Normalize RGB
        max_val = max(red.max(), green.max(), blue.max())
        red_norm = (red / max_val * 255).astype(np.uint8)
        green_norm = (green / max_val * 255).astype(np.uint8)
        blue_norm = (blue / max_val * 255).astype(np.uint8)
                
        # Create mask where all bands are > 0 (valid data)
        mask = (red > 0) & (green > 0) & (blue > 0)
        alpha = (mask * 255).astype(np.uint8)

        # Stack RGBA image
        rgba_array = np.dstack((red_norm, green_norm, blue_norm, alpha))

        # Save as RGBA PNG
        rgb_image = Image.fromarray(rgba_array, mode='RGBA')
        rgb_path = os.path.join(web_folder, "satellite_rgb.png")
        rgb_image.save(rgb_path)

        
        # Calculate and export NDVI (only for primary image)
        ndvi = (nir - red) / (nir + red + 1e-10)
        ndvi_norm = ((ndvi + 1) / 2 * 255).astype(np.uint8)  # Normalize -1,1 to 0,255
        ndvi_colored = plt.cm.RdYlGn(ndvi_norm)[:,:,:3] * 255
        # Use NDVI mask (exclude extreme values)
        mask_ndvi = (ndvi != 0)
        alpha_ndvi = (mask_ndvi * 255).astype(np.uint8)
        ndvi_rgba = np.dstack((ndvi_colored, alpha_ndvi))
        ndvi_image = Image.fromarray(ndvi_rgba.astype(np.uint8), mode='RGBA')
        ndvi_path = os.path.join(web_folder, "ndvi.png")
        ndvi_image.save(ndvi_path)
        
        # Calculate and export NDWI (only for primary image)
        ndwi = (green - nir) / (green + nir + 1e-10)
        ndwi_norm = ((ndwi + 1) / 2 * 255).astype(np.uint8)  # Normalize -1,1 to 0,255
        ndwi_colored = plt.cm.BrBG(ndwi_norm)[:,:,:3] * 255
        # Use NDVI mask (exclude extreme values)
        mask_ndwi = (ndwi != 0)
        alpha_ndwi = (mask_ndwi * 255).astype(np.uint8)
        ndwi_rgba = np.dstack((ndwi_colored, alpha_ndwi))
        ndwi_image = Image.fromarray(ndwi_rgba.astype(np.uint8), mode='RGBA')
        ndwi_path = os.path.join(web_folder, "ndwi.png")
        ndwi_image.save(ndwi_path)
        
        # Initialize bounds info
        bounds_info = {
            "satellite": {
                "path": "satellite_rgb.png",
                "bounds": [[bounds.bottom, bounds.left], [bounds.top, bounds.right]]
            },
            "ndvi": {
                "path": "ndvi.png", 
                "bounds": [[bounds.bottom, bounds.left], [bounds.top, bounds.right]]
            },
            "ndwi": {
                "path": "ndwi.png",
                "bounds": [[bounds.bottom, bounds.left], [bounds.top, bounds.right]]
            }
        }
        
        # If second TIFF is provided, generate change detection layer
        if tif_path_2:
            print("Generating change detection layer...")
            change_data = detect_visual_changes_proper(tif_path, tif_path_2)
            
            # Export change detection as image
            change_magnitude = change_data['change_magnitude']
            
            # Normalize change magnitude to 0-255
            change_norm = ((change_magnitude / change_magnitude.max()) * 255).astype(np.uint8)
            
            # Apply red colormap
            change_colored = plt.cm.Reds(change_norm)[:,:,:3] * 255
            
            # Create transparency mask (threshold for visibility)
            alpha_change = (change_magnitude > 0.01).astype(np.uint8) * 255
            
            # Create RGBA image
            change_rgba = np.dstack((change_colored, alpha_change))
            
            # Save change detection image
            change_image = Image.fromarray(change_rgba.astype(np.uint8), mode='RGBA')
            change_path = os.path.join(web_folder, "change_detection.png")
            change_image.save(change_path)
            
            # Add to bounds info
            bounds_info["change_detection"] = {
                "path": "change_detection.png",
                "bounds": [[bounds.bottom, bounds.left], [bounds.top, bounds.right]]
            }
            
            print("✅ Change detection layer created.")
        
        # Create bounds info for JavaScript
        # bounds_info was already initialized above
        
        # Add GeoJSON boundary if available
        if geojson_bounds:
            bounds_info["geojson_bounds"] = geojson_bounds
            bounds_info["geojson_path"] = geojson_path
        
        # Save bounds info as JSON
        bounds_path = os.path.join(web_folder, "bounds.json")
        with open(bounds_path, 'w') as f:
            json.dump(bounds_info, f, indent=2)

        print(f"Exported images to: {web_folder}")
        print(f"Bounds: {bounds_info['satellite']['bounds']}")
        print("Image bounds:", bounds)
        print("Bounds for Leaflet:", [[bounds.bottom, bounds.left], [bounds.top, bounds.right]])
        
        return bounds_info

# Create a simple HTTP server function for testing
def create_simple_server(web_folder=None, port=8000):
    """
    Create a simple HTTP server to serve the files locally
    """
    import http.server
    import socketserver
    import threading
    import webbrowser
    import os
    from pathlib import Path
    
    current_dir = Path(__file__).resolve().parent
    web_folder = current_dir.parent / 'web'

    # Ensure index.html is in the output folder
    html_path = os.path.join(web_folder, 'index.html')
    if not os.path.exists(html_path):
        print(f"❌ Error: {html_path} not found!")
        return
    
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=web_folder, **kwargs)
        
        def log_message(self, format, *args):
            # Suppress favicon 404 errors in logs
            if "favicon.ico" in args[0] and "404" in args[1]:
                return
            super().log_message(format, *args)
    
    def start_server():
        with socketserver.TCPServer(("", port), Handler) as httpd:
            print(f"Server running at http://localhost:{port}")
            print(f"Files being served from: {web_folder}")
            httpd.serve_forever()
    
    # Start server in background thread
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    
    # Open browser
    webbrowser.open(f'http://localhost:{port}')
    
    return f"http://localhost:{port}"