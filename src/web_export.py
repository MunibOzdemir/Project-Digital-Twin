# Add these imports to your existing notebook
import json
from PIL import Image
import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import shutil

def export_for_web(tif_path, output_folder, geojson_path=None):
    """
    Export satellite data and indices as web-compatible images with bounds
    """

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Step 1: Reproject TIFF to WGS84 using gdalwarp
    tif_wgs84_path = os.path.join(output_folder, "reprojected.tif")
    gdalwarp_cmd = [
        "gdalwarp",
        "-s_srs", "EPSG:28992",   # Assumes source is RD New
        "-t_srs", "EPSG:4326",    # Target WGS84
        tif_path,
        tif_wgs84_path
    ]

    try:
        subprocess.run(gdalwarp_cmd, check=True)
        print("✅ gdalwarp: Reprojection successful.")
    except subprocess.CalledProcessError as e:
        print("❌ Error running gdalwarp:", e)
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
        rgb_path = os.path.join(output_folder, "satellite_rgb.png")
        rgb_image.save(rgb_path)

        
        # Calculate and export NDVI
        ndvi = (nir - red) / (nir + red + 1e-10)
        ndvi_norm = ((ndvi + 1) / 2 * 255).astype(np.uint8)  # Normalize -1,1 to 0,255
        ndvi_colored = plt.cm.RdYlGn(ndvi_norm)[:,:,:3] * 255
        # Use NDVI mask (exclude extreme values)
        mask_ndvi = (ndvi != 0)
        alpha_ndvi = (mask_ndvi * 255).astype(np.uint8)
        ndvi_rgba = np.dstack((ndvi_colored, alpha_ndvi))
        ndvi_image = Image.fromarray(ndvi_rgba.astype(np.uint8), mode='RGBA')
        ndvi_path = os.path.join(output_folder, "ndvi.png")
        ndvi_image.save(ndvi_path)
        
        # Calculate and export NDWI
        ndwi = (green - nir) / (green + nir + 1e-10)
        ndwi_norm = ((ndwi + 1) / 2 * 255).astype(np.uint8)  # Normalize -1,1 to 0,255
        ndwi_colored = plt.cm.BrBG(ndwi_norm)[:,:,:3] * 255
        # Use NDVI mask (exclude extreme values)
        mask_ndwi = (ndwi != 0)
        alpha_ndwi = (mask_ndwi * 255).astype(np.uint8)
        ndwi_rgba = np.dstack((ndwi_colored, alpha_ndwi))
        ndwi_image = Image.fromarray(ndwi_rgba.astype(np.uint8), mode='RGBA')
        ndwi_path = os.path.join(output_folder, "ndwi.png")
        ndwi_image.save(ndwi_path)
        
        # Create bounds info for JavaScript
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
        
        # Add GeoJSON boundary if available
        if geojson_bounds:
            bounds_info["geojson_bounds"] = geojson_bounds
            bounds_info["geojson_path"] = geojson_path
        
        # Save bounds info as JSON
        bounds_path = os.path.join(output_folder, "bounds.json")
        with open(bounds_path, 'w') as f:
            json.dump(bounds_info, f, indent=2)
        
        print(f"Exported images to: {output_folder}")
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