
import satellite_images_nso.api.nso_georegion as nso
import satellite_images_nso.api.sat_manipulator as sat_manipulator
import matplotlib.pyplot as plt
import rasterio
import numpy as np
from dotenv import load_dotenv
import os
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.plot import show
# import requests
# from shapely.geometry import box
import json
import time
import zipfile
import requests
import geopandas as gpd
# from pyproj import Transformer
#import contextily as ctx
from tools import *
from illegal import *

def detect_visual_changes_proper(path2, path1, threshold=0.14):
    """
    Better approach using proper geospatial alignment with threshold
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
        
        # Apply threshold - set values below threshold to 0
        change_magnitude_thresholded = change_magnitude.copy()
        change_magnitude_thresholded[change_magnitude_thresholded < threshold] = 0
        
        # Print statistics
        total_pixels = change_magnitude.size
        changed_pixels = (change_magnitude_thresholded > 0).sum()
        percentage_changed = (changed_pixels / total_pixels) * 100
        
        print(f"Threshold applied: {threshold}")
        print(f"Pixels above threshold: {changed_pixels:,} ({percentage_changed:.2f}% of image)")
        print(f"Average change magnitude: {change_magnitude.mean():.4f}")
        print(f"Max change magnitude: {change_magnitude.max():.4f}")
        
        # Return everything needed for plotting
        return {
            'red1_norm': red1_norm,
            'green1_norm': green1_norm,
            'blue1_norm': blue1_norm,
            'red2_norm': red2_norm,
            'green2_norm': green2_norm,
            'blue2_norm': blue2_norm,
            'change_magnitude': change_magnitude,
            'change_magnitude_thresholded': change_magnitude_thresholded,
            'threshold': threshold
        }

# Simple normalization
def normalize(band):
    return (band - band.min()) / (band.max() - band.min() + 1e-10)

def analyze_change_thresholds(change_data):
    change_magnitude = change_data['change_magnitude']

    # Define range of threshold values to test
    thresholds = np.linspace(0, 0.5, 50)  # Test thresholds from 0 to 0.5
    change_areas = []
    change_percentages = []

    total_pixels = change_magnitude.size

    # Calculate change area for each threshold
    for threshold in thresholds:
        # Count pixels above threshold
        changed_pixels = (change_magnitude > threshold).sum()
        change_areas.append(changed_pixels)
        change_percentages.append(changed_pixels / total_pixels * 100)

    # Create the threshold sensitivity plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Number of changed pixels vs threshold
    ax1.plot(thresholds, change_areas, 'b-', linewidth=2, marker='o', markersize=3)
    ax1.set_xlabel('Threshold Value')
    ax1.set_ylabel('Number of Changed Pixels')
    ax1.set_title('Changed Pixels vs Threshold')
    ax1.grid(True, alpha=0.3)

    # Add some reference lines
    ax1.axhline(y=total_pixels*0.01, color='r', linestyle='--', alpha=0.7, label='1% of image')
    ax1.axhline(y=total_pixels*0.05, color='orange', linestyle='--', alpha=0.7, label='5% of image')
    ax1.axhline(y=total_pixels*0.10, color='g', linestyle='--', alpha=0.7, label='10% of image')
    ax1.axhline(y=total_pixels*0.20, color='g', linestyle='--', alpha=0.7, label='20% of image')
    ax1.legend()

    # Plot 2: Percentage of image changed vs threshold
    ax2.plot(thresholds, change_percentages, 'r-', linewidth=2, marker='s', markersize=3)
    ax2.set_xlabel('Threshold Value')
    ax2.set_ylabel('Percentage of Image Changed (%)')
    ax2.set_title('Change Percentage vs Threshold')
    ax2.grid(True, alpha=0.3)

    # Add reference lines
    ax2.axhline(y=1, color='r', linestyle='--', alpha=0.7, label='1%')
    ax2.axhline(y=5, color='orange', linestyle='--', alpha=0.7, label='5%')
    ax2.axhline(y=10, color='g', linestyle='--', alpha=0.7, label='10%')
    ax2.axhline(y=20, color='g', linestyle='--', alpha=0.7, label='20%')
    ax2.legend()

    plt.tight_layout()
    plt.show()

    # Print some key statistics
    print(f"\nThreshold Analysis Results:")
    print(f"Total pixels in image: {total_pixels:,}")
    print(f"Maximum change magnitude: {change_magnitude.max():.3f}")
    print(f"Mean change magnitude: {change_magnitude.mean():.3f}")

    # Find interesting threshold points
    interesting_thresholds = [0, 0.05, 0.1, 0.15, 0.2, 0.25]
    print(f"\nChange area at different thresholds:")
    for thresh in interesting_thresholds:
        changed_pixels = (change_magnitude > thresh).sum()
        percentage = changed_pixels / total_pixels * 100
        print(f"Threshold {thresh:.2f}: {changed_pixels:,} pixels ({percentage:.2f}% of image)")

def convert_geojson_to_rd_polygon(geojson_path):
    """Convert GeoJSON file directly to RD polygon WKT for BGT API"""
    print(f"🔄 Loading GeoJSON and converting to RD...")
    
    try:
        # Load the GeoJSON file
        alkmaar_gdf = gpd.read_file(geojson_path)
        print(f"📍 Loaded GeoJSON: {len(alkmaar_gdf)} features")
        print(f"   Original CRS: {alkmaar_gdf.crs}")
        
        # Convert to RD (EPSG:28992) which is what BGT API expects
        if alkmaar_gdf.crs and alkmaar_gdf.crs.to_epsg() != 28992:
            alkmaar_rd = alkmaar_gdf.to_crs('EPSG:28992')
            print(f"   Converted to RD (EPSG:28992)")
        else:
            alkmaar_rd = alkmaar_gdf.copy()
        
        # Get the geometry (assuming it's the first feature for municipality boundary)
        geometry_rd = alkmaar_rd.geometry.iloc[0]
        
        # Convert to WKT format that BGT API expects
        polygon_wkt = geometry_rd.wkt
        
        print(f"✅ Created RD polygon WKT")
        print(f"   WKT length: {len(polygon_wkt)} characters")
        
        return polygon_wkt
            
    except Exception as e:
        print(f"❌ Error converting GeoJSON: {str(e)}")
        return None

def create_bgt_download_request(polygon_wkt):
    """Create BGT download request using KNOWN working parameters"""
    base_url = 'https://api.pdok.nl/lv/bgt/download/v1_0/full/custom'
    
    # Use EXACT feature types from working API documentation
    featuretypes = [
        "bak",
        "gebouwinstallatie", 
        "kunstwerkdeel",
        "onbegroeidterreindeel"
    ]
    
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }

    payload = {
        "featuretypes": featuretypes,
        "format": "citygml",
        "geofilter": polygon_wkt
    }
    
    print(f"📤 Creating BGT download request...")
    print(f"🏗️ Requesting features: {featuretypes}")
    print(f"🔧 Using format: citygml")
    
    try:
        response = requests.post(base_url, headers=headers, data=json.dumps(payload), timeout=60)
        
        if response.status_code == 202:
            response_data = response.json()
            download_request_id = response_data.get('downloadRequestId')
            print(f"✅ Download Request ID: {download_request_id}")
            return download_request_id
        else:
            print(f"❌ Request failed: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error creating download request: {str(e)}")
        return None

def check_bgt_download_status(download_request_id, max_wait_minutes=20):
    """Check BGT download status until complete"""
    base_url = 'https://api.pdok.nl/lv/bgt/download/v1_0/full/custom'
    status_url = f'{base_url}/{download_request_id}/status'
    
    max_checks = max_wait_minutes * 2  # Check every 30 seconds
    
    print(f"⏳ Checking download status (max {max_wait_minutes} minutes)...")
    
    for check in range(max_checks):
        try:
            response = requests.get(status_url, timeout=30)
            
            # FIXED: Accept both 200 and 201 status codes
            if response.status_code in [200, 201]:
                status_data = response.json()
                status = status_data.get('status', 'unknown')
                
                if status == 'COMPLETED':
                    download_link = status_data.get('_links', {}).get('download', {}).get('href')
                    if download_link:
                        print(f"✅ Download ready!")
                        return download_link
                    else:
                        print(f"❌ Download completed but no download link found")
                        return None
                        
                elif status == 'FAILED':
                    print(f"❌ Download failed: {status_data}")
                    return None
                    
                else:
                    # Still processing
                    elapsed = (check + 1) * 30
                    print(f"   ⏳ Status: {status} (HTTP {response.status_code}) (elapsed: {elapsed//60}m {elapsed%60}s)")
                    time.sleep(30)
                    
            else:
                print(f"❌ Status check failed: {response.status_code}")
                time.sleep(30)
                
        except Exception as e:
            print(f"⚠️ Status check error: {str(e)}")
            time.sleep(30)
    
    print(f"❌ Download timed out after {max_wait_minutes} minutes")
    return None


def download_bgt_file(download_link, output_folder):
    """Download BGT ZIP file"""
    download_url = f'https://api.pdok.nl{download_link}'
    zip_filename = f'bgt_alkmaar_{int(time.time())}.zip'
    zip_path = os.path.join(output_folder, zip_filename)
    
    print(f"📥 Downloading BGT file...")
    
    try:
        os.makedirs(output_folder, exist_ok=True)
        
        response = requests.get(download_url, stream=True, timeout=300)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r   Progress: {percent:.1f}%", end='')
        
        print(f"\n✅ Download complete: {zip_path}")
        return zip_path
        
    except Exception as e:
        print(f"❌ Download failed: {str(e)}")
        return None

def extract_and_process_bgt_data(zip_path, output_folder):
    """Extract and process BGT data from ZIP file"""
    print(f"📂 Extracting BGT data...")
    
    extract_folder = os.path.join(output_folder, 'bgt_extracted')
    os.makedirs(extract_folder, exist_ok=True)
    
    try:
        # Extract ZIP file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)
            
        print(f"✅ Extracted to: {extract_folder}")
        
        # Find GML files
        bgt_files = []
        for root, dirs, files in os.walk(extract_folder):
            for file in files:
                if file.endswith(('.gml', '.xml')):
                    bgt_files.append(os.path.join(root, file))
                    
        print(f"📋 Found {len(bgt_files)} BGT data files")
        
        if not bgt_files:
            print("⚠️ No GML files found")
            return gpd.GeoDataFrame()
        
        # Load and combine all BGT data
        all_gdfs = []
        
        for file_path in bgt_files:
            try:
                print(f"   Loading: {os.path.basename(file_path)}")
                
                # Try to load with different drivers
                gdf = None
                for driver in ['GML', 'GMLAS']:
                    try:
                        gdf = gpd.read_file(file_path, driver=driver)
                        if not gdf.empty:
                            print(f"     ✅ Loaded {len(gdf)} features with {driver}")
                            break
                    except Exception:
                        continue
                
                if gdf is not None and not gdf.empty:
                    gdf['source_file'] = os.path.basename(file_path)
                    all_gdfs.append(gdf)
                    
            except Exception as e:
                print(f"     ❌ Error loading {file_path}: {str(e)}")
        
        if all_gdfs:
            # Combine all data
            combined_gdf = gpd.pd.concat(all_gdfs, ignore_index=True)
            
            # Convert to WGS84 if needed
            if combined_gdf.crs and combined_gdf.crs.to_epsg() != 4326:
                print(f"🔄 Converting to WGS84...")
                combined_gdf = combined_gdf.to_crs('EPSG:4326')
            
            print(f"✅ Combined BGT data: {len(combined_gdf)} features")
            return combined_gdf
        else:
            print("❌ No valid BGT data could be loaded")
            return gpd.GeoDataFrame()
            
    except Exception as e:
        print(f"❌ Error processing BGT data: {str(e)}")
        return gpd.GeoDataFrame()

def download_bgt_for_alkmaar(geojson_path, output_folder):
    """Complete BGT download workflow for Alkmaar using GeoJSON directly"""
    print("🏗️ Starting BGT data download for Alkmaar...")
    
    # Step 1: Convert GeoJSON to RD polygon WKT
    try:
        polygon_wkt = convert_geojson_to_rd_polygon(geojson_path)
        if not polygon_wkt:
            print("❌ Failed to convert GeoJSON to polygon")
            return gpd.GeoDataFrame()
    except Exception as e:
        print(f"❌ GeoJSON conversion failed: {str(e)}")
        return gpd.GeoDataFrame()
    
    # Step 2: Create download request
    download_request_id = create_bgt_download_request(polygon_wkt)
    if not download_request_id:
        print("❌ Failed to create download request")
        return gpd.GeoDataFrame()
    
    # Step 3: Wait for completion
    download_link = check_bgt_download_status(download_request_id, max_wait_minutes=20)
    if not download_link:
        print("❌ Download preparation failed")
        return gpd.GeoDataFrame()
    
    # Step 4: Download file
    zip_path = download_bgt_file(download_link, output_folder)
    if not zip_path:
        print("❌ File download failed")
        return gpd.GeoDataFrame()
    
    # Step 5: Extract and process
    bgt_gdf = extract_and_process_bgt_data(zip_path, output_folder)
    
    # Step 6: Save and cleanup
    if not bgt_gdf.empty:
        output_path = os.path.join(output_folder, 'bgt_alkmaar_processed.geojson')
        bgt_gdf.to_file(output_path, driver='GeoJSON')
        print(f"💾 Saved processed BGT data to: {output_path}")
        
        # Clean up ZIP file
        try:
            os.remove(zip_path)
            print(f"🗑️ Cleaned up ZIP file")
        except:
            pass
    
    return bgt_gdf

def filter_bgt_data(bgt_data, path_geojson):

    bgt_data_between_2019_2022 = bgt_data[(bgt_data["creationDate"] < "2022-05-01") & (bgt_data["creationDate"] > "2019-03-01")]

    # Load the Alkmaar boundary
    alkmaar_boundary = gpd.read_file(path_geojson)

    # Make sure both datasets have the same CRS
    if bgt_data_between_2019_2022.crs != alkmaar_boundary.crs:
        print(f"Converting BGT data from {bgt_data_between_2019_2022.crs} to {alkmaar_boundary.crs}")
        bgt_data_between_2019_2022 = bgt_data_between_2019_2022.to_crs(alkmaar_boundary.crs)

    # Clip BGT data to Alkmaar boundary - this removes everything outside Alkmaar
    bgt_data_between_2019_2022_clipped = gpd.clip(bgt_data_between_2019_2022, alkmaar_boundary)
    return bgt_data_between_2019_2022_clipped

def plot_bgt_on_satellite_with_boundary(tiff_path, bgt_gdf, title="BGT Features on Satellite Image"):
    """Plot BGT features on satellite image using accurate boundary for filtering"""
    
    if bgt_gdf.empty:
        print("⚠️ No BGT data to plot")
        return
    
    # Open the satellite image
    with rasterio.open(tiff_path) as src:
        # Read RGB bands
        red = src.read(3).astype('float32')
        green = src.read(2).astype('float32') 
        blue = src.read(1).astype('float32')
        
        # Normalize bands
        max_val = max(red.max(), green.max(), blue.max())
        red_norm = red / max_val
        green_norm = green / max_val
        blue_norm = blue / max_val
        
        # Stack to RGB
        rgb = np.stack([red_norm, green_norm, blue_norm], axis=-1)
        
        # Get image properties
        img_bounds = src.bounds
        img_crs = src.crs
        
        print(f"🗺️ Satellite image info:")
        print(f"   CRS: {img_crs}")
        print(f"   Bounds: {img_bounds}")
    
    # Convert BGT data to same CRS as satellite image
    if bgt_gdf.crs != img_crs:
        print(f"🔄 Converting BGT data from {bgt_gdf.crs} to {img_crs}")
        bgt_gdf_proj = bgt_gdf.to_crs(img_crs)
    else:
        bgt_gdf_proj = bgt_gdf.copy()
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(15, 15))
    
    # Display the satellite image
    img_minx, img_miny, img_maxx, img_maxy = img_bounds
    ax.imshow(rgb, extent=[img_minx, img_maxx, img_miny, img_maxy], aspect='equal')
    
    # ⭐ NEW: Filter by source_file instead of geometry type
    if len(bgt_gdf_proj) > 0 and 'source_file' in bgt_gdf_proj.columns:
        # Get unique source files
        unique_sources = bgt_gdf_proj['source_file'].unique()
        
        print(f"📋 Found {len(unique_sources)} different BGT source files:")
        for source in unique_sources:
            count = len(bgt_gdf_proj[bgt_gdf_proj['source_file'] == source])
            print(f"   {source}: {count} features")
        
        # Define colors for different source files
        colors = ['lime', 'blue', 'pink', 'cyan', 'yellow']
        legend_handles = []
        
        for i, source_file in enumerate(unique_sources):
            # Filter data for this source file
            source_data = bgt_gdf_proj[bgt_gdf_proj['source_file'] == source_file]
            color = colors[i % len(colors)]  # Cycle through colors if more sources than colors
            
            if len(source_data) > 0:
                # Plot this source file's features
                source_data.plot(
                    ax=ax,
                    color=color,
                    alpha=0.7,
                    linewidth=1,
                    markersize=20,  # For points
                    label=f'{source_file} ({len(source_data)})'
                )
                
                # Create legend handle based on geometry type
                geom_types = source_data.geometry.geom_type.unique()
                if any(gt in ['Polygon', 'MultiPolygon'] for gt in geom_types):
                    # Use patch for polygon-like features
                    from matplotlib.patches import Patch
                    legend_handles.append(Patch(facecolor=color, alpha=0.7, label=f'{source_file} ({len(source_data)})'))
                else:
                    # Use line/point for other features
                    from matplotlib.lines import Line2D
                    legend_handles.append(Line2D([0], [0], marker='o', color=color, linewidth=2, 
                                               markersize=8, alpha=0.7, label=f'{source_file} ({len(source_data)})'))
        
        # Add the legend
        if legend_handles:
            ax.legend(handles=legend_handles, loc='upper right', framealpha=0.9, fontsize=9, 
                     bbox_to_anchor=(1, 1))
        
        print(f"✅ Plotted {len(bgt_gdf_proj)} BGT features (separated by source file)")
    else:
        print("⚠️ No BGT features found or no 'source_file' column available")
    
        # Set plot properties to satellite image bounds
    ax.set_xlim(img_minx, img_maxx)
    ax.set_ylim(img_miny, img_maxy)
    
    # Remove axis, ticks, and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax

def plot_combined_changes_analysis(tiff_path_2022, tif_path_2019, bgt_gdf, change_data, title="Combined Legal and Visual Changes Analysis"):
    """Plot BGT legal changes and visual changes on satellite image"""
    
    if bgt_gdf.empty:
        print("⚠️ No BGT data to plot")
        return
    
    # Open the satellite image
    with rasterio.open(tiff_path_2022) as src:
        # Read RGB bands
        red = src.read(3).astype('float32')
        green = src.read(2).astype('float32') 
        blue = src.read(1).astype('float32')
        
        # Normalize bands
        max_val = max(red.max(), green.max(), blue.max())
        red_norm = red / max_val
        green_norm = green / max_val
        blue_norm = blue / max_val
        
        # Stack to RGB
        rgb = np.stack([red_norm, green_norm, blue_norm], axis=-1)
        
        # Get image properties
        img_bounds = src.bounds
        img_crs = src.crs
        transform = src.transform
        img_shape = (src.height, src.width)
        
        print(f"🗺️ Satellite image info:")
        print(f"   CRS: {img_crs}")
        print(f"   Bounds: {img_bounds}")
        print(f"   Shape: {img_shape}")
    
    # Convert BGT data to same CRS as satellite image
    if bgt_gdf.crs != img_crs:
        print(f"🔄 Converting BGT data from {bgt_gdf.crs} to {img_crs}")
        bgt_gdf_proj = bgt_gdf.to_crs(img_crs)
    else:
        bgt_gdf_proj = bgt_gdf.copy()
    
    # ⭐ FIX: Regenerate change detection data specifically for this satellite image
    print("🔄 Regenerating change detection data for proper alignment...")
    
    # Use the global tif_path_2019 and ensure alignment with current tiff_path (which should be tiff_path_2022)
    fresh_change_data = detect_visual_changes_proper(tif_path_2019, tiff_path_2022)
    change_magnitude_aligned = fresh_change_data['change_magnitude_thresholded']
    
    print(f"✅ Change data shape: {change_magnitude_aligned.shape}")
    print(f"✅ Satellite image shape: {img_shape}")
    
    # Verify alignment
    if change_magnitude_aligned.shape != img_shape:
        print(f"❌ Shape mismatch! Change: {change_magnitude_aligned.shape}, Image: {img_shape}")
        return None, None
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 16))
    
    # Display the satellite image as base layer
    img_minx, img_miny, img_maxx, img_maxy = img_bounds
    ax.imshow(rgb, extent=[img_minx, img_maxx, img_miny, img_maxy], aspect='equal')
    
    # Layer 1: Visual changes (underneath BGT) - using the same coordinate system as satellite
    print("🔄 Adding visual changes layer with proper CRS alignment...")
    change_overlay = ax.imshow(
        change_magnitude_aligned, 
        extent=[img_minx, img_maxx, img_miny, img_maxy],  # Use satellite image bounds
        cmap='Reds', 
        alpha=0.6,  # Semi-transparent so BGT shows on top
        vmin=0, 
        vmax=0.5,
        aspect='equal',
        origin='upper'  # Ensure proper orientation
    )
    
    # Layer 2: BGT legal changes (on top) - no black borders
    if len(bgt_gdf_proj) > 0 and 'source_file' in bgt_gdf_proj.columns:
        print("🔄 Adding BGT legal changes layer...")
        
        # Get unique source files
        unique_sources = bgt_gdf_proj['source_file'].unique()
        
        print(f"📋 Found {len(unique_sources)} different BGT source files:")
        for source in unique_sources:
            count = len(bgt_gdf_proj[bgt_gdf_proj['source_file'] == source])
            print(f"   {source}: {count} features")
        
        # Define colors for BGT features (bright colors to stand out)
        bgt_colors = ['lime', 'cyan', 'blue', 'purple', 'orange']
        legend_handles = []
        
        for i, source_file in enumerate(unique_sources):
            # Filter data for this source file
            source_data = bgt_gdf_proj[bgt_gdf_proj['source_file'] == source_file]
            color = bgt_colors[i % len(bgt_colors)]
            
            if len(source_data) > 0:
                # Plot BGT features WITHOUT black borders for cleaner look
                source_data.plot(
                    ax=ax,
                    facecolor=color,
                    edgecolor=color,  # Same color as face for seamless look
                    alpha=0.8,  # High visibility
                    linewidth=0.5,  # Minimal border
                    markersize=25,
                    label=f'Legal: {source_file} ({len(source_data)})'
                )
                
                # Create legend handle
                from matplotlib.patches import Patch
                legend_handles.append(
                    Patch(facecolor=color, alpha=0.8, 
                          label=f'Legal: {source_file} ({len(source_data)})')
                )
        
        # Add change detection to legend
        legend_handles.append(
            Patch(facecolor='red', alpha=0.6, label='Visual Changes (Potential Illegal)')
        )
        
        # Add the legend
        if legend_handles:
            ax.legend(handles=legend_handles, loc='upper left', framealpha=0.9, 
                     fontsize=10, bbox_to_anchor=(0, 1))
        
        print(f"✅ Plotted {len(bgt_gdf_proj)} BGT legal changes over visual changes")
    else:
        print("⚠️ No BGT features found or no 'source_file' column available")
    
    # Set plot properties to match satellite image bounds exactly
    ax.set_xlim(img_minx, img_maxx)
    ax.set_ylim(img_miny, img_maxy)
    
    # Add title and remove axes
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    
    # Add colorbar for change intensity
    cbar = plt.colorbar(change_overlay, ax=ax, shrink=0.6, aspect=20, pad=0.02)
    cbar.set_label('Visual Change Intensity', rotation=270, labelpad=20)
    
    plt.tight_layout()
    plt.show()
    
    # Print analysis summary using fresh data
    print(f"\n📊 Combined Analysis Summary:")
    print(f"   Legal changes (BGT): {len(bgt_gdf_proj)} features")
    visual_changes_pixels = (change_magnitude_aligned > 0).sum()
    total_pixels = change_magnitude_aligned.size
    visual_change_percentage = (visual_changes_pixels / total_pixels) * 100
    print(f"   Visual changes: {visual_changes_pixels:,} pixels ({visual_change_percentage:.2f}% of image)")
    print(f"   🔍 Red areas show potential illegal changes")
    print(f"   🟢 Colored polygons show legal BGT changes")
    print(f"   ⚖️ Areas with both may indicate legal changes that are visually detectable")
    
    return fig, ax

def plot_changes_and_bgt_overlay(tiff_path, bgt_gdf, tif_path_2019, title="Legal vs Potential Illegal Changes (2019-2022)"):
    """Plot change detection and BGT data without satellite background"""
    
    if bgt_gdf.empty:
        print("⚠️ No BGT data to plot")
        return None, None
    
    # Get satellite image info for proper CRS and bounds
    with rasterio.open(tiff_path) as src:
        satellite_crs = src.crs
        img_bounds = src.bounds
        img_shape = (src.height, src.width)
    
    # Convert BGT data to satellite CRS (same as other cells)
    if bgt_gdf.crs != satellite_crs:
        print(f"🔄 Converting BGT data from {bgt_gdf.crs} to {satellite_crs}")
        bgt_converted = bgt_gdf.to_crs(satellite_crs)
    else:
        bgt_converted = bgt_gdf.copy()
    
    # Generate fresh change detection data for proper alignment
    print("🔄 Generating change detection data for overlay...")
    fresh_change_data = detect_visual_changes_proper(tif_path_2019, tiff_path)
    change_magnitude_aligned = fresh_change_data['change_magnitude_thresholded']
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Layer 1: Change detection data as base layer
    img_minx, img_miny, img_maxx, img_maxy = img_bounds
    change_overlay = ax.imshow(
        change_magnitude_aligned, 
        extent=[img_minx, img_maxx, img_miny, img_maxy],
        cmap='Reds', 
        alpha=0.8,  # More visible since no satellite background
        vmin=0, 
        vmax=0.5,
        aspect='equal',
        origin='upper'
    )
    
    # Layer 2: BGT legal changes on top
    if 'source_file' in bgt_converted.columns:
        unique_sources = bgt_converted['source_file'].unique()
        
        print(f"📋 Found {len(unique_sources)} different BGT source files:")
        for source in unique_sources:
            count = len(bgt_converted[bgt_converted['source_file'] == source])
            print(f"   {source}: {count} features")
        
        # Define colors for BGT features (bright colors to stand out)
        bgt_colors = ['lime', 'cyan', 'yellow', 'magenta', 'orange']
        legend_handles = []
        
        for i, source_file in enumerate(unique_sources):
            # Filter data for this source file
            source_data = bgt_converted[bgt_converted['source_file'] == source_file]
            color = bgt_colors[i % len(bgt_colors)]
            
            if len(source_data) > 0:
                # Plot BGT features
                source_data.plot(
                    ax=ax,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.9,  # High visibility
                    linewidth=0.5,
                    markersize=25,
                    label=f'Legal: {source_file} ({len(source_data)})'
                )
                
                # Create legend handle
                from matplotlib.patches import Patch
                legend_handles.append(
                    Patch(facecolor=color, alpha=0.9, 
                          label=f'Legal: {source_file} ({len(source_data)})')
                )
        
        # Add change detection to legend
        legend_handles.append(
            Patch(facecolor='red', alpha=0.8, label='Visual Changes (Potential Illegal)')
        )
        
        # Add the legend
        if legend_handles:
            ax.legend(handles=legend_handles, loc='upper left', framealpha=0.9, 
                     fontsize=10, bbox_to_anchor=(0, 1))
    
    # Set plot properties
    ax.set_xlim(img_minx, img_maxx)
    ax.set_ylim(img_miny, img_maxy)
    
    # Add title and remove axes
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    
    # Add colorbar for change intensity
    cbar = plt.colorbar(change_overlay, ax=ax, shrink=0.6, aspect=20, pad=0.02)
    cbar.set_label('Visual Change Intensity', rotation=270, labelpad=20)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    visual_changes_pixels = (change_magnitude_aligned > 0).sum()
    total_pixels = change_magnitude_aligned.size
    visual_change_percentage = (visual_changes_pixels / total_pixels) * 100
    
    print(f"\n📊 Combined Changes Analysis:")
    print(f"   Legal changes (BGT): {len(bgt_converted)} features")
    print(f"   Visual changes: {visual_changes_pixels:,} pixels ({visual_change_percentage:.2f}% of area)")
    print(f"   🔍 Red areas show potential illegal changes")
    print(f"   🟢 Colored polygons show legal BGT changes")
    
    return fig, ax
