# web_backend.py - Updated backend with illegal construction only
import json
import os
import hashlib
import pickle
import base64
import io
import time
import zipfile
from pathlib import Path
from datetime import datetime, timedelta
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import requests
import geopandas as gpd
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from scipy import ndimage
from skimage import measure

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# ================================
# CONFIGURATION
# ================================

CLIENT_ID = "sh-93c6fbd0-8c4a-4e40-8c59-d06889413797"
CLIENT_SECRET = "LKVq6MTE0S3kohQjRI1Yuj03aU5frOTm"
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"

# ================================
# PROJECT PATHS
# ================================

def get_project_paths():
    """Get correct paths for project directories"""
    current_dir = Path(__file__).parent
    
    if current_dir.name == 'src':
        project_root = current_dir.parent
    else:
        project_root = current_dir
    
    static_dir = project_root / 'web'
    cache_dir = project_root / 'cache'
    data_dir = project_root / 'data'
    
    static_dir.mkdir(exist_ok=True)
    cache_dir.mkdir(exist_ok=True)
    
    return static_dir, cache_dir, data_dir

STATIC_DIR, CACHE_DIR, DATA_DIR = get_project_paths()

print(f"Static files: {STATIC_DIR}")
print(f"Cache directory: {CACHE_DIR}")
print(f"Data directory: {DATA_DIR}")

# ================================
# AUTO-CLEAR CACHE ON STARTUP (FOR TESTING)
# ================================

def clear_cache_on_startup():
    """Clear all cache files on server startup for testing purposes"""
    try:
        cache_files = list(CACHE_DIR.glob('*.pkl'))
        for f in cache_files:
            f.unlink()
        if cache_files:
            print(f"Testing mode: Cleared {len(cache_files)} cache files on startup")
        else:
            print("Testing mode: No cache files to clear")
    except Exception as e:
        print(f"Failed to clear cache on startup: {e}")

# Clear cache on startup for testing
clear_cache_on_startup()

# ================================
# GEOJSON PROCESSING
# ================================

def get_bounds_and_geometry_from_geojson(geojson_path=None):
    """Extract bounds and geometry from alkmaar.geojson file"""
    if geojson_path is None:
        geojson_path = DATA_DIR / 'alkmaar.geojson'
    
    try:
        print(f"Reading GeoJSON from: {geojson_path}")
        with open(geojson_path, 'r') as f:
            geojson_data = json.load(f)
        
        # Handle different GeoJSON structures
        if 'features' in geojson_data:
            geometry = geojson_data['features'][0]['geometry']
        elif 'geometry' in geojson_data:
            geometry = geojson_data['geometry']
        else:
            geometry = geojson_data
        
        print(f"   Geometry type: {geometry['type']}")
        
        # Extract ALL coordinates
        all_coords = []
        
        if geometry['type'] == 'Polygon':
            for ring in geometry['coordinates']:
                all_coords.extend(ring)
        elif geometry['type'] == 'MultiPolygon':
            for polygon in geometry['coordinates']:
                for ring in polygon:
                    all_coords.extend(ring)
        
        # Get bounds
        lons = [coord[0] for coord in all_coords]
        lats = [coord[1] for coord in all_coords]
        
        min_lon = min(lons)
        max_lon = max(lons)
        min_lat = min(lats)
        max_lat = max(lats)
        
        # Format for Leaflet: [[min_lat, min_lon], [max_lat, max_lon]]
        bounds = [[min_lat, min_lon], [max_lat, max_lon]]
        
        print(f"Municipality bounds: {bounds}")
        print(f"   Area: {max_lat-min_lat:.4f}° × {max_lon-min_lon:.4f}°")
        
        return bounds, geometry
        
    except Exception as e:
        print(f"Error reading GeoJSON: {e}")
        # Default gemeente Alkmaar bounds
        default_bounds = [[52.55, 4.65], [52.70, 4.85]]
        default_geometry = {
            "type": "Polygon",
            "coordinates": [[
                [4.65, 52.55], [4.85, 52.55], 
                [4.85, 52.70], [4.65, 52.70], [4.65, 52.55]
            ]]
        }
        return default_bounds, default_geometry

def get_bounds_from_geometry(geometry):
    """Extract bounding box from GeoJSON geometry."""
    if geometry['type'] == 'Polygon':
        coords = geometry['coordinates'][0]
    elif geometry['type'] == 'MultiPolygon':
        coords = []
        for poly in geometry['coordinates']:
            coords.extend(poly[0])
    else:
        raise ValueError(f"Unsupported geometry type: {geometry['type']}")
    
    lons = [coord[0] for coord in coords]
    lats = [coord[1] for coord in coords]
    
    return (min(lons), min(lats), max(lons), max(lats))

# ================================
# SURFACE WATER MASK FROM GEOJSON
# ================================

def load_surface_water_mask():
    """
    Load surface water GeoJSON and prepare it for masking.
    Returns GeoDataFrame with water body geometries.
    """
    surface_water_path = DATA_DIR / 'surface_water.geojson'
    try:
        print(f"Loading surface water from: {surface_water_path}")
        water_gdf = gpd.read_file(surface_water_path)
        # Ensure CRS is WGS84 (EPSG:4326) to match Sentinel Hub data
        if water_gdf.crs != 'EPSG:4326':
            water_gdf = water_gdf.to_crs('EPSG:4326')
        print(f"   Loaded {len(water_gdf)} water body features")
        return water_gdf
    except FileNotFoundError:
        print(f"Surface water file not found at {surface_water_path}")
        print("   Water mask will not be available for NDCI")
        return None
    except Exception as e:
        print(f"Error loading surface water data: {e}")
        return None

def create_water_mask(image_shape, bounds, water_gdf):
    """
    Create a binary mask for water bodies using surface water GeoJSON.
    """
    if water_gdf is None or water_gdf.empty:
        print("No water body data available - returning full mask")
        return np.ones(image_shape, dtype=bool)
    
    height, width = image_shape
    minx, miny, maxx, maxy = bounds
    transform = from_bounds(minx, miny, maxx, maxy, width, height)
    
    print(f"Creating water mask from {len(water_gdf)} water bodies")
    
    water_mask = rasterize(
        [(geom, 1) for geom in water_gdf.geometry if geom is not None],
        out_shape=image_shape,
        transform=transform,
        fill=0,
        dtype='uint8'
    )
    
    water_pixels = np.sum(water_mask)
    total_pixels = image_shape[0] * image_shape[1]
    print(f"   Water mask: {water_pixels}/{total_pixels} pixels ({water_pixels/total_pixels*100:.1f}%)")
    
    return water_mask.astype(bool)

# ================================
# BGT DATA DOWNLOADING
# ================================

def convert_geojson_to_rd_polygon(geojson_path):
    """Convert GeoJSON file directly to RD polygon WKT for BGT API"""
    print(f"Loading GeoJSON and converting to RD...")
    
    try:
        # Load the GeoJSON file
        alkmaar_gdf = gpd.read_file(geojson_path)
        print(f"Loaded GeoJSON: {len(alkmaar_gdf)} features")
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
        
        print(f"Created RD polygon WKT")
        print(f"   WKT length: {len(polygon_wkt)} characters")
        
        return polygon_wkt
            
    except Exception as e:
        print(f"Error converting GeoJSON: {str(e)}")
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
    
    print(f"Creating BGT download request...")
    print(f"Requesting features: {featuretypes}")
    print(f"Using format: citygml")
    
    try:
        response = requests.post(base_url, headers=headers, data=json.dumps(payload), timeout=60)
        
        if response.status_code == 202:
            response_data = response.json()
            download_request_id = response_data.get('downloadRequestId')
            print(f"Download Request ID: {download_request_id}")
            return download_request_id
        else:
            print(f"Request failed: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"Error creating download request: {str(e)}")
        return None

def check_bgt_download_status(download_request_id, max_wait_minutes=20):
    """Check BGT download status until complete"""
    base_url = 'https://api.pdok.nl/lv/bgt/download/v1_0/full/custom'
    status_url = f'{base_url}/{download_request_id}/status'
    
    max_checks = max_wait_minutes * 2  # Check every 30 seconds
    
    print(f"Checking download status (max {max_wait_minutes} minutes)...")
    
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
                        print(f"Download ready!")
                        return download_link
                    else:
                        print(f"Download completed but no download link found")
                        return None
                        
                elif status == 'FAILED':
                    print(f"Download failed: {status_data}")
                    return None
                    
                else:
                    # Still processing
                    elapsed = (check + 1) * 30
                    print(f"   Status: {status} (HTTP {response.status_code}) (elapsed: {elapsed//60}m {elapsed%60}s)")
                    time.sleep(30)
                    
            else:
                print(f"Status check failed: {response.status_code}")
                time.sleep(30)
                
        except Exception as e:
            print(f"Status check error: {str(e)}")
            time.sleep(30)
    
    print(f"Download timed out after {max_wait_minutes} minutes")
    return None

def download_bgt_file(download_link, output_folder):
    """Download BGT ZIP file"""
    download_url = f'https://api.pdok.nl{download_link}'
    zip_filename = f'bgt_alkmaar_{int(time.time())}.zip'
    zip_path = os.path.join(output_folder, zip_filename)
    
    print(f"Downloading BGT file...")
    
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
        
        print(f"\nDownload complete: {zip_path}")
        return zip_path
        
    except Exception as e:
        print(f"Download failed: {str(e)}")
        return None

def extract_and_process_bgt_data(zip_path, output_folder):
    """Extract and process BGT data from ZIP file"""
    print(f"Extracting BGT data...")
    
    extract_folder = os.path.join(output_folder, 'bgt_extracted')
    os.makedirs(extract_folder, exist_ok=True)
    
    try:
        # Extract ZIP file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)
            
        print(f"Extracted to: {extract_folder}")
        
        # Find GML files
        bgt_files = []
        for root, dirs, files in os.walk(extract_folder):
            for file in files:
                if file.endswith(('.gml', '.xml')):
                    bgt_files.append(os.path.join(root, file))
                    
        print(f"Found {len(bgt_files)} BGT data files")
        
        if not bgt_files:
            print("No GML files found")
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
                            print(f"     Loaded {len(gdf)} features with {driver}")
                            break
                    except Exception:
                        continue
                
                if gdf is not None and not gdf.empty:
                    gdf['source_file'] = os.path.basename(file_path)
                    all_gdfs.append(gdf)
                    
            except Exception as e:
                print(f"     Error loading {file_path}: {str(e)}")
        
        if all_gdfs:
            # Combine all data
            combined_gdf = gpd.pd.concat(all_gdfs, ignore_index=True)
            
            # Convert to WGS84 if needed
            if combined_gdf.crs and combined_gdf.crs.to_epsg() != 4326:
                print(f"Converting to WGS84...")
                combined_gdf = combined_gdf.to_crs('EPSG:4326')
            
            print(f"Combined BGT data: {len(combined_gdf)} features")
            return combined_gdf
        else:
            print("No valid BGT data could be loaded")
            return gpd.GeoDataFrame()
            
    except Exception as e:
        print(f"Error processing BGT data: {str(e)}")
        return gpd.GeoDataFrame()

def download_bgt_for_alkmaar(geojson_path, output_folder):
    """Complete BGT download workflow for Alkmaar using GeoJSON directly"""
    print("Starting BGT data download for Alkmaar...")
    
    # Step 1: Convert GeoJSON to RD polygon WKT
    try:
        polygon_wkt = convert_geojson_to_rd_polygon(geojson_path)
        if not polygon_wkt:
            print("Failed to convert GeoJSON to polygon")
            return gpd.GeoDataFrame()
    except Exception as e:
        print(f"GeoJSON conversion failed: {str(e)}")
        return gpd.GeoDataFrame()
    
    # Step 2: Create download request
    download_request_id = create_bgt_download_request(polygon_wkt)
    if not download_request_id:
        print("Failed to create download request")
        return gpd.GeoDataFrame()
    
    # Step 3: Wait for completion
    download_link = check_bgt_download_status(download_request_id, max_wait_minutes=20)
    if not download_link:
        print("Download preparation failed")
        return gpd.GeoDataFrame()
    
    # Step 4: Download file
    zip_path = download_bgt_file(download_link, output_folder)
    if not zip_path:
        print("File download failed")
        return gpd.GeoDataFrame()
    
    # Step 5: Extract and process
    bgt_gdf = extract_and_process_bgt_data(zip_path, output_folder)
    
    # Step 6: Save and cleanup
    if not bgt_gdf.empty:
        output_path = os.path.join(output_folder, 'bgt_alkmaar_processed.geojson')
        bgt_gdf.to_file(output_path, driver='GeoJSON')
        print(f"Saved processed BGT data to: {output_path}")
        
        # Clean up ZIP file
        try:
            os.remove(zip_path)
            print(f"Cleaned up ZIP file")
        except:
            pass
    
    return bgt_gdf

def ensure_bgt_data_available():
    """Ensure BGT data is available, download if necessary"""
    bgt_processed_path = DATA_DIR / 'bgt_alkmaar_processed.geojson'
    alkmaar_geojson_path = DATA_DIR / 'alkmaar.geojson'
    
    # Check if processed BGT data already exists
    if bgt_processed_path.exists():
        print(f"BGT data already available at: {bgt_processed_path}")
        return True
    
    # Check if we have the municipality boundary to work with
    if not alkmaar_geojson_path.exists():
        print(f"Municipality boundary file not found at: {alkmaar_geojson_path}")
        print("Cannot download BGT data without municipality boundaries")
        return False
    
    # Download BGT data
    print("BGT data not found, attempting to download...")
    try:
        bgt_gdf = download_bgt_for_alkmaar(str(alkmaar_geojson_path), str(DATA_DIR))
        if not bgt_gdf.empty:
            print("BGT data downloaded successfully")
            return True
        else:
            print("BGT download returned empty dataset")
            return False
    except Exception as e:
        print(f"BGT download failed: {e}")
        return False

# ================================
# BGT DATA LOADING
# ================================

def load_bgt_data():
    """Load pre-processed BGT data if available"""
    bgt_path = DATA_DIR / 'bgt_alkmaar_processed.geojson'
    try:
        if bgt_path.exists():
            print(f"Loading BGT data from: {bgt_path}")
            bgt_gdf = gpd.read_file(bgt_path)
            
            # Convert to WGS84 to match satellite/radar data
            if bgt_gdf.crs and bgt_gdf.crs != 'EPSG:4326':
                print(f"   Converting BGT from {bgt_gdf.crs} to EPSG:4326")
                bgt_gdf = bgt_gdf.to_crs('EPSG:4326')
            
            # Filter by date if needed
            if 'creationDate' in bgt_gdf.columns:
                bgt_gdf = bgt_gdf[
                    (bgt_gdf["creationDate"] < "2022-05-01") & 
                    (bgt_gdf["creationDate"] > "2019-03-01")
                ]
            print(f"   Loaded {len(bgt_gdf)} BGT features (2019-2022)")
            return bgt_gdf
        else:
            print(f"No BGT data found at {bgt_path}")
            return None
    except Exception as e:
        print(f"Error loading BGT data: {e}")
        return None

# Load water bodies and BGT data once at startup
WATER_BODIES = load_surface_water_mask()

# Ensure BGT data is available before loading
print("Checking BGT data availability...")
if ensure_bgt_data_available():
    BGT_DATA = load_bgt_data()
else:
    print("Continuing without BGT data - illegal construction detection will use radar only")
    BGT_DATA = None

# Get municipality bounds and geometry
ALKMAAR_BOUNDS, ALKMAAR_GEOMETRY = get_bounds_and_geometry_from_geojson()

# ================================
# AUTHENTICATION
# ================================

def get_sentinel_token():
    """Get access token from Sentinel Hub"""
    try:
        response = requests.post(
            TOKEN_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET,
            },
        )
        response.raise_for_status()
        return response.json()["access_token"]
    except Exception as e:
        print(f"Failed to get token: {e}")
        return None

# ================================
# CACHING
# ================================

def get_cache_key(layer_type, date, max_cloud=30, **kwargs):
    """Generate cache key from parameters"""
    cache_data = f"{layer_type}_{date}_{max_cloud}_{json.dumps(kwargs, sort_keys=True)}"
    return hashlib.md5(cache_data.encode()).hexdigest()

def save_to_cache(cache_key, data):
    """Save data to cache"""
    try:
        cache_file = CACHE_DIR / f"{cache_key}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"Cached: {cache_key[:8]}...")
    except Exception as e:
        print(f"Cache save failed: {e}")

def load_from_cache(cache_key):
    """Load data from cache"""
    try:
        cache_file = CACHE_DIR / f"{cache_key}.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            print(f"Cache hit: {cache_key[:8]}...")
            return data
    except Exception as e:
        print(f"Cache load failed: {e}")
    return None

# ================================
# SENTINEL HUB DATA FETCHING - OPTICAL
# ================================

def fetch_sentinel_data(product_type, date, max_cloud=30, apply_water_mask=False):
    """Fetch satellite data from Sentinel Hub with transparency support"""
    
    token = get_sentinel_token()
    if not token:
        return None
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    # Parse date
    if isinstance(date, str):
        date_obj = datetime.fromisoformat(date)
    else:
        date_obj = date
    
    # Create date range
    start_date = (date_obj - timedelta(days=2)).strftime("%Y-%m-%dT00:00:00Z")
    end_date = (date_obj + timedelta(days=3)).strftime("%Y-%m-%dT23:59:59Z")
    
    print(f"Fetching {product_type} for {date}")
    if apply_water_mask and product_type == 'NDCI' and WATER_BODIES is not None:
        print(f"   Will apply surface water mask to NDCI")
    
    # Evalscript with dataMask for transparency
    if product_type == 'RGB':
        evalscript = """
            //VERSION=3
            function setup() {
                return {
                    input: ["B04", "B03", "B02", "dataMask"],
                    output: { bands: 4 }
                };
            }
            function evaluatePixel(sample) {
                // Simple linear stretch with transparency
                let alpha = sample.dataMask;
                return [
                    sample.B04 * 2.5,
                    sample.B03 * 2.5,
                    sample.B02 * 2.5,
                    alpha
                ];
            }
        """
    elif product_type == 'NDCI':
        # Regular NDCI - water masking will be applied in post-processing
        evalscript = """
            //VERSION=3
            function setup() {
                return {
                    input: ["B05", "B04", "dataMask"],
                    output: { bands: 2 }
                };
            }
            function evaluatePixel(sample) {
                let ndci = (sample.B05 - sample.B04) / (sample.B05 + sample.B04 + 0.001);
                return [ndci, sample.dataMask];
            }
        """
    else:
        # Default to RGB
        evalscript = """
            //VERSION=3
            function setup() {
                return {
                    input: ["B04", "B03", "B02", "dataMask"],
                    output: { bands: 4 }
                };
            }
            function evaluatePixel(sample) {
                return [
                    sample.B04 * 2.5,
                    sample.B03 * 2.5,
                    sample.B02 * 2.5,
                    sample.dataMask
                ];
            }
        """
    
    # Request payload
    payload = {
        "input": {
            "bounds": {
                "geometry": ALKMAAR_GEOMETRY,
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}
            },
            "data": [{
                "type": "S2L2A",
                "dataFilter": {
                    "timeRange": {"from": start_date, "to": end_date},
                    "maxCloudCoverage": max_cloud
                },
                "processing": {"mosaickingOrder": "leastCC"}
            }]
        },
        "evalscript": evalscript,
        "output": {
            "width": 2048,
            "height": 2048,
            "responses": [{
                "identifier": "default",
                "format": {"type": "image/png"}
            }]
        }
    }
    
    try:
        response = requests.post(PROCESS_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        # Load image directly as PNG
        img = Image.open(io.BytesIO(response.content))
        
        # Convert to numpy array
        data = np.array(img, dtype=np.float32)
        
        # Normalize to 0-1 range
        if data.max() > 255:
            data = data / 65535.0  # 16-bit image
        elif data.max() > 1:
            data = data / 255.0  # 8-bit image
        
        # Apply surface water mask for NDCI if requested
        if apply_water_mask and product_type == 'NDCI' and WATER_BODIES is not None:
            bounds = get_bounds_from_geometry(ALKMAAR_GEOMETRY)
            water_mask = create_water_mask(data.shape[:2], bounds, WATER_BODIES)
            
            if data.ndim == 3 and data.shape[2] >= 2:
                # Apply water mask to NDCI values and alpha channel
                ndci_values = data[:, :, 0]
                alpha_values = data[:, :, 1]
                
                # Mask NDCI values (set to NaN where not water)
                masked_ndci = np.where(water_mask, ndci_values, np.nan)
                
                # Mask alpha values (set to 0 where not water)
                masked_alpha = np.where(water_mask, alpha_values, 0)
                
                # Update the data array
                data[:, :, 0] = masked_ndci
                data[:, :, 1] = masked_alpha
                
                print(f"Applied surface water mask to NDCI")
        
        print(f"Successfully fetched {product_type}")
        print(f"   Shape: {data.shape}, Range: [{data.min():.3f}, {data.max():.3f}]")
        
        return data
        
    except Exception as e:
        print(f"Error fetching {product_type}: {e}")
        return None

# ================================
# SENTINEL-1 RADAR DATA FETCHING (FROM PROJECT.PY)
# ================================

def fetch_sentinel1_product(product_type, start_date, orbit_direction="DESCENDING"):
    """
    Fetch Sentinel-1 radar products (from radar_copernicus_test.py).
    
    Args:
        product_type: str - 'VV', 'VH', 'RGB_VV_VH'
        start_date: str - ISO date string (YYYY-MM-DD)
        orbit_direction: str - 'ASCENDING' or 'DESCENDING'
    
    Returns:
        numpy array with the requested product
    """
    
    token = get_sentinel_token()
    if not token:
        return None
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    # Convert dates to proper format
    if isinstance(start_date, str):
        start_dt = datetime.fromisoformat(start_date)
    else:
        start_dt = start_date
    
    # Create a month-wide window for radar data
    end_dt = start_dt + timedelta(days=30)
    
    start_iso = start_dt.strftime("%Y-%m-%dT00:00:00Z")
    end_iso = end_dt.strftime("%Y-%m-%dT23:59:59Z")
    
    print(f"Fetching Sentinel-1 {product_type} for {start_date}")
    
    # Create evalscript based on product type - Fixed for UINT8
    if product_type == 'VV':
        evalscript = """
        //VERSION=3
        function setup() {
            return {
                input: ["VV"],
                output: { 
                    bands: 1,
                    sampleType: "UINT8"
                }
            };
        }
        function evaluatePixel(sample) {
            // Convert to dB, handle zero/negative values
            let vv_linear = Math.max(sample.VV, 0.0001);
            let vv_db = 10 * Math.log(vv_linear) / Math.LN10;
            
            // Normalize dB values to 0-1 range (typical range -25 to 0 dB)
            let normalized = Math.max(0, Math.min(1, (vv_db + 25) / 25));
            
            return [normalized * 255];
        }
        """
    elif product_type == 'VH':
        evalscript = """
        //VERSION=3
        function setup() {
            return {
                input: ["VH"],
                output: { 
                    bands: 1,
                    sampleType: "UINT8"
                }
            };
        }
        function evaluatePixel(sample) {
            // Convert to dB, handle zero/negative values
            let vh_linear = Math.max(sample.VH, 0.0001);
            let vh_db = 10 * Math.log(vh_linear) / Math.LN10;
            
            // Normalize dB values to 0-1 range (typical range -30 to -5 dB for VH)
            let normalized = Math.max(0, Math.min(1, (vh_db + 30) / 25));
            
            return [normalized * 255];
        }
        """
    elif product_type == 'RGB_VV_VH':
        evalscript = """
        //VERSION=3
        function setup() {
            return {
                input: ["VV", "VH"],
                output: { 
                    bands: 3,
                    sampleType: "UINT8"
                }
            };
        }
        function evaluatePixel(sample) {
            // Convert to dB and normalize for RGB display
            let vv_linear = Math.max(sample.VV, 0.0001);
            let vh_linear = Math.max(sample.VH, 0.0001);
            
            let vv_db = 10 * Math.log(vv_linear) / Math.LN10;
            let vh_db = 10 * Math.log(vh_linear) / Math.LN10;
            
            // Normalize to 0-1 range
            let vv_norm = Math.max(0, Math.min(1, (vv_db + 25) / 25));
            let vh_norm = Math.max(0, Math.min(1, (vh_db + 30) / 25));
            
            return [vv_norm * 255, vh_norm * 255, (vv_norm + vh_norm) * 127];
        }
        """
    
    # Payload for Sentinel Hub API
    payload = {
        "input": {
            "bounds": {
                "geometry": ALKMAAR_GEOMETRY,
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
            },
            "data": [
                {
                    "type": "S1GRD",
                    "dataFilter": {
                        "timeRange": {"from": start_iso, "to": end_iso},
                        "orbitDirection": orbit_direction
                    },
                    "processing": {
                        "mosaickingOrder": "mostRecent"
                    },
                }
            ],
        },
        "evalscript": evalscript,
        "output": {
            "width": 2048,
            "height": 2048,
            "responses": [{"identifier": "default", "format": {"type": "image/png"}}],
        },
    }
    
    try:
        # Fetch data
        response = requests.post(PROCESS_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code != 200:
            print(f"Error {response.status_code}: {response.text}")
            return None
        
        # Process PNG image
        img = Image.open(io.BytesIO(response.content))
        arr = np.array(img, dtype=np.float32)
        
        # Scale from 0-255 to 0-1 range
        arr = arr / 255.0
        
        # Convert back to dB for single band data (VV/VH)
        if product_type == 'VV':
            # Convert normalized values back to dB range (-25 to 0 dB)
            arr = (arr * 25) - 25
        elif product_type == 'VH':
            # Convert normalized values back to dB range (-30 to -5 dB)
            arr = (arr * 25) - 30
        # For RGB, keep 0-1 range
        
        print(f"Successfully fetched Sentinel-1 {product_type}")
        print(f"   Shape: {arr.shape}, Range: [{np.nanmin(arr):.3f}, {np.nanmax(arr):.3f}]")
        
        return arr
        
    except Exception as e:
        print(f"Error fetching Sentinel-1 data: {e}")
        return None

# ================================
# RADAR CHANGE DETECTION (FROM PROJECT.PY)
# ================================

def detect_radar_changes(date1='2019-06-01', date2='2022-06-01', threshold=0.05):
    """
    Detect changes using radar data (simplified from Project.py).
    
    Args:
        date1: str - Reference date
        date2: str - Comparison date  
        threshold: float - Change threshold
    
    Returns:
        dict with change detection results
    """
    print(f"Detecting radar changes between {date1} and {date2}")
    
    # Check cache first
    cache_key = get_cache_key('radar_change', f"{date1}_{date2}", threshold=threshold)
    cached_result = load_from_cache(cache_key)
    if cached_result is not None:
        return cached_result
    
    try:
        # Fetch radar data for both dates
        print("Fetching reference radar data...")
        ref_data = fetch_sentinel1_product(
            product_type='VV',
            start_date=date1,
            orbit_direction='DESCENDING'
        )
        
        print("Fetching comparison radar data...")
        comp_data = fetch_sentinel1_product(
            product_type='VV',
            start_date=date2,
            orbit_direction='DESCENDING'
        )
        
        if ref_data is None or comp_data is None:
            print("Failed to fetch radar data")
            return None
        
        # Calculate change map
        change_map = comp_data - ref_data
        
        # Apply threshold for positive changes (potential construction)
        positive_changes_mask = change_map > threshold
        
        # Calculate statistics
        total_positive = np.sum(positive_changes_mask)
        total_pixels = change_map.size
        positive_percentage = (total_positive / total_pixels) * 100
        
        print(f"Radar change detection complete")
        print(f"   Positive changes: {positive_percentage:.3f}% of area")
        
        result = {
            'reference_data': ref_data,
            'comparison_data': comp_data,
            'change_map': change_map,
            'positive_changes_mask': positive_changes_mask,
            'statistics': {
                'total_positive': int(total_positive),
                'total_pixels': int(total_pixels),
                'positive_percentage': float(positive_percentage)
            },
            'dates': {'reference': date1, 'comparison': date2}
        }
        
        # Cache the result
        save_to_cache(cache_key, result)
        
        return result
        
    except Exception as e:
        print(f"Error in radar change detection: {e}")
        return None

def detect_illegal_construction_radar(date1='2019-06-01', date2='2022-06-01'):
    """
    Detect potential illegal construction using radar + BGT (from Project.py).
    Now accepts custom dates as parameters.
    """
    print(f"Detecting potential illegal construction ({date1} to {date2})")
    
    # Get radar changes for the specified dates
    radar_results = detect_radar_changes(date1, date2, threshold=0.05)
    
    if radar_results is None:
        return None
    
    # Prepare result
    result = {
        'radar_changes': radar_results,
        'bgt_available': BGT_DATA is not None,
        'bgt_features': len(BGT_DATA) if BGT_DATA is not None else 0
    }
    
    return result

# ================================
# IMAGE PROCESSING
# ================================

def array_to_base64_image(data, product_type='RGB', opacity=0.8):
    """Convert numpy array to base64 image with transparent background"""
    if data is None:
        return None
    
    try:
        print(f"Converting {product_type} to image with opacity {opacity}")
        print(f"Input shape: {data.shape}, range: [{data.min():.3f}, {data.max():.3f}]")
        
        if product_type == 'RGB':
            # Handle RGB data
            if data.ndim == 3:
                if data.shape[2] == 4:
                    # RGBA - data already has alpha channel from dataMask
                    rgb_data = data[:, :, :3]
                    alpha_raw = data[:, :, 3]
                    
                    # Convert alpha to binary (0 or 255) with opacity
                    alpha = np.where(alpha_raw > 0.1, int(255 * opacity), 0).astype(np.uint8)
                    
                elif data.shape[2] == 3:
                    # RGB without alpha - create alpha based on black pixels
                    rgb_data = data.copy()
                    
                    # Create alpha mask: transparent where all RGB values are very low (black)
                    threshold = 0.05
                    is_black = np.all(rgb_data < threshold, axis=2)
                    is_nodata = np.all(rgb_data == 0, axis=2)
                    
                    # Combine conditions: transparent where black or no data
                    alpha = np.where(is_black | is_nodata, 0, int(255 * opacity)).astype(np.uint8)
                    
                else:
                    print(f"Unexpected channels: {data.shape[2]}")
                    return None
                
                # Ensure RGB data is in 0-1 range
                rgb_data = np.clip(rgb_data, 0, 1)
                
                # Apply simple brightness boost
                rgb_data = rgb_data * 1.2
                rgb_data = np.clip(rgb_data, 0, 1)
                
                # Convert to uint8
                rgb_uint8 = (rgb_data * 255).astype(np.uint8)
                
                # Stack RGBA
                rgba = np.dstack((rgb_uint8, alpha))
                
                # Create image with alpha channel
                image = Image.fromarray(rgba, mode='RGBA')
                
                print(f"Created RGBA image with transparency")
                
            else:
                print(f"Unexpected shape for RGB: {data.shape}")
                return None
                
        else:
            # Handle index products (NDCI)
            alpha = None
            water_mask_applied = False
            
            if data.ndim == 3:
                if data.shape[2] >= 2:
                    # Index with dataMask
                    index_data = data[:, :, 0]
                    alpha_raw = data[:, :, 1]
                    
                    # Check if this is water-masked data (has NaN values)
                    if np.any(np.isnan(index_data)):
                        water_mask_applied = True
                        # For water-masked data, alpha should be 0 where NaN
                        alpha = np.where(np.isnan(index_data), 0, alpha_raw * opacity)
                    else:
                        alpha = (alpha_raw > 0.1) * opacity  # Convert to float mask with opacity
                else:
                    # Just take first band
                    index_data = data[:, :, 0]
            elif data.ndim == 2:
                index_data = data
            else:
                print(f"Unexpected shape for {product_type}: {data.shape}")
                return None
            
            # Create mask for valid data if alpha not provided
            if alpha is None:
                alpha = ((index_data != 0) & ~np.isnan(index_data)) * opacity
            
            # Normalize to 0-1 for colormap
            if product_type == 'NDCI':
                # Indices are in range -1 to 1
                index_norm = (index_data + 1) / 2
            else:
                # Generic normalization
                valid_data = index_data[alpha > 0]
                if len(valid_data) > 0:
                    data_min = valid_data.min()
                    data_max = valid_data.max()
                    if data_max > data_min:
                        index_norm = (index_data - data_min) / (data_max - data_min)
                    else:
                        index_norm = np.zeros_like(index_data)
                else:
                    index_norm = np.zeros_like(index_data)
            
            index_norm = np.clip(index_norm, 0, 1)
            
            # Apply colormap
            if product_type == 'NDCI':
                cmap = plt.cm.RdYlBu_r
            else:
                cmap = plt.cm.viridis
            
            colored = cmap(index_norm)
            
            # Set alpha channel based on mask
            colored[:, :, 3] = alpha
            
            rgba = (colored * 255).astype(np.uint8)
            
            image = Image.fromarray(rgba, mode='RGBA')
            
            if water_mask_applied:
                print(f"Surface water mask applied to {product_type}")
        
        # Convert to base64
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        print(f"Image created successfully with transparent background")
        return img_base64
        
    except Exception as e:
        print(f"Error creating image: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_illegal_construction_visualization(radar_results, include_bgt=False):
    """Create visualization for illegal construction detection with optional BGT overlay"""
    
    if radar_results is None:
        return None
    
    try:
        # Extract data
        if 'radar_changes' in radar_results:
            # This is from illegal construction detection
            change_data = radar_results['radar_changes']
        else:
            # Direct radar change result
            change_data = radar_results
        
        positive_changes_mask = change_data['positive_changes_mask']
        
        # Create RGBA image
        height, width = positive_changes_mask.shape
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        
        # Set red for positive changes (potential construction)
        rgba[positive_changes_mask, 0] = 255  # Red channel
        rgba[positive_changes_mask, 3] = 200  # Alpha
        
        # If BGT overlay requested and available - PROPERLY RASTERIZE BGT
        if include_bgt and BGT_DATA is not None:
            try:
                # Get bounds and create transform
                bounds = get_bounds_from_geometry(ALKMAAR_GEOMETRY)
                transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], width, height)
                
                # Rasterize BGT polygons to same resolution as radar
                bgt_mask = rasterize(
                    [(geom, 1) for geom in BGT_DATA.geometry if geom is not None],
                    out_shape=(height, width),
                    transform=transform,
                    fill=0,
                    dtype='uint8'
                )
                
                # Add green overlay for BGT areas
                rgba[bgt_mask > 0, 1] = 255  # Green channel for legal construction
                rgba[bgt_mask > 0, 3] = 200  # Alpha
                
                print(f"Added BGT overlay: {np.sum(bgt_mask > 0)} pixels")
                
            except Exception as e:
                print(f"Could not add BGT overlay: {e}")
        
        # Create PIL image
        image = Image.fromarray(rgba, mode='RGBA')
        
        # Convert to base64
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return img_base64
        
    except Exception as e:
        print(f"Error creating illegal construction visualization: {e}")
        return None

# ================================
# API ENDPOINTS
# ================================

@app.route('/')
def serve_index():
    """Serve the main HTML page"""
    try:
        return send_from_directory(STATIC_DIR, 'index.html')
    except FileNotFoundError:
        return "index.html not found in web directory", 404

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    try:
        return send_from_directory(STATIC_DIR, filename)
    except FileNotFoundError:
        return f"File {filename} not found", 404

@app.route('/api/satellite-data', methods=['POST'])
def get_satellite_data():
    """Get satellite data for specified parameters"""
    try:
        data = request.get_json()
        product_type = data.get('product_type', 'RGB')
        date = data.get('date')
        opacity = data.get('opacity', 0.8)
        
        # Validate required parameters
        if not date:
            return jsonify({
                'success': False,
                'error': 'Date parameter is required'
            }), 400
        
        # For NDCI, always apply water mask by default
        if product_type == 'NDCI':
            apply_water_mask = True
            print(f"   NDCI: Water mask applied by default")
        else:
            apply_water_mask = False
        
        # No max cloud constraint - use 100% to get any available data
        max_cloud = 100
        
        print(f"\nRequest: {product_type} for {date} (no cloud limit, opacity: {opacity})")
        if apply_water_mask and product_type == 'NDCI':
            print(f"   Surface water mask will be applied")
        
        # Check cache first
        cache_key = get_cache_key('satellite', date, max_cloud, 
                                 product_type=product_type, 
                                 water_mask=apply_water_mask)
        cached_data = load_from_cache(cache_key)
        
        if cached_data is not None:
            satellite_data = cached_data
        else:
            # Fetch from Sentinel Hub
            satellite_data = fetch_sentinel_data(product_type, date, max_cloud, apply_water_mask)
            
            if satellite_data is not None:
                save_to_cache(cache_key, satellite_data)
        
        if satellite_data is None:
            return jsonify({
                'success': False,
                'error': f'No data available for {product_type} on {date}'
            }), 404
        
        # Convert to base64 image with specified opacity
        image_b64 = array_to_base64_image(satellite_data, product_type=product_type, opacity=opacity)
        
        if image_b64 is None:
            return jsonify({
                'success': False,
                'error': 'Failed to generate image'
            }), 500
        
        return jsonify({
            'success': True,
            'imageUrl': f'data:image/png;base64,{image_b64}',
            'bounds': ALKMAAR_BOUNDS,
            'date': date,
            'product_type': product_type,
            'water_mask_applied': apply_water_mask and product_type == 'NDCI' and WATER_BODIES is not None
        })
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/illegal-construction', methods=['POST'])
def get_illegal_construction():
    """Detect potential illegal construction using radar + BGT"""
    try:
        data = request.get_json()
        # Get dates from request - now customizable
        date1 = data.get('date1', '2019-06-01')
        date2 = data.get('date2', '2022-06-01')
        
        print(f"\nIllegal construction detection ({date1} to {date2})")
        
        # Detect illegal construction with specified dates
        results = detect_illegal_construction_radar(date1, date2)
        
        if results is None:
            return jsonify({
                'success': False,
                'error': 'Failed to detect illegal construction'
            }), 500
        
        # Create visualization with BGT overlay indication
        image_b64 = create_illegal_construction_visualization(results, include_bgt=True)
        
        if image_b64 is None:
            return jsonify({
                'success': False,
                'error': 'Failed to create visualization'
            }), 500
        
        return jsonify({
            'success': True,
            'imageUrl': f'data:image/png;base64,{image_b64}',
            'bounds': ALKMAAR_BOUNDS,
            'bgt_features': results['bgt_features'],
            'statistics': results['radar_changes']['statistics'],
            'dates': {'from': date1, 'to': date2}
        })
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/download-bgt', methods=['POST'])
def download_bgt_data():
    """Download BGT data for Alkmaar"""
    try:
        print("Manual BGT download requested...")
        
        alkmaar_geojson_path = DATA_DIR / 'alkmaar.geojson'
        
        if not alkmaar_geojson_path.exists():
            return jsonify({
                'success': False,
                'error': 'Municipality boundary file (alkmaar.geojson) not found'
            }), 400
        
        # Download BGT data
        bgt_gdf = download_bgt_for_alkmaar(str(alkmaar_geojson_path), str(DATA_DIR))
        
        if not bgt_gdf.empty:
            # Reload BGT data globally
            global BGT_DATA
            BGT_DATA = load_bgt_data()
            
            return jsonify({
                'success': True,
                'message': f'BGT data downloaded successfully. {len(bgt_gdf)} features processed.',
                'bgt_features': len(bgt_gdf)
            })
        else:
            return jsonify({
                'success': False,
                'error': 'BGT download returned empty dataset'
            }), 500
            
    except Exception as e:
        print(f"Error downloading BGT data: {e}")
        return jsonify({
            'success': False, 
            'error': f'BGT download failed: {str(e)}'
        }), 500
def clear_cache():
    """Clear cache for specific product or all"""
    try:
        data = request.get_json() or {}
        product_type = data.get('product_type', 'all')
        
        if product_type == 'all':
            cache_files = list(CACHE_DIR.glob('*.pkl'))
            for f in cache_files:
                f.unlink()
            message = f"Cleared {len(cache_files)} cache files"
        else:
            count = 0
            for f in CACHE_DIR.glob('*.pkl'):
                if product_type.lower() in f.name.lower():
                    f.unlink()
                    count += 1
            message = f"Cleared {count} cache files for {product_type}"
        
        print(f"{message}")
        return jsonify({'success': True, 'message': message})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/info', methods=['GET'])
def get_info():
    """Get system information"""
    water_info = "available" if WATER_BODIES is not None else "not available"
    water_count = len(WATER_BODIES) if WATER_BODIES is not None else 0
    
    bgt_info = "available" if BGT_DATA is not None else "not available"
    bgt_count = len(BGT_DATA) if BGT_DATA is not None else 0
    
    return jsonify({
        'bounds': ALKMAAR_BOUNDS,
        'bounds_source': 'alkmaar.geojson' if (DATA_DIR / 'alkmaar.geojson').exists() else 'default',
        'geometry_type': ALKMAAR_GEOMETRY.get('type', 'unknown'),
        'cache_files': len(list(CACHE_DIR.glob('*.pkl'))) if CACHE_DIR.exists() else 0,
        'area_km2': round((ALKMAAR_BOUNDS[1][1] - ALKMAAR_BOUNDS[0][1]) * 
                          (ALKMAAR_BOUNDS[1][0] - ALKMAAR_BOUNDS[0][0]) * 111 * 111, 1),
        'surface_water': water_info,
        'water_bodies_count': water_count,
        'bgt_data': bgt_info,
        'bgt_features': bgt_count
    })

# ================================
# MAIN
# ================================

def main(port=5000, debug=True):
    """Run the web interface"""
    print("Alkmaar Municipality Satellite & Radar Analysis")
    print("=" * 60)
    
    print(f"\nMunicipality Coverage:")
    print(f"   Bounds: {ALKMAAR_BOUNDS}")
    area_km2 = (ALKMAAR_BOUNDS[1][1] - ALKMAAR_BOUNDS[0][1]) * \
               (ALKMAAR_BOUNDS[1][0] - ALKMAAR_BOUNDS[0][0]) * 111 * 111
    print(f"   Area: ~{area_km2:.1f} km²")
    
    print(f"\nSurface Water Data:")
    if WATER_BODIES is not None:
        print(f"   Loaded {len(WATER_BODIES)} water body features")
        print(f"   Water mask available for NDCI")
    else:
        print(f"   No surface water data found")
        print(f"   Water mask will not be available")
    
    print(f"\nBGT Data:")
    if BGT_DATA is not None:
        print(f"   Loaded {len(BGT_DATA)} BGT features (2019-2022)")
        print(f"   Legal construction data available")
        print(f"   Green overlay will show on illegal construction detection")
    else:
        print(f"   No BGT data found")
        print(f"   Illegal construction detection will use radar only")
    
    # Test connection
    print(f"\nTesting Sentinel Hub connection...")
    token = get_sentinel_token()
    if token:
        print(f"   Successfully authenticated")
    else:
        print(f"   Authentication failed")
    
    print(f"\nAPI Endpoints:")
    print(f"   POST /api/satellite-data      - Optical imagery (RGB, NDCI)")
    print(f"   POST /api/illegal-construction - Illegal construction analysis")
    print(f"   POST /api/download-bgt        - Download BGT data")
    print(f"   POST /api/clear-cache        - Clear cache")
    print(f"   GET  /api/info              - System information")
    
    print(f"\nStarting server on http://localhost:{port}")
    print(f"\nTips:")
    print(f"   • Cache is cleared on startup for testing")
    print(f"   • All date ranges are now customizable")
    print(f"   • BGT data shows as green overlay")
    print(f"   • Red areas = radar-detected changes")
    print(f"   • Green areas = BGT documented (legal)")
    print(f"   • Red without green = potential illegal")
    
    print(f"\nPress Ctrl+C to stop the server")
    
    try:
        app.run(host='0.0.0.0', port=port, debug=debug)
    except KeyboardInterrupt:
        print(f"\nServer stopped")

if __name__ == "__main__":
    main()