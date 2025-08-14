# web_backend.py - Updated backend with surface water GeoJSON masking
import json
import os
import hashlib
import pickle
import base64
import io
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

print(f"📂 Static files: {STATIC_DIR}")
print(f"💾 Cache directory: {CACHE_DIR}")
print(f"📊 Data directory: {DATA_DIR}")

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
            print(f"🗑️ Testing mode: Cleared {len(cache_files)} cache files on startup")
        else:
            print("🗑️ Testing mode: No cache files to clear")
    except Exception as e:
        print(f"⚠️ Failed to clear cache on startup: {e}")

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
        print(f"📂 Reading GeoJSON from: {geojson_path}")
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
        
        print(f"🗺️ Municipality bounds: {bounds}")
        print(f"   Area: {max_lat-min_lat:.4f}° × {max_lon-min_lon:.4f}°")
        
        return bounds, geometry
        
    except Exception as e:
        print(f"⚠️ Error reading GeoJSON: {e}")
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
        print(f"🌊 Loading surface water from: {surface_water_path}")
        water_gdf = gpd.read_file(surface_water_path)
        # Ensure CRS is WGS84 (EPSG:4326) to match Sentinel Hub data
        if water_gdf.crs != 'EPSG:4326':
            water_gdf = water_gdf.to_crs('EPSG:4326')
        print(f"   Loaded {len(water_gdf)} water body features")
        return water_gdf
    except FileNotFoundError:
        print(f"⚠️ Surface water file not found at {surface_water_path}")
        print("   Water mask will not be available for NDCI")
        return None
    except Exception as e:
        print(f"⚠️ Error loading surface water data: {e}")
        return None

def get_bounds_from_geometry(geometry):
    """Extract bounding box from GeoJSON geometry."""
    if geometry['type'] == 'Polygon':
        coords = geometry['coordinates'][0]
    elif geometry['type'] == 'MultiPolygon':
        # Get all coordinates and flatten
        coords = []
        for poly in geometry['coordinates']:
            coords.extend(poly[0])
    else:
        raise ValueError(f"Unsupported geometry type: {geometry['type']}")
    
    lons = [coord[0] for coord in coords]
    lats = [coord[1] for coord in coords]
    
    return (min(lons), min(lats), max(lons), max(lats))

def create_water_mask(image_shape, bounds, water_gdf):
    """
    Create a binary mask for water bodies using surface water GeoJSON.
    
    Args:
        image_shape: tuple (height, width) of the target image
        bounds: tuple (minx, miny, maxx, maxy) in EPSG:4326
        water_gdf: GeoDataFrame containing water body polygons
    
    Returns:
        Binary mask array where 1 = water, 0 = land
    """
    if water_gdf is None or water_gdf.empty:
        print("⚠️ No water body data available - returning full mask")
        return np.ones(image_shape, dtype=bool)  # Return all True (no masking)
    
    # Create transform from bounds to image coordinates
    height, width = image_shape
    minx, miny, maxx, maxy = bounds
    transform = from_bounds(minx, miny, maxx, maxy, width, height)
    
    print(f"🌊 Creating water mask from {len(water_gdf)} water bodies")
    
    # Rasterize water body polygons
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

# Load water bodies once at startup
WATER_BODIES = load_surface_water_mask()

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
        print(f"❌ Failed to get token: {e}")
        return None

# ================================
# CACHING
# ================================

def get_cache_key(layer_type, date, max_cloud, **kwargs):
    """Generate cache key from parameters"""
    cache_data = f"{layer_type}_{date}_{max_cloud}_{json.dumps(kwargs, sort_keys=True)}"
    return hashlib.md5(cache_data.encode()).hexdigest()

def save_to_cache(cache_key, data):
    """Save data to cache"""
    try:
        cache_file = CACHE_DIR / f"{cache_key}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"💾 Cached: {cache_key[:8]}...")
    except Exception as e:
        print(f"⚠️ Cache save failed: {e}")

def load_from_cache(cache_key):
    """Load data from cache"""
    try:
        cache_file = CACHE_DIR / f"{cache_key}.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            print(f"🎯 Cache hit: {cache_key[:8]}...")
            return data
    except Exception as e:
        print(f"⚠️ Cache load failed: {e}")
    return None

# ================================
# AVAILABLE DATES - ENHANCED WITH CLOUD COVERAGE DATA
# ================================

def get_available_dates_with_cloud_info(start_date, end_date, max_cloud=10):
    """
    Query Sentinel Hub catalog for available dates with cloud coverage information.
    Returns list of dictionaries with date and cloud coverage.
    """
    token = get_sentinel_token()
    if not token:
        print("⚠️ No token available, using fallback dates")
        return get_available_dates_fallback_with_cloud(start_date, end_date, max_cloud)
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    catalog_url = "https://sh.dataspace.copernicus.eu/api/v1/catalog/search"
    
    # Convert dates
    if isinstance(start_date, str):
        start_dt = datetime.fromisoformat(start_date)
    else:
        start_dt = start_date
        
    if isinstance(end_date, str):
        end_dt = datetime.fromisoformat(end_date)
    else:
        end_dt = end_date
    
    start_iso = start_dt.strftime("%Y-%m-%dT00:00:00Z")
    end_iso = end_dt.strftime("%Y-%m-%dT23:59:59Z")
    
    # Get bounds for search
    bbox = [ALKMAAR_BOUNDS[0][1], ALKMAAR_BOUNDS[0][0], 
            ALKMAAR_BOUNDS[1][1], ALKMAAR_BOUNDS[1][0]]
    
    search_payload = {
        "collections": ["sentinel-2-l2a"],
        "datetime": f"{start_iso}/{end_iso}",
        "bbox": bbox,
        "limit": 200,  # Increased limit to get more data
        "query": {
            "eo:cloud_cover": {"lt": max_cloud}
        }
    }
    
    try:
        response = requests.post(catalog_url, headers=headers, json=search_payload, timeout=15)
        response.raise_for_status()
        catalog_data = response.json()
        
        # Group by date and get the best (lowest cloud) acquisition per day
        date_cloud_map = {}
        for feature in catalog_data.get('features', []):
            date_str = feature['properties']['datetime']
            cloud_cover = feature['properties'].get('eo:cloud_cover', 100)
            date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            date_key = date_obj.strftime('%Y-%m-%d')
            
            # Keep the acquisition with lowest cloud coverage for each date
            if date_key not in date_cloud_map or cloud_cover < date_cloud_map[date_key]['cloud_cover']:
                date_cloud_map[date_key] = {
                    'date': date_key,
                    'cloud_cover': round(cloud_cover, 1),
                    'datetime': date_obj
                }
        
        # Convert to sorted list
        available_dates = sorted(date_cloud_map.values(), key=lambda x: x['datetime'])
        
        print(f"✅ Found {len(available_dates)} dates with <{max_cloud}% cloud coverage")
        return available_dates
        
    except (requests.exceptions.RequestException, requests.exceptions.HTTPError) as e:
        print(f"⚠️ Catalog API error ({e}), using fallback method")
        return get_available_dates_fallback_with_cloud(start_date, end_date, max_cloud)
    except Exception as e:
        print(f"❌ Unexpected error querying catalog: {e}")
        return get_available_dates_fallback_with_cloud(start_date, end_date, max_cloud)

def get_available_dates_fallback_with_cloud(start_date, end_date, max_cloud=10):
    """
    Fallback method to generate available dates with simulated cloud coverage.
    """
    print("🔄 Using fallback date generation with simulated cloud data")
    
    # Parse dates
    if isinstance(start_date, str):
        start_dt = datetime.fromisoformat(start_date)
    else:
        start_dt = start_date
        
    if isinstance(end_date, str):
        end_dt = datetime.fromisoformat(end_date)
    else:
        end_dt = end_date
    
    # Generate dates based on Sentinel-2 revisit cycle (approximately every 5 days)
    available_dates = []
    current_date = start_dt
    
    # Start with a known Sentinel-2 acquisition date and work from there
    sentinel_start = datetime(2015, 6, 23)
    days_diff = (start_dt - sentinel_start).days
    
    # Find the first acquisition date after start_date
    cycle_day = days_diff % 5  # 5-day cycle
    first_date = start_dt + timedelta(days=(5 - cycle_day) % 5)
    
    import random
    random.seed(42)  # For consistent results
    
    current_date = first_date
    while current_date <= end_dt:
        # Only simulate low cloud coverage dates for fallback
        cloud_cover = random.uniform(2, 8)  # Always low clouds for fallback
        
        available_dates.append({
            'date': current_date.strftime('%Y-%m-%d'),
            'cloud_cover': round(cloud_cover, 1),
            'datetime': current_date
        })
        
        current_date += timedelta(days=5)  # Every 5 days approximately
    
    print(f"✅ Generated {len(available_dates)} fallback dates with low cloud data")
    return available_dates

def get_latest_date_for_cloud_coverage(start_date, end_date, max_cloud=10):
    """
    Get the latest available date within the specified cloud coverage threshold.
    """
    dates_with_cloud = get_available_dates_with_cloud_info(start_date, end_date, max_cloud)
    
    if not dates_with_cloud:
        return None
    
    # Filter by cloud coverage and get the latest date
    suitable_dates = [d for d in dates_with_cloud if d['cloud_cover'] <= max_cloud]
    
    if not suitable_dates:
        return None
    
    # Return the latest date
    latest = max(suitable_dates, key=lambda x: x['datetime'])
    return latest

# ================================
# SENTINEL HUB DATA FETCHING - WITH SURFACE WATER MASK
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
    
    print(f"🛰️ Fetching {product_type} for {date}")
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
    elif product_type == 'NDVI':
        evalscript = """
            //VERSION=3
            function setup() {
                return {
                    input: ["B08", "B04", "dataMask"],
                    output: { bands: 2 }
                };
            }
            function evaluatePixel(sample) {
                let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04 + 0.001);
                return [ndvi, sample.dataMask];
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
    elif product_type == 'NDWI':
        evalscript = """
            //VERSION=3
            function setup() {
                return {
                    input: ["B03", "B08", "dataMask"],
                    output: { bands: 2 }
                };
            }
            function evaluatePixel(sample) {
                let ndwi = (sample.B03 - sample.B08) / (sample.B03 + sample.B08 + 0.001);
                return [ndwi, sample.dataMask];
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
                
                print(f"✅ Applied surface water mask to NDCI")
        
        print(f"✅ Successfully fetched {product_type}")
        print(f"   Shape: {data.shape}, Range: [{data.min():.3f}, {data.max():.3f}]")
        
        return data
        
    except Exception as e:
        print(f"❌ Error fetching {product_type}: {e}")
        return None

# ================================
# IMAGE PROCESSING - SIMPLIFIED
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
            # Handle index products (NDVI, NDCI, NDWI)
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
            if product_type in ['NDVI', 'NDCI', 'NDWI']:
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
            if product_type == 'NDVI':
                cmap = plt.cm.RdYlGn
            elif product_type == 'NDCI':
                cmap = plt.cm.RdYlBu_r
            elif product_type == 'NDWI':
                cmap = plt.cm.BrBG
            else:
                cmap = plt.cm.viridis
            
            colored = cmap(index_norm)
            
            # Set alpha channel based on mask
            colored[:, :, 3] = alpha
            
            rgba = (colored * 255).astype(np.uint8)
            
            image = Image.fromarray(rgba, mode='RGBA')
            
            if water_mask_applied:
                print(f"✅ Surface water mask applied to {product_type}")
        
        # Convert to base64
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        print(f"✅ Image created successfully with transparent background")
        return img_base64
        
    except Exception as e:
        print(f"❌ Error creating image: {e}")
        import traceback
        traceback.print_exc()
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
        date = data.get('date')  # No default date - should come from frontend
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
        
        print(f"\n📡 Request: {product_type} for {date} (no cloud limit, opacity: {opacity})")
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
        print(f"❌ Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/clear-cache', methods=['POST'])
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
        
        print(f"🗑️ {message}")
        return jsonify({'success': True, 'message': message})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/info', methods=['GET'])
def get_info():
    """Get system information"""
    water_info = "available" if WATER_BODIES is not None else "not available"
    water_count = len(WATER_BODIES) if WATER_BODIES is not None else 0
    
    return jsonify({
        'bounds': ALKMAAR_BOUNDS,
        'bounds_source': 'alkmaar.geojson' if (DATA_DIR / 'alkmaar.geojson').exists() else 'default',
        'geometry_type': ALKMAAR_GEOMETRY.get('type', 'unknown'),
        'cache_files': len(list(CACHE_DIR.glob('*.pkl'))) if CACHE_DIR.exists() else 0,
        'area_km2': round((ALKMAAR_BOUNDS[1][1] - ALKMAAR_BOUNDS[0][1]) * 
                          (ALKMAAR_BOUNDS[1][0] - ALKMAAR_BOUNDS[0][0]) * 111 * 111, 1),
        'surface_water': water_info,
        'water_bodies_count': water_count
    })

# ================================
# MAIN
# ================================

def main(port=5000, debug=True):
    """Run the web interface"""
    print("🚀 Alkmaar Municipality Satellite Analysis")
    print("=" * 60)
    
    print(f"\n🗺️ Municipality Coverage:")
    print(f"   Bounds: {ALKMAAR_BOUNDS}")
    area_km2 = (ALKMAAR_BOUNDS[1][1] - ALKMAAR_BOUNDS[0][1]) * \
               (ALKMAAR_BOUNDS[1][0] - ALKMAAR_BOUNDS[0][0]) * 111 * 111
    print(f"   Area: ~{area_km2:.1f} km²")
    
    print(f"\n🌊 Surface Water Data:")
    if WATER_BODIES is not None:
        print(f"   ✅ Loaded {len(WATER_BODIES)} water body features")
        print(f"   ✅ Water mask available for NDCI")
    else:
        print(f"   ⚠️ No surface water data found")
        print(f"   ⚠️ Water mask will not be available")
    
    # Test connection
    print(f"\n📡 Testing Sentinel Hub connection...")
    token = get_sentinel_token()
    if token:
        print(f"   ✅ Successfully authenticated")
    else:
        print(f"   ❌ Authentication failed")
    
    print(f"\n📡 API Endpoints:")
    print(f"   POST /api/satellite-data")
    print(f"   POST /api/clear-cache")
    print(f"   GET  /api/info")
    
    print(f"\n🌐 Starting server on http://localhost:{port}")
    print(f"\n💡 Tips:")
    print(f"   • Cache is cleared on startup for testing")
    print(f"   • Available dates use fallback if catalog API is down")
    print(f"   • Surface water mask applied to NDCI when requested")
    print(f"   • Try dates with <30% cloud for best results")
    
    print(f"\nPress Ctrl+C to stop the server")
    
    try:
        app.run(host='0.0.0.0', port=port, debug=debug)
    except KeyboardInterrupt:
        print(f"\n🛑 Server stopped")

if __name__ == "__main__":
    main()