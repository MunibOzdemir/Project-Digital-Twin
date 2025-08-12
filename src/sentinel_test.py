# %%
import requests  # Send HTTP requests to Sentinel Hub API
import json  # Read GeoJSON file as a Python dict
import numpy as np  # Numerical calculations and array manipulation
from io import BytesIO  # Use BytesIO to load binary image data into memory
from PIL import Image  # Open image and convert to NumPy array
import matplotlib.pyplot as plt  # Create visualizations
from datetime import datetime, timedelta  # Manipulate dates
import geopandas as gpd  # For handling GeoJSON data
from rasterio.features import rasterize  # For converting vector to raster
from rasterio.transform import from_bounds  # For creating geospatial transforms
import shapely.geometry  # For geometry operations

# --- Configuration & credentials ---
CLIENT_ID = "sh-93c6fbd0-8c4a-4e40-8c59-d06889413797"
CLIENT_SECRET = "LKVq6MTE0S3kohQjRI1Yuj03aU5frOTm"
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"

# Path to your GeoJSON files
GEOJSON_PATH = r"C:\Users\munib\Desktop\Aanbesteding\Project\Project-Digital-Twin\data\alkmaar.geojson"
SURFACE_WATER_PATH = r"C:\Users\munib\Desktop\Aanbesteding\Project\Project-Digital-Twin\data\surface_water.geojson"  # Add your surface water GeoJSON path

# Maximum cloud coverage in percent (feel free to raise if imagery is limited)
MAX_CLOUD = 1

# --- Read GeoJSON and extract correct geometry ---
with open(GEOJSON_PATH) as f:
    gj = json.load(f)

# If 'features' key exists, take the first feature; otherwise use direct geometry or whole object
geom = gj["features"][0]["geometry"] if "features" in gj else gj.get("geometry", gj)

# --- Load surface water bodies ---
def load_surface_water_mask():
    """
    Load surface water GeoJSON and prepare it for masking.
    Returns GeoDataFrame with water body geometries.
    """
    try:
        water_gdf = gpd.read_file(SURFACE_WATER_PATH)
        # Ensure CRS is WGS84 (EPSG:4326) to match Sentinel Hub data
        if water_gdf.crs != 'EPSG:4326':
            water_gdf = water_gdf.to_crs('EPSG:4326')
        return water_gdf
    except FileNotFoundError:
        print(f"Surface water file not found at {SURFACE_WATER_PATH}")
        return None
    except Exception as e:
        print(f"Error loading surface water data: {e}")
        return None

# Load water bodies once
water_bodies = load_surface_water_mask()

# --- Function to get access token from Sentinel Hub ---
def get_token():
    """
    Authenticate using client_credentials grant.
    Returns a bearer token for further API calls.
    """
    r = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
        },
    )
    r.raise_for_status()  # Raise error if response is not OK
    return r.json()["access_token"]


# Fetch token once and build headers with Authorization
token = get_token()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

# %%
# --- Function to create water mask ---
def create_water_mask(image_shape, bounds, water_gdf):
    """
    Create a binary mask for water bodies.
    
    Args:
        image_shape: tuple (height, width) of the target image
        bounds: tuple (minx, miny, maxx, maxy) in EPSG:4326
        water_gdf: GeoDataFrame containing water body polygons
    
    Returns:
        Binary mask array where 1 = water, 0 = land
    """
    if water_gdf is None or water_gdf.empty:
        print("No water body data available, skipping masking")
        return np.ones(image_shape, dtype=bool)  # Return all True (no masking)
    
    # Create transform from bounds to image coordinates
    height, width = image_shape
    minx, miny, maxx, maxy = bounds
    transform = from_bounds(minx, miny, maxx, maxy, width, height)
    
    # Rasterize water body polygons
    water_mask = rasterize(
        [(geom, 1) for geom in water_gdf.geometry],
        out_shape=image_shape,
        transform=transform,
        fill=0,
        dtype='uint8'
    )
    
    return water_mask.astype(bool)

# --- Function to get image bounds from geometry ---
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

# %%
# --- Generic function to fetch Sentinel-2 data for any bands ---
def fetch_s2(
    bands,
    start_iso,
    end_iso,
    width=2048,
    height=2048,
    max_cloud=MAX_CLOUD,
    mosaicking_order="leastCC",
    apply_water_mask=True,
):
    """
    Fetch pixel reflectance values for specified bands over the given time period.
    bands         : list of band names (e.g. ["B04","B03","B02"])
    start_iso     : ISO date string for start of period
    end_iso       : ISO date string for end of period
    width, height : resolution of the image in pixels
    max_cloud     : maximum allowed cloud coverage (%)
    mosaicking_order: 'leastCC' or 'mostRecent'
    apply_water_mask: whether to mask to water bodies only

    Returns a NumPy array of shape=(h, w, len(bands)), dtype=float32, values [0,1].
    """
    # Dynamically build the evalscript to retrieve only the requested bands
    evalscript = f"""
    //VERSION=3
    function setup() {{
      return {{
        input: {bands},
        output: {{ bands: {len(bands)} }}
      }};
    }}
    function evaluatePixel(sample) {{
      return [{', '.join(f"sample.{b}" for b in bands)}];
    }}
    """

    # Payload with geometry bounds, date filter, and processing options
    payload = {
        "input": {
            "bounds": {
                "geometry": geom,
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
            },
            "data": [
                {
                    "type": "S2L2A",  # Sentinel-2 Level-2A products
                    "dataFilter": {
                        "timeRange": {"from": start_iso, "to": end_iso},
                        "maxCloudCoverage": max_cloud,
                    },
                    "processing": {"mosaickingOrder": mosaicking_order},
                }
            ],
        },
        "evalscript": evalscript,
        "output": {
            "width": width,
            "height": height,
            "responses": [{"identifier": "default", "format": {"type": "image/png"}}],
        },
    }

    # Send request, get PNG image back as bytes
    r = requests.post(PROCESS_URL, headers=headers, json=payload)
    r.raise_for_status()

    # Open the PNG in memory and convert to NumPy array
    img = Image.open(BytesIO(r.content))
    arr = np.array(img, dtype=np.uint8)

    # Some PNGs include an alpha channel, we strip it here
    if arr.ndim == 3 and arr.shape[2] > len(bands):
        arr = arr[..., : len(bands)]

    # Scale 0–255 values to 0.0–1.0 floats for further processing
    arr = arr.astype(np.float32) / 255.0
    
    # Apply water mask if requested and available
    if apply_water_mask and water_bodies is not None:
        bounds = get_bounds_from_geometry(geom)
        water_mask = create_water_mask(arr.shape[:2], bounds, water_bodies)
        
        # Apply mask - set non-water pixels to NaN
        if arr.ndim == 3:
            # Multi-band image
            water_mask_3d = np.repeat(water_mask[:, :, np.newaxis], arr.shape[2], axis=2)
            arr = np.where(water_mask_3d, arr, np.nan)
        else:
            # Single band image
            arr = np.where(water_mask, arr, np.nan)
    
    return arr

# %%
# --- Helper functions for specific products ---
def fetch_true_color(start_iso, end_iso, apply_water_mask=False):
    """
    Fetch true-color (RGB) image: B04 (red), B03 (green), B02 (blue).
    Note: Usually we don't want to mask RGB images, so default is False.
    """
    return fetch_s2(["B04", "B03", "B02"], start_iso, end_iso, apply_water_mask=apply_water_mask)


def fetch_ndci(start_iso, end_iso, apply_water_mask=True):
    """
    Compute the Normalized Difference Chlorophyll Index:
      NDCI = (B05 - B04) / (B05 + B04)
    B05 = red-edge band (channel 0), B04 = red (channel 1).
    By default, applies water masking since NDCI is most relevant for water bodies.
    """
    arr = fetch_s2(["B05", "B04"], start_iso, end_iso, apply_water_mask=apply_water_mask)
    b5, b4 = arr[..., 0], arr[..., 1]
    # Add small value to denominator to prevent division by zero
    ndci = (b5 - b4) / (b5 + b4 + 1e-6)
    
    # If water masking was applied, NaN values are already set
    # If not, we might still want to mask out clearly non-water areas based on NDCI values
    if not apply_water_mask:
        # Optional: mask out extreme values that are likely not water
        ndci = np.where((ndci < -0.8) | (ndci > 0.8), np.nan, ndci)
    
    return ndci


# --- 1) True-color visualization for exactly two periods ---
dates_tc = {
    "May 2024": ("2024-05-01T00:00:00Z", "2024-05-31T23:59:59Z"),
    "April 2025": ("2025-04-01T00:00:00Z", "2025-04-30T23:59:59Z"),
}

# Fetch both RGB images (without water masking for context)
print("Fetching true-color images...")
tc_maps = {
    label: fetch_true_color(start, end, apply_water_mask=False) 
    for label, (start, end) in dates_tc.items()
}

# Plot in a single row with 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
for ax, (label, img) in zip(axes, tc_maps.items()):
    # Handle NaN values in display
    display_img = np.where(np.isnan(img), 0, img)
    ax.imshow(display_img)
    ax.set_title(f"True-color {label}")
    ax.axis("off")

plt.tight_layout()
plt.show()

# --- 2) Compute and visualize NDCI with water masking ---
print("Fetching NDCI data with water body masking...")
ndci_maps = {
    label: fetch_ndci(start, end, apply_water_mask=True) 
    for label, (start, end) in dates_tc.items()
}

# Compute difference between April 2025 and May 2024
delta_ndci = ndci_maps["April 2025"] - ndci_maps["May 2024"]

# Create 3 subplots: NDCI May 2024, NDCI April 2025, and ΔNDCI
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Define plotting settings for each axis
settings = [
    (ndci_maps["May 2024"], "NDCI May 2024 (Water Bodies Only)", "Spectral", (-1, 1)),
    (ndci_maps["April 2025"], "NDCI April 2025 (Water Bodies Only)", "Spectral", (-1, 1)),
    (delta_ndci, "Δ NDCI (Water Bodies Only)", "RdBu", (-1, 1)),
]

# Plot all three and store the imshow handles for colorbar
im_handles = []
for ax, (data, title, cmap_name, vlim) in zip(axes, settings):
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color='white')  # or color=(1, 1, 1, 0) for transparent
    im = ax.imshow(data, cmap=cmap, vmin=vlim[0], vmax=vlim[1])
    ax.set_title(title)
    ax.axis("off")
    im_handles.append(im)


# Add colorbars
for i, (ax, im) in enumerate(zip(axes, im_handles)):
    fig.colorbar(im, ax=ax, shrink=0.7)

plt.tight_layout()
plt.show()

# --- 3) Optional: Show water mask overlay on true-color image ---
if water_bodies is not None:
    print("Creating water mask visualization...")
    # Get bounds and create mask for visualization
    bounds = get_bounds_from_geometry(geom)
    sample_img = tc_maps["May 2024"]
    water_mask = create_water_mask(sample_img.shape[:2], bounds, water_bodies)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original true-color
    axes[0].imshow(sample_img)
    axes[0].set_title("Original True-color (May 2024)")
    axes[0].axis("off")
    
    # Water mask
    axes[1].imshow(water_mask, cmap='Blues')
    axes[1].set_title("Water Bodies Mask")
    axes[1].axis("off")
    
    # NDCI with mask
    masked_ndci = ndci_maps["May 2024"]  # Keep NaNs
    cmap = plt.get_cmap("Spectral").copy()
    cmap.set_bad(color='white')  # or (1, 1, 1, 0) for transparent
    im = axes[2].imshow(masked_ndci, cmap=cmap, vmin=-1, vmax=1)
    axes[2].set_title("NDCI (Masked to Water Bodies)")
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], shrink=0.7)

    
    plt.tight_layout()
    plt.show()

# %%
# --- Print some statistics ---
if water_bodies is not None:
    print(f"\nWater body statistics:")
    print(f"Number of water bodies: {len(water_bodies)}")
    print(f"Total water area: {water_bodies.geometry.area.sum():.6f} degrees²")
    
    # NDCI statistics for water bodies
    for label, ndci_data in ndci_maps.items():
        valid_pixels = ~np.isnan(ndci_data)
        if np.any(valid_pixels):
            mean_ndci = np.nanmean(ndci_data)
            std_ndci = np.nanstd(ndci_data)
            min_ndci = np.nanmin(ndci_data)
            max_ndci = np.nanmax(ndci_data)
            
            print(f"\nNDCI Statistics for {label}:")
            print(f"  Mean: {mean_ndci:.4f}")
            print(f"  Std:  {std_ndci:.4f}")
            print(f"  Min:  {min_ndci:.4f}")
            print(f"  Max:  {max_ndci:.4f}")
            print(f"  Valid pixels: {np.sum(valid_pixels)}")
        else:
            print(f"\nNo valid NDCI data for {label}")

print("\nAnalysis complete!")