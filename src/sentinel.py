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
import pandas as pd  # For data manipulation and analysis
from dateutil.relativedelta import relativedelta  # For month calculations

# --- Configuration & credentials ---
CLIENT_ID = "sh-93c6fbd0-8c4a-4e40-8c59-d06889413797"
CLIENT_SECRET = "LKVq6MTE0S3kohQjRI1Yuj03aU5frOTm"
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"

# Path to your GeoJSON files
GEOJSON_PATH = "../data/alkmaar.geojson"
SURFACE_WATER_PATH = "../data/surface_water.geojson"

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
        if water_gdf.crs != "EPSG:4326":
            water_gdf = water_gdf.to_crs("EPSG:4326")
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
        dtype="uint8",
    )

    return water_mask.astype(bool)


# --- Function to get image bounds from geometry ---
def get_bounds_from_geometry(geometry):
    """Extract bounding box from GeoJSON geometry."""
    if geometry["type"] == "Polygon":
        coords = geometry["coordinates"][0]
    elif geometry["type"] == "MultiPolygon":
        # Get all coordinates and flatten
        coords = []
        for poly in geometry["coordinates"]:
            coords.extend(poly[0])
    else:
        raise ValueError(f"Unsupported geometry type: {geometry['type']}")

    lons = [coord[0] for coord in coords]
    lats = [coord[1] for coord in coords]

    return (min(lons), min(lats), max(lons), max(lats))


# %%
# --- Generic function to fetch any Sentinel-2 product ---
def fetch_sentinel2_product(
    product_type,
    start_date,
    end_date=None,
    max_cloud=10,
    width=2048,
    height=2048,
    apply_water_mask=True,
    mosaicking_order="leastCC",
):
    """
    Generic function to fetch different Sentinel-2 products.

    Args:
        product_type: str - 'RGB', 'NDVI', 'NDWI', or 'NDCI'
        start_date: str - ISO date string (YYYY-MM-DD) or datetime object
        end_date: str or None - ISO date string or datetime object. If None, uses start_date + 1 month
        max_cloud: int - Maximum cloud coverage percentage
        width, height: int - Image resolution in pixels
        apply_water_mask: bool - Whether to mask to water bodies only
        mosaicking_order: str - 'leastCC' or 'mostRecent'

    Returns:
        numpy array with the requested product
    """

    # Convert dates to proper format
    if isinstance(start_date, str):
        start_dt = datetime.fromisoformat(start_date)
    else:
        start_dt = start_date

    if end_date is None:
        end_dt = start_dt + relativedelta(months=1)
    elif isinstance(end_date, str):
        end_dt = datetime.fromisoformat(end_date)
    else:
        end_dt = end_date

    start_iso = start_dt.strftime("%Y-%m-%dT00:00:00Z")
    end_iso = end_dt.strftime("%Y-%m-%dT23:59:59Z")

    # Define band requirements and calculations for each product
    product_configs = {
        "RGB": {
            "bands": ["B04", "B03", "B02"],
            "calculation": lambda arr: arr,  # Return as-is
            "default_mask": False,  # Usually don't mask RGB for context
        },
        "NDVI": {
            "bands": ["B08", "B04"],
            "calculation": lambda arr: (arr[..., 0] - arr[..., 1])
            / (arr[..., 0] + arr[..., 1] + 1e-6),
            "default_mask": True,
        },
        "NDWI": {
            "bands": ["B03", "B08"],
            "calculation": lambda arr: (arr[..., 0] - arr[..., 1])
            / (arr[..., 0] + arr[..., 1] + 1e-6),
            "default_mask": True,
        },
        "NDCI": {
            "bands": ["B05", "B04"],
            "calculation": lambda arr: (arr[..., 0] - arr[..., 1])
            / (arr[..., 0] + arr[..., 1] + 1e-6),
            "default_mask": True,
        },
    }

    if product_type not in product_configs:
        raise ValueError(
            f"Unsupported product type: {product_type}. Choose from: {list(product_configs.keys())}"
        )

    config = product_configs[product_type]
    bands = config["bands"]

    # Use default masking behavior if not explicitly specified
    if apply_water_mask is None:
        apply_water_mask = config["default_mask"]

    # Build evalscript
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

    # Payload for Sentinel Hub API
    payload = {
        "input": {
            "bounds": {
                "geometry": geom,
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
            },
            "data": [
                {
                    "type": "S2L2A",
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

    # Fetch data
    r = requests.post(PROCESS_URL, headers=headers, json=payload)
    r.raise_for_status()

    # Process image
    img = Image.open(BytesIO(r.content))
    arr = np.array(img, dtype=np.uint8)

    # Remove alpha channel if present
    if arr.ndim == 3 and arr.shape[2] > len(bands):
        arr = arr[..., : len(bands)]

    # Scale to 0-1 range
    arr = arr.astype(np.float32) / 255.0

    # Apply calculation for the specific product
    result = config["calculation"](arr)

    # Apply water mask if requested
    if apply_water_mask and water_bodies is not None:
        bounds = get_bounds_from_geometry(geom)
        water_mask = create_water_mask(result.shape[:2], bounds, water_bodies)

        if result.ndim == 3:
            # Multi-band image (RGB)
            water_mask_3d = np.repeat(
                water_mask[:, :, np.newaxis], result.shape[2], axis=2
            )
            result = np.where(water_mask_3d, result, np.nan)
        else:
            # Single band image (indices)
            result = np.where(water_mask, result, np.nan)

    return result


# --- Helper functions for specific products ---
def fetch_true_color(start_iso, end_iso):
    """
    Fetch true-color (RGB) image: B04 (red), B03 (green), B02 (blue).
    """
    return fetch_s2(["B04", "B03", "B02"], start_iso, end_iso)


def fetch_ndci(start_iso, end_iso):
    """
    Compute the Normalized Difference Chlorophyll Index:
      NDCI = (B05 - B04) / (B05 + B04)
    B05 = red-edge band (channel 0), B04 = red (channel 1).
    """
    arr = fetch_s2(["B05", "B04"], start_iso, end_iso)
    b5, b4 = arr[..., 0], arr[..., 1]
    # Add small value to denominator to prevent division by zero
    return (b5 - b4) / (b5 + b4 + 1e-6)


# --- 1) True-color visualization for exactly two periods ---
dates_tc = {
    "May 2024": ("2024-05-01T00:00:00Z", "2024-05-31T23:59:59Z"),
    "April 2025": ("2025-04-01T00:00:00Z", "2025-04-30T23:59:59Z"),
}

# Fetch both RGB images and store in a dictionary
tc_maps = {
    label: fetch_true_color(start, end) for label, (start, end) in dates_tc.items()
}

# Plot in a single row with 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for ax, (label, img) in zip(axes, tc_maps.items()):
    ax.imshow(img)
    ax.set_title(f"True-color {label}")
    ax.axis("off")  # hide axes for cleaner look

plt.tight_layout()
plt.show()

# --- 2) Compute and visualize NDCI and ΔNDCI ---
# Use same periods as true-color for consistent comparison
dates_ndci = dates_tc

# Fetch NDCI maps
ndci_maps = {
    label: fetch_ndci(start, end) for label, (start, end) in dates_ndci.items()
}

# Compute difference between April 2025 and May 2024
delta_ndci = ndci_maps["April 2025"] - ndci_maps["May 2024"]

# Create 3 subplots: NDCI May 2024, NDCI April 2025, and ΔNDCI
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Define plotting settings for each axis
settings = [
    (ndci_maps["May 2024"], "NDCI May 2024", "Spectral", (-1, 1)),
    (ndci_maps["April 2025"], "NDCI April 2025", "Spectral", (-1, 1)),
    (delta_ndci, "Δ NDCI", "RdBu", (-1, 1)),
]

# Plot all three and store the imshow handles for colorbar
im_handles = []
for ax, (data, title, cmap, vlim) in zip(axes, settings):
    im = ax.imshow(data, cmap=cmap, vmin=vlim[0], vmax=vlim[1])
    ax.set_title(title)
    ax.axis("off")
    im_handles.append(im)

# Add one colorbar to the ΔNDCI plot (right subplot)
fig.colorbar(im_handles[-1], ax=axes[-1], shrink=0.7)

plt.tight_layout()
plt.show()
