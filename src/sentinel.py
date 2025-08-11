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


# --- Generic function to fetch Sentinel-2 data for any bands ---
def fetch_s2(
    bands,
    start_iso,
    end_iso,
    width=512,
    height=512,
    max_cloud=MAX_CLOUD,
    mosaicking_order="leastCC",
):
    """
    Fetch pixel reflectance values for specified bands over the given time period.
    bands         : list of band names (e.g. ["B04","B03","B02"])
    start_iso     : ISO date string for start of period
    end_iso       : ISO date string for end of period
    width, height : resolution of the image in pixels
    max_cloud     : maximum allowed cloud coverage (%)
    mosaicking_order: 'leastCC' or 'mostRecent'

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
    return arr.astype(np.float32) / 255.0


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
