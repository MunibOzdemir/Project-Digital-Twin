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
import pandas as pd  # For data manipulation and analysis
from dateutil.relativedelta import relativedelta  # For month calculations

# --- Configuration & credentials ---
CLIENT_ID = "sh-93c6fbd0-8c4a-4e40-8c59-d06889413797"
CLIENT_SECRET = "LKVq6MTE0S3kohQjRI1Yuj03aU5frOTm"
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"

# Path to your GeoJSON files
GEOJSON_PATH = r"C:\Users\munib\Desktop\Aanbesteding\Project\Project-Digital-Twin\data\alkmaar.geojson"
SURFACE_WATER_PATH = r"C:\Users\munib\Desktop\Aanbesteding\Project\Project-Digital-Twin\data\surface_water.geojson"

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
# --- Generic function to fetch any Sentinel-2 product ---
def fetch_sentinel2_product(
    product_type,
    start_date,
    end_date=None,
    max_cloud=10,
    width=2048,
    height=2048,
    apply_water_mask=True,
    mosaicking_order="leastCC"
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
        'RGB': {
            'bands': ['B04', 'B03', 'B02'],
            'calculation': lambda arr: arr,  # Return as-is
            'default_mask': False  # Usually don't mask RGB for context
        },
        'NDVI': {
            'bands': ['B08', 'B04'],
            'calculation': lambda arr: (arr[..., 0] - arr[..., 1]) / (arr[..., 0] + arr[..., 1] + 1e-6),
            'default_mask': True
        },
        'NDWI': {
            'bands': ['B03', 'B08'],
            'calculation': lambda arr: (arr[..., 0] - arr[..., 1]) / (arr[..., 0] + arr[..., 1] + 1e-6),
            'default_mask': True
        },
        'NDCI': {
            'bands': ['B05', 'B04'],
            'calculation': lambda arr: (arr[..., 0] - arr[..., 1]) / (arr[..., 0] + arr[..., 1] + 1e-6),
            'default_mask': True
        }
    }
    
    if product_type not in product_configs:
        raise ValueError(f"Unsupported product type: {product_type}. Choose from: {list(product_configs.keys())}")
    
    config = product_configs[product_type]
    bands = config['bands']
    
    # Use default masking behavior if not explicitly specified
    if apply_water_mask is None:
        apply_water_mask = config['default_mask']
    
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
    result = config['calculation'](arr)
    
    # Apply water mask if requested
    if apply_water_mask and water_bodies is not None:
        bounds = get_bounds_from_geometry(geom)
        water_mask = create_water_mask(result.shape[:2], bounds, water_bodies)
        
        if result.ndim == 3:
            # Multi-band image (RGB)
            water_mask_3d = np.repeat(water_mask[:, :, np.newaxis], result.shape[2], axis=2)
            result = np.where(water_mask_3d, result, np.nan)
        else:
            # Single band image (indices)
            result = np.where(water_mask, result, np.nan)
    
    return result

# %%
# --- Function to get all available dates from Sentinel Hub Catalog ---
def get_available_dates(year, max_cloud=50):
    """
    Query Sentinel Hub catalog to get all available dates for the specified year.
    
    Args:
        year: int - Year to query
        max_cloud: int - Maximum cloud coverage to filter results
    
    Returns:
        List of available dates as datetime objects
    """
    catalog_url = "https://sh.dataspace.copernicus.eu/api/v1/catalog/search"
    
    # Define search parameters for the entire year
    start_date = f"{year}-01-01T00:00:00Z"
    end_date = f"{year + 1}-01-01T00:00:00Z"
    
    search_payload = {
        "collections": ["sentinel-2-l2a"],
        "datetime": f"{start_date}/{end_date}",
        "bbox": get_bounds_from_geometry(geom),
        "limit": 500,  # Maximum results per page
        "query": {
            "eo:cloud_cover": {"lt": max_cloud}
        }
    }
    
    try:
        response = requests.post(catalog_url, headers=headers, json=search_payload)
        response.raise_for_status()
        catalog_data = response.json()
        
        # Extract dates from catalog results
        available_dates = []
        for feature in catalog_data.get('features', []):
            date_str = feature['properties']['datetime']
            # Parse the datetime string (format: 2024-05-15T10:30:25.123Z)
            date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            available_dates.append(date_obj.replace(tzinfo=None))  # Remove timezone for consistency
        
        # Remove duplicates and sort
        available_dates = sorted(list(set(available_dates)))
        
        print(f"Found {len(available_dates)} available Sentinel-2 acquisitions in {year} with <{max_cloud}% cloud cover")
        return available_dates
        
    except Exception as e:
        print(f"Error querying catalog: {e}")
        print("Falling back to daily interval approach...")
        return None

# --- Function to validate NDCI throughout a year with all available data ---
def validate_ndci_annual(year=2024, max_cloud=20, use_all_available=True, time_window_days=3):
    """
    Calculate NDCI throughout a year to validate the calculation using all available data.
    
    Args:
        year: int - Year to analyze
        max_cloud: int - Maximum cloud coverage
        use_all_available: bool - If True, query catalog for all available dates; if False, use monthly intervals
        time_window_days: int - Number of days around each date to search for data
    
    Returns:
        dict with dates as keys and NDCI statistics as values
    """
    print(f"Validating NDCI calculation for year {year}...")
    
    results = {}
    
    if use_all_available:
        # Get all available dates from catalog
        available_dates = get_available_dates(year, max_cloud)
        
        if available_dates is None:
            # Fallback to monthly intervals if catalog query fails
            print("Using monthly fallback approach...")
            return validate_ndci_annual(year, max_cloud, use_all_available=False)
        
        # Process each available date
        processed_dates = set()
        for i, date in enumerate(available_dates):
            # Skip if we've already processed a very similar date
            date_key = date.strftime("%Y-%m-%d")
            if any(abs((date - pd).days) < 2 for pd in processed_dates):
                continue
                
            try:
                print(f"Processing {date_key} ({i+1}/{len(available_dates)})...")
                
                # Create time window around this date
                start_date = date - timedelta(days=time_window_days//2)
                end_date = date + timedelta(days=time_window_days//2)
                
                # Fetch NDCI for this time window
                ndci = fetch_sentinel2_product(
                    product_type='NDCI',
                    start_date=start_date,
                    end_date=end_date,
                    max_cloud=max_cloud,
                    apply_water_mask=True,
                    mosaicking_order="leastCC"  # Use least cloudy for individual dates
                )
                
                # Calculate statistics
                valid_mask = ~np.isnan(ndci)
                if np.any(valid_mask):
                    stats = {
                        'mean': np.nanmean(ndci),
                        'median': np.nanmedian(ndci),
                        'std': np.nanstd(ndci),
                        'min': np.nanmin(ndci),
                        'max': np.nanmax(ndci),
                        'valid_pixels': np.sum(valid_mask),
                        'date': date,
                        'month': date.month,
                        'day_of_year': date.timetuple().tm_yday,
                        'acquisition_date': date_key
                    }
                    results[date_key] = stats
                    processed_dates.add(date)
                    print(f"  Mean NDCI: {stats['mean']:.4f}, Valid pixels: {stats['valid_pixels']}")
                else:
                    print(f"  No valid data for {date_key}")
                    
            except Exception as e:
                print(f"  Error processing {date_key}: {e}")
                continue
    
    else:
        # Monthly analysis (original approach)
        intervals = []
        for month in range(1, 13):
            start_date = datetime(year, month, 1)
            if month == 12:
                end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
            else:
                end_date = datetime(year, month + 1, 1) - timedelta(days=1)
            intervals.append((start_date, end_date, f"{year}-{month:02d}"))
        
        for start_date, end_date, label in intervals:
            try:
                print(f"Processing {label}...")
                
                # Fetch NDCI for this period
                ndci = fetch_sentinel2_product(
                    product_type='NDCI',
                    start_date=start_date,
                    end_date=end_date,
                    max_cloud=max_cloud,
                    apply_water_mask=True
                )
                
                # Calculate statistics
                valid_mask = ~np.isnan(ndci)
                if np.any(valid_mask):
                    stats = {
                        'mean': np.nanmean(ndci),
                        'median': np.nanmedian(ndci),
                        'std': np.nanstd(ndci),
                        'min': np.nanmin(ndci),
                        'max': np.nanmax(ndci),
                        'valid_pixels': np.sum(valid_mask),
                        'date': start_date,
                        'month': start_date.month,
                        'day_of_year': start_date.timetuple().tm_yday if hasattr(start_date, 'timetuple') else None
                    }
                    results[label] = stats
                    print(f"  Mean NDCI: {stats['mean']:.4f}, Valid pixels: {stats['valid_pixels']}")
                else:
                    print(f"  No valid data for {label}")
                    
            except Exception as e:
                print(f"  Error processing {label}: {e}")
                continue
    
    print(f"Successfully processed {len(results)} time periods/dates")
    return results

# --- Enhanced function to plot annual NDCI validation ---
def plot_ndci_validation(validation_results, use_all_available=True):
    """
    Plot the annual NDCI validation results with enhanced visualizations.
    
    Args:
        validation_results: dict from validate_ndci_annual()
        use_all_available: bool - If True, creates time-series plot; if False, monthly aggregation
    """
    if not validation_results:
        print("No validation results to plot")
        return
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(validation_results).T
    
    # Ensure proper data types for date column
    try:
        df['date'] = pd.to_datetime([
            stats['date'] if isinstance(stats['date'], datetime) 
            else datetime.strptime(stats['acquisition_date'], '%Y-%m-%d') 
            if 'acquisition_date' in stats else stats['date'] 
            for stats in validation_results.values()
        ])
    except Exception as e:
        print(f"Warning: Date conversion issue: {e}")
        # Fallback to simple date handling
        df['date'] = [stats.get('date', stats.get('acquisition_date', 'unknown')) 
                     for stats in validation_results.values()]
    
    # Ensure numeric columns are properly typed
    numeric_columns = ['mean', 'std', 'min', 'max', 'valid_pixels', 'month']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.sort_values('date')
    
    if use_all_available and 'day_of_year' in df.columns:
        # Time series analysis with all available data
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        
        # Plot 1: Full time series
        axes[0, 0].scatter(df['day_of_year'], df['mean'], alpha=0.6, s=30)
        axes[0, 0].plot(df['day_of_year'], df['mean'].rolling(window=10, center=True).mean(), 
                       color='red', linewidth=2, label='10-point moving average')
        axes[0, 0].set_xlabel('Day of Year')
        axes[0, 0].set_ylabel('Mean NDCI')
        axes[0, 0].set_title('Daily NDCI Throughout the Year')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Highlight seasons
        season_colors = {'Winter': 'lightblue', 'Spring': 'lightgreen', 
                        'Summer': 'lightyellow', 'Fall': 'orange'}
        season_ranges = [(1, 79, 'Winter'), (80, 171, 'Spring'), 
                        (172, 266, 'Summer'), (267, 365, 'Fall')]
        
        for start, end, season in season_ranges:
            axes[0, 0].axvspan(start, end, alpha=0.2, color=season_colors[season], label=season)
        
        # Plot 2: Monthly aggregation from daily data
        monthly_stats = df.groupby('month').agg({
            'mean': ['mean', 'std', 'count'],
            'valid_pixels': 'sum'
        }).round(4)
        monthly_stats.columns = ['mean_ndci', 'std_ndci', 'count', 'total_pixels']
        
        months = monthly_stats.index.tolist()
        monthly_means = monthly_stats['mean_ndci'].tolist()
        monthly_stds = monthly_stats['std_ndci'].tolist()
        
        axes[0, 1].errorbar(months, monthly_means, yerr=monthly_stds, 
                           fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=8)
        axes[0, 1].set_xlabel('Month')
        axes[0, 1].set_ylabel('Mean NDCI ± Std')
        axes[0, 1].set_title('Monthly NDCI Statistics (from daily data)')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_xticks(range(1, 13))
        
        # Highlight summer months
        summer_months = [6, 7, 8]
        for month in summer_months:
            if month in months:
                idx = months.index(month)
                axes[0, 1].scatter(month, monthly_means[idx], color='red', s=150, zorder=5)
        
        # Plot 3: Data availability
        axes[1, 0].bar(monthly_stats.index, monthly_stats['count'], alpha=0.7, color='skyblue')
        axes[1, 0].set_xlabel('Month')
        axes[1, 0].set_ylabel('Number of Acquisitions')
        axes[1, 0].set_title('Data Availability by Month')
        axes[1, 0].set_xticks(range(1, 13))
        
        # Plot 4: Seasonal comparison with more detail
        season_data = {
            'Winter (Dec-Feb)': df[df['month'].isin([12, 1, 2])]['mean'],
            'Spring (Mar-May)': df[df['month'].isin([3, 4, 5])]['mean'],
            'Summer (Jun-Aug)': df[df['month'].isin([6, 7, 8])]['mean'],
            'Fall (Sep-Nov)': df[df['month'].isin([9, 10, 11])]['mean']
        }
        
        # Box plot for seasonal comparison
        seasonal_data = [data.dropna().tolist() for data in season_data.values() if len(data.dropna()) > 0]
        seasonal_labels = [label for label, data in season_data.items() if len(data.dropna()) > 0]
        
        if seasonal_data:
            bp = axes[1, 1].boxplot(seasonal_data, labels=seasonal_labels, patch_artist=True)
            colors = ['lightblue', 'lightgreen', 'red', 'orange']
            for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            axes[1, 1].set_ylabel('NDCI')
            axes[1, 1].set_title('Seasonal NDCI Distribution')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # Enhanced statistics
        print(f"\n=== Enhanced NDCI Validation Summary ===")
        print(f"Total acquisitions: {len(df)}")
        print(f"Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
        
        # Seasonal statistics
        for season, data in season_data.items():
            if len(data.dropna()) > 0:
                # Ensure numeric data
                data_numeric = pd.to_numeric(data, errors='coerce').dropna()
                if len(data_numeric) > 0:
                    print(f"{season}: Mean={data_numeric.mean():.4f}, Std={data_numeric.std():.4f}, N={len(data_numeric)}")
                else:
                    print(f"{season}: No valid numeric data")
        
        # Summer vs non-summer comparison
        summer_data = df[df['month'].isin([6, 7, 8])]['mean'].dropna()
        non_summer_data = df[~df['month'].isin([6, 7, 8])]['mean'].dropna()
        
        if len(summer_data) > 0 and len(non_summer_data) > 0:
            # Convert to numeric arrays to avoid dtype issues
            summer_data = pd.to_numeric(summer_data, errors='coerce').dropna()
            non_summer_data = pd.to_numeric(non_summer_data, errors='coerce').dropna()
            
            if len(summer_data) > 1 and len(non_summer_data) > 1:
                summer_avg = summer_data.mean()
                non_summer_avg = non_summer_data.mean()
                print(f"\nSummer vs Non-Summer:")
                print(f"Summer average: {summer_avg:.4f} (N={len(summer_data)})")
                print(f"Non-summer average: {non_summer_avg:.4f} (N={len(non_summer_data)})")
                print(f"Difference: {((summer_avg - non_summer_avg) / abs(non_summer_avg) * 100):+.1f}%")
                
                # Statistical significance test with proper error handling
                try:
                    from scipy import stats
                    # Convert to numpy arrays with float64 dtype
                    summer_array = np.array(summer_data, dtype=np.float64)
                    non_summer_array = np.array(non_summer_data, dtype=np.float64)
                    
                    # Check for valid data
                    if (len(summer_array) > 1 and len(non_summer_array) > 1 and 
                        np.all(np.isfinite(summer_array)) and np.all(np.isfinite(non_summer_array))):
                        
                        t_stat, p_value = stats.ttest_ind(summer_array, non_summer_array)
                        print(f"T-test p-value: {p_value:.6f}")
                        if p_value < 0.05:
                            print("✓ Summer increase is statistically significant (p<0.05)")
                        else:
                            print("⚠ Summer increase is not statistically significant (p≥0.05)")
                    else:
                        print("⚠ Cannot perform statistical test - insufficient valid data")
                        
                except Exception as e:
                    print(f"⚠ Statistical test failed: {e}")
                    print("• Proceeding with descriptive statistics only")
                
                if summer_avg > non_summer_avg:
                    print("✓ NDCI shows expected summer increase - calculation likely correct!")
                else:
                    print("⚠ NDCI does not show expected summer increase - check calculation or data")
            else:
                print("\n⚠ Insufficient data for statistical comparison")
        else:
            print("\n⚠ No seasonal data available for comparison")
    
    else:
        # Fall back to original monthly plotting if not using all available data
        months = df['month'].tolist()
        means = df['mean'].tolist()
        stds = df['std'].tolist()
        valid_counts = df['valid_pixels'].tolist()
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Mean NDCI over months
        axes[0, 0].plot(months, means, 'o-', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('Month')
        axes[0, 0].set_ylabel('Mean NDCI')
        axes[0, 0].set_title('Monthly Mean NDCI (Water Bodies)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xticks(range(1, 13))
        
        # Highlight summer months
        summer_months = [6, 7, 8]
        for month in summer_months:
            if month in months:
                idx = months.index(month)
                axes[0, 0].scatter(month, means[idx], color='red', s=100, zorder=5)
        
        # Plot 2: NDCI variability (error bars)
        axes[0, 1].errorbar(months, means, yerr=stds, fmt='o-', capsize=5, capthick=2)
        axes[0, 1].set_xlabel('Month')
        axes[0, 1].set_ylabel('NDCI ± Std Dev')
        axes[0, 1].set_title('Monthly NDCI Variability')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_xticks(range(1, 13))
        
        # Plot 3: Valid pixel count
        axes[1, 0].bar(months, valid_counts, alpha=0.7, color='skyblue')
        axes[1, 0].set_xlabel('Month')
        axes[1, 0].set_ylabel('Valid Pixels')
        axes[1, 0].set_title('Data Availability by Month')
        axes[1, 0].set_xticks(range(1, 13))
        
        # Plot 4: Summer vs other seasons comparison
        summer_means = [means[months.index(m)] for m in summer_months if m in months]
        other_means = [means[i] for i, m in enumerate(months) if m not in summer_months]
        
        if summer_means and other_means:
            categories = ['Summer\n(Jun-Aug)', 'Other\n(Sep-May)']
            values = [np.mean(summer_means), np.mean(other_means)]
            colors = ['red', 'blue']
            
            bars = axes[1, 1].bar(categories, values, color=colors, alpha=0.7)
            axes[1, 1].set_ylabel('Mean NDCI')
            axes[1, 1].set_title('Summer vs Other Seasons')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                               f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        if summer_means and other_means:
            summer_avg = np.mean(summer_means)
            other_avg = np.mean(other_means)
            print(f"\n=== NDCI Validation Summary ===")
            print(f"Summer average NDCI: {summer_avg:.4f}")
            print(f"Other months average NDCI: {other_avg:.4f}")
            print(f"Summer increase: {((summer_avg - other_avg) / abs(other_avg) * 100):+.1f}%")
            
            if summer_avg > other_avg:
                print("✓ NDCI shows expected summer increase - calculation likely correct!")
            else:
                print("⚠ NDCI does not show expected summer increase - check calculation or data")

# %%
# --- Example usage of the new functions ---

# 1. Fetch different products for specific dates
print("=== Testing Generic Product Function ===")

# Example: Get RGB for May 2024
rgb_may = fetch_sentinel2_product(
    product_type='RGB',
    start_date='2024-05-01',
    max_cloud=10,
    apply_water_mask=False  # Keep full context for RGB
)

# Example: Get NDCI for April 2025 (water bodies only)
ndci_april = fetch_sentinel2_product(
    product_type='NDCI',
    start_date='2025-04-01',
    max_cloud=5,
    apply_water_mask=True
)

# Example: Get NDVI for comparison
ndvi_april = fetch_sentinel2_product(
    product_type='NDVI',
    start_date='2025-04-01',
    max_cloud=5,
    apply_water_mask=True
)

# Visualize the results
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# RGB
axes[0].imshow(rgb_may)
axes[0].set_title('RGB - May 2024')
axes[0].axis('off')

# NDCI
im1 = axes[1].imshow(ndci_april, cmap='Spectral', vmin=-1, vmax=1)
axes[1].set_title('NDCI - April 2025 (Water Bodies)')
axes[1].axis('off')
fig.colorbar(im1, ax=axes[1], shrink=0.7)

# NDVI
im2 = axes[2].imshow(ndvi_april, cmap='RdYlGn', vmin=-1, vmax=1)
axes[2].set_title('NDVI - April 2025 (Water Bodies)')
axes[2].axis('off')
fig.colorbar(im2, ax=axes[2], shrink=0.7)

plt.tight_layout()
plt.show()

# %%
# 2. Enhanced NDCI validation with all available data
print("\n=== Enhanced NDCI Annual Validation ===")

# Run validation with all available dates (this will take longer but provide better validation)
validation_results = validate_ndci_annual(
    year=2024,
    max_cloud=30,  # Higher cloud tolerance for validation
    use_all_available=True,  # Use all available acquisition dates
    time_window_days=3  # Small time window around each date
)

# Plot enhanced validation results
if validation_results:
    plot_ndci_validation(validation_results, use_all_available=True)
    
    # Create a detailed DataFrame for analysis
    df_results = pd.DataFrame(validation_results).T
    print("\nSample of detailed NDCI statistics:")
    
    # Check which columns actually exist and display them
    available_columns = df_results.columns.tolist()
    print(f"Available columns: {available_columns}")
    
    # Display relevant columns that exist
    display_columns = []
    for col in ['date', 'month', 'mean', 'std', 'valid_pixels', 'day_of_year']:
        if col in available_columns:
            display_columns.append(col)
    
    if display_columns:
        print(df_results[display_columns].head(10).round(4))
    else:
        print("Basic DataFrame structure:")
        print(df_results.head(10))
    
    # Export results to CSV for further analysis (optional)
    # df_results.to_csv(f'ndci_validation_{validation_results[list(validation_results.keys())[0]]["date"].year}.csv')
    
else:
    print("No validation results obtained. Trying monthly fallback...")
    # Fallback to monthly validation
    validation_results_monthly = validate_ndci_annual(
        year=2024,
        max_cloud=50,
        use_all_available=False
    )
    if validation_results_monthly:
        plot_ndci_validation(validation_results_monthly, use_all_available=False)

print("\nEnhanced validation analysis complete!")

print("\nAnalysis complete!")