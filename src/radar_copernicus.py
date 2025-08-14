# %%
import requests
import json
import numpy as np
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import geopandas as gpd
from rasterio.features import rasterize
from rasterio.transform import from_bounds
import shapely.geometry
import pandas as pd
from dateutil.relativedelta import relativedelta
from scipy import ndimage, stats
from skimage import filters, measure, morphology
import warnings
from tools import get_geojson_path
warnings.filterwarnings('ignore')

# --- Configuration & credentials ---
CLIENT_ID = "sh-93c6fbd0-8c4a-4e40-8c59-d06889413797"
CLIENT_SECRET = "LKVq6MTE0S3kohQjRI1Yuj03aU5frOTm"
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"

# Path to your GeoJSON files
GEOJSON_PATH = path_geojson = get_geojson_path('alkmaar.geojson')
SURFACE_WATER_PATH = r"C:\Users\munib\Desktop\Aanbesteding\Project\Project-Digital-Twin\data\surface_water.geojson"

# Read GeoJSON and extract correct geometry
with open(GEOJSON_PATH) as f:
    gj = json.load(f)

geom = gj["features"][0]["geometry"] if "features" in gj else gj.get("geometry", gj)

# Get token and headers
def get_token():
    r = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
        },
    )
    r.raise_for_status()
    return r.json()["access_token"]

token = get_token()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

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

# Improved Sentinel-1 fetch function
def fetch_sentinel1_product(
    product_type,
    start_date,
    end_date=None,
    orbit_direction="DESCENDING",
    width=1024,
    height=1024,
    mosaicking_order="mostRecent"
):
    """Enhanced Sentinel-1 product fetching with better error handling."""
    
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
    
    # Evalscripts optimized for different applications
    if product_type == 'VV':
        evalscript = """
        //VERSION=3
        function setup() {
            return {
                input: ["VV"],
                output: { bands: 1, sampleType: "UINT8" }
            };
        }
        function evaluatePixel(sample) {
            let vv_linear = Math.max(sample.VV, 0.0001);
            let vv_db = 10 * Math.log(vv_linear) / Math.LN10;
            let normalized = Math.max(0, Math.min(1, (vv_db + 25) / 25));
            return [normalized];
        }
        """
    elif product_type == 'VH':
        evalscript = """
        //VERSION=3
        function setup() {
            return {
                input: ["VH"],
                output: { bands: 1, sampleType: "UINT8" }
            };
        }
        function evaluatePixel(sample) {
            let vh_linear = Math.max(sample.VH, 0.0001);
            let vh_db = 10 * Math.log(vh_linear) / Math.LN10;
            let normalized = Math.max(0, Math.min(1, (vh_db + 30) / 25));
            return [normalized];
        }
        """
    elif product_type == 'VV_VH_RATIO':
        evalscript = """
        //VERSION=3
        function setup() {
            return {
                input: ["VV", "VH"],
                output: { bands: 1, sampleType: "UINT8" }
            };
        }
        function evaluatePixel(sample) {
            let vv_linear = Math.max(sample.VV, 0.0001);
            let vh_linear = Math.max(sample.VH, 0.0001);
            let ratio = vv_linear / vh_linear;
            let ratio_db = 10 * Math.log(ratio) / Math.LN10;
            // Normalize ratio (typical range 0-15 dB)
            let normalized = Math.max(0, Math.min(1, ratio_db / 15));
            return [normalized];
        }
        """
    elif product_type == 'RGB_VV_VH':
        evalscript = """
        //VERSION=3
        function setup() {
            return {
                input: ["VV", "VH"],
                output: { bands: 3, sampleType: "UINT8" }
            };
        }
        function evaluatePixel(sample) {
            let vv_linear = Math.max(sample.VV, 0.0001);
            let vh_linear = Math.max(sample.VH, 0.0001);
            
            let vv_db = 10 * Math.log(vv_linear) / Math.LN10;
            let vh_db = 10 * Math.log(vh_linear) / Math.LN10;
            
            let vv_norm = Math.max(0, Math.min(1, (vv_db + 25) / 25));
            let vh_norm = Math.max(0, Math.min(1, (vh_db + 30) / 25));
            
            return [vv_norm, vh_norm, (vv_norm + vh_norm) / 2];
        }
        """
    else:
        raise ValueError(f"Unsupported product type: {product_type}")
    
    payload = {
        "input": {
            "bounds": {
                "geometry": geom,
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
            },
            "data": [
                {
                    "type": "S1GRD",
                    "dataFilter": {
                        "timeRange": {"from": start_iso, "to": end_iso},
                        "orbitDirection": orbit_direction
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
    
    try:
        r = requests.post(PROCESS_URL, headers=headers, json=payload)
        if r.status_code != 200:
            print(f"Error {r.status_code}: {r.text}")
            r.raise_for_status()
        
        img = Image.open(BytesIO(r.content))
        arr = np.array(img, dtype=np.float32) / 255.0
        
        # Convert back to appropriate units
        if product_type == 'VV':
            arr = (arr * 25) - 25  # Convert to dB
        elif product_type == 'VH':
            arr = (arr * 25) - 30  # Convert to dB
        elif product_type == 'VV_VH_RATIO':
            arr = arr * 15  # Convert to ratio dB
        
        return arr.squeeze() if arr.ndim == 3 and arr.shape[2] == 1 else arr
        
    except Exception as e:
        print(f"Error fetching {product_type} data: {e}")
        raise

# %%
# === 1. SEASONAL FLOOD MAPPING ===

def detect_flood_extent(
    baseline_dates,
    flood_dates,
    water_threshold_vv=-18,
    water_threshold_vh=-22,
    min_flood_area=50,
    orbit_direction="DESCENDING"
):
    """
    Detect flood extent by comparing baseline and flood conditions.
    
    Args:
        baseline_dates: list of dates for baseline (dry) conditions
        flood_dates: list of dates during potential flooding
        water_threshold_vv: dB threshold for water detection in VV
        water_threshold_vh: dB threshold for water detection in VH  
        min_flood_area: minimum connected area for valid flood zones
        orbit_direction: satellite orbit direction
    
    Returns:
        dict with flood analysis results
    """
    print("=== FLOOD DETECTION ANALYSIS ===")
    print(f"Analyzing flood extent for Alkmaar water management...")
    
    try:
        # Get baseline (dry) conditions
        print("Fetching baseline conditions...")
        baseline_vv_images = []
        baseline_vh_images = []
        
        for date in baseline_dates:
            try:
                vv_data = fetch_sentinel1_product('VV', date, orbit_direction=orbit_direction)
                vh_data = fetch_sentinel1_product('VH', date, orbit_direction=orbit_direction)
                baseline_vv_images.append(vv_data)
                baseline_vh_images.append(vh_data)
                print(f"  ✓ Baseline data for {date}")
            except Exception as e:
                print(f"  ✗ Failed to get baseline data for {date}: {e}")
        
        # Get flood conditions
        print("Fetching flood conditions...")
        flood_vv_images = []
        flood_vh_images = []
        
        for date in flood_dates:
            try:
                vv_data = fetch_sentinel1_product('VV', date, orbit_direction=orbit_direction)
                vh_data = fetch_sentinel1_product('VH', date, orbit_direction=orbit_direction)
                flood_vv_images.append(vv_data)
                flood_vh_images.append(vh_data)
                print(f"  ✓ Flood data for {date}")
            except Exception as e:
                print(f"  ✗ Failed to get flood data for {date}: {e}")
        
        if not baseline_vv_images or not flood_vv_images:
            print("Insufficient data for flood analysis")
            return None
        
        # Calculate mean baseline and flood conditions
        baseline_vv = np.mean(baseline_vv_images, axis=0)
        baseline_vh = np.mean(baseline_vh_images, axis=0)
        flood_vv = np.mean(flood_vv_images, axis=0)
        flood_vh = np.mean(flood_vh_images, axis=0)
        
        # Water detection using dual polarization
        baseline_water = (baseline_vv < water_threshold_vv) & (baseline_vh < water_threshold_vh)
        flood_water = (flood_vv < water_threshold_vv) & (flood_vh < water_threshold_vh)
        
        # Calculate flood extent (new water areas)
        new_flood_areas = flood_water & ~baseline_water
        
        # Remove small isolated areas
        if min_flood_area > 1:
            new_flood_areas = morphology.remove_small_objects(
                new_flood_areas, min_size=min_flood_area
            )
        
        # Calculate statistics
        baseline_water_area = np.sum(baseline_water)
        flood_water_area = np.sum(flood_water)
        new_flood_area = np.sum(new_flood_areas)
        total_pixels = baseline_vv.size
        
        # Identify flood regions
        flood_regions = measure.regionprops(measure.label(new_flood_areas))
        
        results = {
            'baseline_vv': baseline_vv,
            'baseline_vh': baseline_vh,
            'flood_vv': flood_vv,
            'flood_vh': flood_vh,
            'baseline_water': baseline_water,
            'flood_water': flood_water,
            'new_flood_areas': new_flood_areas,
            'flood_regions': flood_regions,
            'statistics': {
                'baseline_water_area': baseline_water_area,
                'flood_water_area': flood_water_area,
                'new_flood_area': new_flood_area,
                'total_pixels': total_pixels,
                'baseline_water_percentage': (baseline_water_area / total_pixels) * 100,
                'flood_water_percentage': (flood_water_area / total_pixels) * 100,
                'new_flood_percentage': (new_flood_area / total_pixels) * 100,
                'flood_increase_factor': flood_water_area / max(baseline_water_area, 1),
                'num_flood_regions': len(flood_regions)
            },
            'dates': {
                'baseline': baseline_dates,
                'flood': flood_dates
            },
            'thresholds': {
                'vv_threshold': water_threshold_vv,
                'vh_threshold': water_threshold_vh
            }
        }
        
        print(f"✓ Flood analysis completed!")
        print(f"  Baseline water: {results['statistics']['baseline_water_percentage']:.2f}%")
        print(f"  Flood water: {results['statistics']['flood_water_percentage']:.2f}%")
        print(f"  New flood areas: {results['statistics']['new_flood_percentage']:.2f}%")
        
        return results
        
    except Exception as e:
        print(f"Error in flood detection: {e}")
        return None

def plot_flood_analysis(flood_results):
    """Plot comprehensive flood analysis results."""
    if not flood_results:
        return
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Baseline conditions
    im1 = axes[0, 0].imshow(flood_results['baseline_vv'], cmap='viridis', vmin=-25, vmax=0)
    axes[0, 0].set_title('Baseline VV (dB)')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)
    
    im2 = axes[0, 1].imshow(flood_results['flood_vv'], cmap='viridis', vmin=-25, vmax=0)
    axes[0, 1].set_title('Flood VV (dB)')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)
    
    # Water detection
    axes[0, 2].imshow(flood_results['baseline_water'], cmap='Blues')
    axes[0, 2].set_title(f'Baseline Water\n{flood_results["statistics"]["baseline_water_percentage"]:.1f}%')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(flood_results['flood_water'], cmap='Blues')
    axes[0, 3].set_title(f'Flood Water\n{flood_results["statistics"]["flood_water_percentage"]:.1f}%')
    axes[0, 3].axis('off')
    
    # Changes analysis
    vv_change = flood_results['flood_vv'] - flood_results['baseline_vv']
    im3 = axes[1, 0].imshow(vv_change, cmap='RdBu_r', vmin=-5, vmax=5)
    axes[1, 0].set_title('VV Change (dB)')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0], shrink=0.8)
    
    # New flood areas
    axes[1, 1].imshow(flood_results['new_flood_areas'], cmap='Reds')
    axes[1, 1].set_title(f'New Flood Areas\n{flood_results["statistics"]["new_flood_percentage"]:.2f}%')
    axes[1, 1].axis('off')
    
    # Combined flood map
    flood_overlay = np.zeros((*flood_results['baseline_vv'].shape, 3))
    flood_overlay[flood_results['baseline_water'], :] = [0, 0, 0.7]  # Existing water - blue
    flood_overlay[flood_results['new_flood_areas'], :] = [1, 0, 0]  # New floods - red
    
    axes[1, 2].imshow(flood_overlay)
    axes[1, 2].set_title('Flood Map\n(Blue: Existing, Red: New)')
    axes[1, 2].axis('off')
    
    # Statistics
    stats = flood_results['statistics']
    categories = ['Baseline\nWater', 'Flood\nWater', 'New\nFlood']
    percentages = [stats['baseline_water_percentage'], 
                  stats['flood_water_percentage'], 
                  stats['new_flood_percentage']]
    colors = ['blue', 'cyan', 'red']
    
    bars = axes[1, 3].bar(categories, percentages, color=colors, alpha=0.7)
    axes[1, 3].set_ylabel('Percentage of Area (%)')
    axes[1, 3].set_title('Water Coverage Statistics')
    
    for bar, value in zip(bars, percentages):
        axes[1, 3].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                       f'{value:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Summary report
    print(f"\n=== FLOOD ANALYSIS REPORT ===")
    print(f"Analysis period: {flood_results['dates']['baseline']} (baseline) vs {flood_results['dates']['flood']} (flood)")
    print(f"Detection thresholds: VV < {flood_results['thresholds']['vv_threshold']} dB, VH < {flood_results['thresholds']['vh_threshold']} dB")
    print(f"\nWater Coverage:")
    print(f"  Normal conditions: {stats['baseline_water_percentage']:.2f}% of area")
    print(f"  Flood conditions: {stats['flood_water_percentage']:.2f}% of area")
    print(f"  New flood areas: {stats['new_flood_percentage']:.2f}% of area")
    print(f"  Flood increase factor: {stats['flood_increase_factor']:.1f}x")
    print(f"  Number of flood regions: {stats['num_flood_regions']}")
    
    if stats['new_flood_percentage'] > 1.0:
        print("\n🚨 SIGNIFICANT FLOODING DETECTED")
        print("   Recommended actions:")
        print("   • Activate flood management protocols")
        print("   • Monitor water levels in affected areas")
        print("   • Check drainage and pumping systems")
    elif stats['new_flood_percentage'] > 0.1:
        print("\n⚠️  Minor flooding detected")
        print("   • Monitor situation")
        print("   • Check local drainage capacity")
    else:
        print("\n✅ No significant flooding detected")

# %%
# === 2. CONSTRUCTION TIMELINE ANALYSIS ===

def analyze_construction_timeline(
    start_date,
    end_date,
    construction_areas=None,
    time_interval_months=2,
    change_threshold=1.5,
    orbit_direction="DESCENDING"
):
    """
    Analyze construction activity timeline using radar time series.
    
    Args:
        start_date: str - Start of analysis period
        end_date: str - End of analysis period
        construction_areas: GeoDataFrame of known construction zones (optional)
        time_interval_months: int - Months between analysis points
        change_threshold: float - dB threshold for construction detection
        orbit_direction: str - Satellite orbit direction
    
    Returns:
        dict with construction timeline analysis
    """
    print("=== CONSTRUCTION TIMELINE ANALYSIS ===")
    print(f"Tracking construction activity from {start_date} to {end_date}...")
    
    try:
        # Generate time series dates
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)
        
        dates = []
        current_date = start_dt
        while current_date <= end_dt:
            dates.append(current_date.strftime('%Y-%m-%d'))
            current_date += relativedelta(months=time_interval_months)
        
        print(f"Analysis dates: {dates}")
        
        # Fetch time series data
        vv_series = []
        ratio_series = []
        rgb_series = []
        valid_dates = []
        
        for date in dates:
            try:
                print(f"Fetching data for {date}...")
                vv_data = fetch_sentinel1_product('VV', date, orbit_direction=orbit_direction)
                ratio_data = fetch_sentinel1_product('VV_VH_RATIO', date, orbit_direction=orbit_direction)
                rgb_data = fetch_sentinel1_product('RGB_VV_VH', date, orbit_direction=orbit_direction)
                
                vv_series.append(vv_data)
                ratio_series.append(ratio_data)
                rgb_series.append(rgb_data)
                valid_dates.append(date)
                print(f"  ✓ Success")
                
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                continue
        
        if len(vv_series) < 2:
            print("Insufficient data for timeline analysis")
            return None
        
        # Analyze construction indicators
        construction_timeline = []
        change_maps = []
        
        reference_vv = vv_series[0]
        reference_ratio = ratio_series[0]
        
        for i, (vv_current, ratio_current, date) in enumerate(zip(vv_series[1:], ratio_series[1:], valid_dates[1:]), 1):
            # Calculate changes
            vv_change = vv_current - reference_vv
            ratio_change = ratio_current - reference_ratio
            
            # Construction indicators (positive changes in both VV and ratio)
            construction_mask = (vv_change > change_threshold) & (ratio_change > 2)
            
            # Remove small isolated changes
            construction_mask = morphology.remove_small_objects(construction_mask, min_size=25)
            
            # Calculate statistics
            construction_area = np.sum(construction_mask)
            total_pixels = vv_change.size
            construction_percentage = (construction_area / total_pixels) * 100
            
            # Identify construction regions
            construction_regions = measure.regionprops(measure.label(construction_mask))
            
            timeline_entry = {
                'date': date,
                'vv_change': vv_change,
                'ratio_change': ratio_change,
                'construction_mask': construction_mask,
                'construction_regions': construction_regions,
                'statistics': {
                    'construction_area': construction_area,
                    'construction_percentage': construction_percentage,
                    'num_construction_sites': len(construction_regions),
                    'mean_vv_change': np.nanmean(vv_change),
                    'mean_ratio_change': np.nanmean(ratio_change)
                }
            }
            
            construction_timeline.append(timeline_entry)
            change_maps.append(vv_change)
            
            print(f"  {date}: {construction_percentage:.2f}% construction activity")
        
        results = {
            'timeline': construction_timeline,
            'vv_series': vv_series,
            'ratio_series': ratio_series,
            'rgb_series': rgb_series,
            'change_maps': change_maps,
            'dates': valid_dates,
            'reference_date': valid_dates[0],
            'parameters': {
                'change_threshold': change_threshold,
                'time_interval_months': time_interval_months
            }
        }
        
        print(f"✓ Construction timeline analysis completed!")
        print(f"  Analyzed {len(valid_dates)} time points")
        print(f"  Reference date: {valid_dates[0]}")
        
        return results
        
    except Exception as e:
        print(f"Error in construction timeline analysis: {e}")
        return None

def plot_construction_timeline(timeline_results):
    """Plot construction timeline analysis results."""
    if not timeline_results:
        return
    
    timeline = timeline_results['timeline']
    dates = [entry['date'] for entry in timeline]
    construction_percentages = [entry['statistics']['construction_percentage'] for entry in timeline]
    num_sites = [entry['statistics']['num_construction_sites'] for entry in timeline]
    
    # Timeline overview
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Construction activity over time
    axes[0, 0].plot(dates, construction_percentages, 'o-', linewidth=2, markersize=8, color='red')
    axes[0, 0].set_ylabel('Construction Activity (%)')
    axes[0, 0].set_title('Construction Activity Timeline')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Number of construction sites
    axes[0, 1].bar(dates, num_sites, alpha=0.7, color='orange')
    axes[0, 1].set_ylabel('Number of Sites')
    axes[0, 1].set_title('Active Construction Sites')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Latest construction map
    if timeline:
        latest_entry = timeline[-1]
        im1 = axes[1, 0].imshow(latest_entry['vv_change'], cmap='RdBu_r', vmin=-3, vmax=3)
        axes[1, 0].set_title(f'Latest VV Change\n{latest_entry["date"]}')
        axes[1, 0].axis('off')
        plt.colorbar(im1, ax=axes[1, 0], shrink=0.8)
        
        axes[1, 1].imshow(latest_entry['construction_mask'], cmap='Reds')
        axes[1, 1].set_title(f'Construction Areas\n{latest_entry["statistics"]["construction_percentage"]:.2f}%')
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Detailed time series plots
    n_periods = len(timeline)
    if n_periods > 1:
        fig, axes = plt.subplots(2, min(4, n_periods), figsize=(4*min(4, n_periods), 8))
        if n_periods == 1:
            axes = axes.reshape(2, 1)
        
        for i, entry in enumerate(timeline[:4]):  # Show first 4 periods
            # VV change maps
            im1 = axes[0, i].imshow(entry['vv_change'], cmap='RdBu_r', vmin=-3, vmax=3)
            axes[0, i].set_title(f'VV Change\n{entry["date"]}')
            axes[0, i].axis('off')
            
            # Construction masks
            axes[1, i].imshow(entry['construction_mask'], cmap='Reds')
            axes[1, i].set_title(f'Construction\n{entry["statistics"]["construction_percentage"]:.1f}%')
            axes[1, i].axis('off')
        
        # Hide unused subplots
        for j in range(len(timeline), 4):
            if j < axes.shape[1]:
                axes[0, j].axis('off')
                axes[1, j].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    # Summary statistics
    print(f"\n=== CONSTRUCTION TIMELINE SUMMARY ===")
    print(f"Analysis period: {timeline_results['reference_date']} to {dates[-1]}")
    print(f"Time intervals: {len(dates)} periods")
    
    max_activity = max(construction_percentages)
    max_activity_date = dates[construction_percentages.index(max_activity)]
    print(f"Peak construction activity: {max_activity:.2f}% on {max_activity_date}")
    
    total_sites = sum(num_sites)
    print(f"Total construction sites detected: {total_sites}")
    
    # Trend analysis
    if len(construction_percentages) > 2:
        trend_slope = np.polyfit(range(len(construction_percentages)), construction_percentages, 1)[0]
        if trend_slope > 0.1:
            print("📈 Construction activity is increasing over time")
        elif trend_slope < -0.1:
            print("📉 Construction activity is decreasing over time")  
        else:
            print("📊 Construction activity is relatively stable")

# %%
# === 3. AGRICULTURAL MONITORING ===

def monitor_agricultural_cycles(
    start_date,
    end_date,
    agricultural_areas=None,
    time_interval_months=1,
    orbit_direction="DESCENDING"
):
    """
    Monitor agricultural activities and crop cycles using radar signatures.
    
    Args:
        start_date: str - Start of monitoring period
        end_date: str - End of monitoring period
        agricultural_areas: GeoDataFrame of agricultural zones (optional)
        time_interval_months: int - Months between observations
        orbit_direction: str - Satellite orbit direction
    
    Returns:
        dict with agricultural monitoring results
    """
    print("=== AGRICULTURAL MONITORING ===")
    print(f"Monitoring agricultural activities from {start_date} to {end_date}...")
    
    try:
        # Generate monthly time series
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)
        
        dates = []
        current_date = start_dt
        while current_date <= end_dt:
            dates.append(current_date.strftime('%Y-%m-%d'))
            current_date += relativedelta(months=time_interval_months)
        
        print(f"Monitoring dates: {dates}")
        
        # Fetch agricultural indicators
        vv_series = []
        vh_series = []
        ratio_series = []
        valid_dates = []
        
        for date in dates:
            try:
                print(f"Fetching agricultural data for {date}...")
                vv_data = fetch_sentinel1_product('VV', date, orbit_direction=orbit_direction)
                vh_data = fetch_sentinel1_product('VH', date, orbit_direction=orbit_direction)
                ratio_data = fetch_sentinel1_product('VV_VH_RATIO', date, orbit_direction=orbit_direction)
                
                vv_series.append(vv_data)
                vh_series.append(vh_data)
                ratio_series.append(ratio_data)
                valid_dates.append(date)
                print(f"  ✓ Success")
                
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                continue
        
        if len(vv_series) < 3:
            print("Insufficient data for agricultural monitoring")
            return None
        
        # Analyze agricultural indicators
        agricultural_timeline = []
        
        for i, (vv, vh, ratio, date) in enumerate(zip(vv_series, vh_series, ratio_series, valid_dates)):
            # Agricultural classification using polarimetric indicators
            
            # Vegetation detection (high VH, lower VV/VH ratio)
            vegetation_mask = (vh > -20) & (ratio < 8)
            
            # Bare soil detection (low VH, high VV/VH ratio)
            bare_soil_mask = (vh < -25) & (ratio > 10)
            
            # Crop fields (intermediate values, seasonal variation)
            crop_mask = (vh > -23) & (vh < -18) & (ratio > 6) & (ratio < 12)
            
            # Calculate area statistics
            total_pixels = vv.size
            vegetation_area = np.sum(vegetation_mask)
            bare_soil_area = np.sum(bare_soil_mask)
            crop_area = np.sum(crop_mask)
            
            vegetation_percentage = (vegetation_area / total_pixels) * 100
            bare_soil_percentage = (bare_soil_area / total_pixels) * 100
            crop_percentage = (crop_area / total_pixels) * 100
            
            # Seasonal indicators
            date_obj = datetime.strptime(date, '%Y-%m-%d')
            month = date_obj.month
            season = 'Spring' if 3 <= month <= 5 else 'Summer' if 6 <= month <= 8 else 'Fall' if 9 <= month <= 11 else 'Winter'
            
            timeline_entry = {
                'date': date,
                'month': month,
                'season': season,
                'vv': vv,
                'vh': vh,
                'ratio': ratio,
                'vegetation_mask': vegetation_mask,
                'bare_soil_mask': bare_soil_mask,
                'crop_mask': crop_mask,
                'statistics': {
                    'vegetation_percentage': vegetation_percentage,
                    'bare_soil_percentage': bare_soil_percentage,
                    'crop_percentage': crop_percentage,
                    'mean_vv': np.nanmean(vv),
                    'mean_vh': np.nanmean(vh),
                    'mean_ratio': np.nanmean(ratio)
                }
            }
            
            agricultural_timeline.append(timeline_entry)
            print(f"  {date} ({season}): Vegetation {vegetation_percentage:.1f}%, Crops {crop_percentage:.1f}%, Bare soil {bare_soil_percentage:.1f}%")
        
        results = {
            'timeline': agricultural_timeline,
            'dates': valid_dates,
            'parameters': {
                'time_interval_months': time_interval_months,
                'start_date': start_date,
                'end_date': end_date
            }
        }
        
        print(f"✓ Agricultural monitoring completed!")
        print(f"  Analyzed {len(valid_dates)} time points")
        
        return results
        
    except Exception as e:
        print(f"Error in agricultural monitoring: {e}")
        return None

def plot_agricultural_monitoring(agri_results):
    """Plot agricultural monitoring results."""
    if not agri_results:
        return
    
    timeline = agri_results['timeline']
    dates = [entry['date'] for entry in timeline]
    
    # Extract time series data
    vegetation_percentages = [entry['statistics']['vegetation_percentage'] for entry in timeline]
    crop_percentages = [entry['statistics']['crop_percentage'] for entry in timeline]
    bare_soil_percentages = [entry['statistics']['bare_soil_percentage'] for entry in timeline]
    months = [entry['month'] for entry in timeline]
    
    # Time series plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Land cover time series
    axes[0, 0].plot(dates, vegetation_percentages, 'o-', color='green', label='Vegetation', linewidth=2)
    axes[0, 0].plot(dates, crop_percentages, 's-', color='orange', label='Crops', linewidth=2)
    axes[0, 0].plot(dates, bare_soil_percentages, '^-', color='brown', label='Bare Soil', linewidth=2)
    axes[0, 0].set_ylabel('Area Coverage (%)')
    axes[0, 0].set_title('Agricultural Land Cover Timeline')
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Seasonal patterns
    season_data = {}
    for entry in timeline:
        season = entry['season']
        if season not in season_data:
            season_data[season] = {'vegetation': [], 'crops': [], 'bare_soil': []}
        season_data[season]['vegetation'].append(entry['statistics']['vegetation_percentage'])
        season_data[season]['crops'].append(entry['statistics']['crop_percentage'])
        season_data[season]['bare_soil'].append(entry['statistics']['bare_soil_percentage'])
    
    seasons = ['Spring', 'Summer', 'Fall', 'Winter']
    season_colors = ['lightgreen', 'gold', 'orange', 'lightblue']
    
    for i, (land_type, color_base) in enumerate([('vegetation', 'green'), ('crops', 'orange'), ('bare_soil', 'brown')]):
        season_means = []
        season_labels = []
        for season in seasons:
            if season in season_data and season_data[season][land_type]:
                season_means.append(np.mean(season_data[season][land_type]))
                season_labels.append(season)
        
        if season_means:
            x_pos = [j + i*0.25 for j in range(len(season_labels))]
            axes[0, 1].bar(x_pos, season_means, width=0.2, label=land_type.title(), 
                          color=color_base, alpha=0.7)
    
    axes[0, 1].set_ylabel('Average Coverage (%)')
    axes[0, 1].set_title('Seasonal Land Cover Patterns')
    axes[0, 1].set_xticks(range(len(seasons)))
    axes[0, 1].set_xticklabels(seasons)
    axes[0, 1].legend()
    
    # Latest land cover map
    if timeline:
        latest_entry = timeline[-1]
        
        # Create RGB land cover map
        land_cover_rgb = np.zeros((*latest_entry['vv'].shape, 3))
        land_cover_rgb[latest_entry['vegetation_mask']] = [0, 1, 0]  # Green for vegetation
        land_cover_rgb[latest_entry['crop_mask']] = [1, 0.65, 0]     # Orange for crops
        land_cover_rgb[latest_entry['bare_soil_mask']] = [0.6, 0.3, 0]  # Brown for bare soil
        
        axes[1, 0].imshow(land_cover_rgb)
        axes[1, 0].set_title(f'Land Cover Classification\n{latest_entry["date"]} ({latest_entry["season"]})')
        axes[1, 0].axis('off')
        
        # Radar signature plot (VV vs VH)
        vv_flat = latest_entry['vv'].flatten()
        vh_flat = latest_entry['vh'].flatten()
        
        # Sample for plotting (to avoid too many points)
        n_samples = min(5000, len(vv_flat))
        indices = np.random.choice(len(vv_flat), n_samples, replace=False)
        
        scatter = axes[1, 1].scatter(vv_flat[indices], vh_flat[indices], 
                                   c=latest_entry['ratio'].flatten()[indices], 
                                   cmap='viridis', alpha=0.6, s=1)
        axes[1, 1].set_xlabel('VV Backscatter (dB)')
        axes[1, 1].set_ylabel('VH Backscatter (dB)')
        axes[1, 1].set_title('Radar Signature Distribution')
        plt.colorbar(scatter, ax=axes[1, 1], label='VV/VH Ratio (dB)')
    
    plt.tight_layout()
    plt.show()
    
    # Agricultural cycle analysis
    print(f"\n=== AGRICULTURAL CYCLE ANALYSIS ===")
    print(f"Monitoring period: {agri_results['parameters']['start_date']} to {agri_results['parameters']['end_date']}")
    
    # Seasonal statistics
    for season in seasons:
        if season in season_data:
            veg_mean = np.mean(season_data[season]['vegetation']) if season_data[season]['vegetation'] else 0
            crop_mean = np.mean(season_data[season]['crops']) if season_data[season]['crops'] else 0
            soil_mean = np.mean(season_data[season]['bare_soil']) if season_data[season]['bare_soil'] else 0
            
            print(f"\n{season}:")
            print(f"  Vegetation: {veg_mean:.1f}%")
            print(f"  Crops: {crop_mean:.1f}%") 
            print(f"  Bare Soil: {soil_mean:.1f}%")
    
    # Agricultural insights
    print(f"\n=== AGRICULTURAL INSIGHTS ===")
    
    max_crop_idx = crop_percentages.index(max(crop_percentages))
    max_crop_date = dates[max_crop_idx]
    max_crop_month = months[max_crop_idx]
    
    print(f"🌾 Peak crop activity: {max(crop_percentages):.1f}% in {max_crop_date} (month {max_crop_month})")
    
    if max_crop_month in [6, 7, 8]:
        print("   ✓ Peak crop activity in summer - normal growing season")
    elif max_crop_month in [4, 5, 9, 10]:
        print("   • Peak crop activity in spring/fall - possible winter crops or harvesting")
    
    # Variability analysis
    crop_std = np.std(crop_percentages)
    if crop_std > 5:
        print(f"📊 High agricultural variability (σ={crop_std:.1f}%) - active farming region")
    else:
        print(f"📊 Low agricultural variability (σ={crop_std:.1f}%) - stable land use")

# %%
# === 4. ICE DETECTION ON CANALS ===

def detect_ice_on_canals(
    winter_dates,
    baseline_dates=None,
    ice_threshold_vv=-12,
    ice_threshold_ratio=15,
    min_ice_area=20,
    orbit_direction="DESCENDING"
):
    """
    Detect ice formation on canals and water bodies during winter.
    
    Args:
        winter_dates: list of dates during potential ice formation
        baseline_dates: list of dates for ice-free reference (optional)
        ice_threshold_vv: dB threshold for ice detection (higher than water)
        ice_threshold_ratio: VV/VH ratio threshold for ice
        min_ice_area: minimum connected area for valid ice detection
        orbit_direction: str - Satellite orbit direction
    
    Returns:
        dict with ice detection results
    """
    print("=== ICE DETECTION ON CANALS ===")
    print(f"Detecting ice formation during winter months...")
    
    try:
        # Load surface water areas if available
        try:
            water_bodies = gpd.read_file(SURFACE_WATER_PATH)
            if water_bodies.crs != 'EPSG:4326':
                water_bodies = water_bodies.to_crs('EPSG:4326')
            print("✓ Using surface water polygons for targeted ice detection")
        except:
            water_bodies = None
            print("• Using general ice detection (no water polygon available)")
        
        # Get baseline (ice-free) conditions if provided
        baseline_vv = None
        baseline_ratio = None
        
        if baseline_dates:
            print("Fetching baseline (ice-free) conditions...")
            baseline_vv_images = []
            baseline_ratio_images = []
            
            for date in baseline_dates:
                try:
                    vv_data = fetch_sentinel1_product('VV', date, orbit_direction=orbit_direction)
                    ratio_data = fetch_sentinel1_product('VV_VH_RATIO', date, orbit_direction=orbit_direction)
                    baseline_vv_images.append(vv_data)
                    baseline_ratio_images.append(ratio_data)
                    print(f"  ✓ Baseline data for {date}")
                except Exception as e:
                    print(f"  ✗ Failed baseline for {date}: {e}")
            
            if baseline_vv_images:
                baseline_vv = np.mean(baseline_vv_images, axis=0)
                baseline_ratio = np.mean(baseline_ratio_images, axis=0)
        
        # Analyze ice conditions for each winter date
        ice_timeline = []
        
        for date in winter_dates:
            try:
                print(f"Analyzing ice conditions for {date}...")
                
                vv_data = fetch_sentinel1_product('VV', date, orbit_direction=orbit_direction)
                ratio_data = fetch_sentinel1_product('VV_VH_RATIO', date, orbit_direction=orbit_direction)
                
                # Ice detection using radar signatures
                # Ice typically has higher backscatter than water and higher VV/VH ratio
                ice_signature = (vv_data > ice_threshold_vv) & (ratio_data > ice_threshold_ratio)
                
                # If we have water body polygons, focus detection there
                if water_bodies is not None:
                    bounds = get_bounds_from_geometry(geom)
                    water_mask = create_water_mask(vv_data.shape, bounds, water_bodies)
                    ice_signature = ice_signature & water_mask
                
                # Remove small isolated areas
                if min_ice_area > 1:
                    ice_signature = morphology.remove_small_objects(ice_signature, min_size=min_ice_area)
                
                # Calculate ice extent
                ice_area = np.sum(ice_signature)
                total_pixels = vv_data.size
                ice_percentage = (ice_area / total_pixels) * 100
                
                # If we have water bodies, calculate ice coverage within water areas
                ice_in_water_percentage = 0
                if water_bodies is not None:
                    water_pixels = np.sum(water_mask)
                    ice_in_water = np.sum(ice_signature & water_mask)
                    ice_in_water_percentage = (ice_in_water / max(water_pixels, 1)) * 100
                
                # Compare with baseline if available
                vv_change = None
                ratio_change = None
                if baseline_vv is not None:
                    vv_change = vv_data - baseline_vv
                    ratio_change = ratio_data - baseline_ratio
                
                # Identify ice regions
                ice_regions = measure.regionprops(measure.label(ice_signature))
                
                # Temperature indicator (estimate from radar signature changes)
                if baseline_vv is not None:
                    temp_indicator = np.nanmean(vv_change)  # Positive change suggests freezing
                else:
                    temp_indicator = np.nanmean(vv_data)  # Absolute backscatter level
                
                timeline_entry = {
                    'date': date,
                    'vv': vv_data,
                    'ratio': ratio_data,
                    'ice_signature': ice_signature,
                    'ice_regions': ice_regions,
                    'vv_change': vv_change,
                    'ratio_change': ratio_change,
                    'statistics': {
                        'ice_area': ice_area,
                        'ice_percentage': ice_percentage,
                        'ice_in_water_percentage': ice_in_water_percentage,
                        'num_ice_regions': len(ice_regions),
                        'mean_vv': np.nanmean(vv_data),
                        'mean_ratio': np.nanmean(ratio_data),
                        'temperature_indicator': temp_indicator
                    }
                }
                
                ice_timeline.append(timeline_entry)
                print(f"  ✓ Ice coverage: {ice_percentage:.2f}% total, {ice_in_water_percentage:.1f}% in water bodies")
                
            except Exception as e:
                print(f"  ✗ Failed for {date}: {e}")
                continue
        
        if not ice_timeline:
            print("No valid ice detection data")
            return None
        
        results = {
            'timeline': ice_timeline,
            'baseline_vv': baseline_vv,
            'baseline_ratio': baseline_ratio,
            'water_bodies': water_bodies,
            'dates': winter_dates,
            'parameters': {
                'ice_threshold_vv': ice_threshold_vv,
                'ice_threshold_ratio': ice_threshold_ratio,
                'min_ice_area': min_ice_area
            }
        }
        
        print(f"✓ Ice detection analysis completed!")
        print(f"  Analyzed {len(ice_timeline)} winter dates")
        
        return results
        
    except Exception as e:
        print(f"Error in ice detection: {e}")
        return None

def plot_ice_detection(ice_results):
    """Plot ice detection analysis results."""
    if not ice_results:
        return
    
    timeline = ice_results['timeline']
    dates = [entry['date'] for entry in timeline]
    ice_percentages = [entry['statistics']['ice_percentage'] for entry in timeline]
    ice_in_water_percentages = [entry['statistics']['ice_in_water_percentage'] for entry in timeline]
    temp_indicators = [entry['statistics']['temperature_indicator'] for entry in timeline]
    
    # Time series analysis
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Ice coverage over time
    axes[0, 0].plot(dates, ice_percentages, 'o-', color='cyan', linewidth=2, markersize=8, label='Total Area')
    if ice_results['water_bodies'] is not None:
        axes[0, 0].plot(dates, ice_in_water_percentages, 's-', color='blue', linewidth=2, markersize=6, label='Water Bodies')
    axes[0, 0].set_ylabel('Ice Coverage (%)')
    axes[0, 0].set_title('Ice Formation Timeline')
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Temperature indicator
    axes[0, 1].plot(dates, temp_indicators, 'o-', color='red', linewidth=2, markersize=8)
    axes[0, 1].set_ylabel('Temperature Indicator (dB)')
    axes[0, 1].set_title('Radar-derived Temperature Indicator')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Freezing threshold')
    
    # Latest ice map
    if timeline:
        latest_entry = timeline[-1]
        
        # Ice detection map
        axes[1, 0].imshow(latest_entry['ice_signature'], cmap='Blues')
        axes[1, 0].set_title(f'Ice Detection\n{latest_entry["date"]} ({latest_entry["statistics"]["ice_percentage"]:.1f}%)')
        axes[1, 0].axis('off')
        
        # VV backscatter with ice overlay
        ice_overlay = np.zeros((*latest_entry['vv'].shape, 3))
        # Background: VV backscatter in grayscale
        vv_norm = (latest_entry['vv'] + 25) / 25  # Normalize VV
        vv_norm = np.clip(vv_norm, 0, 1)
        ice_overlay[:, :, 0] = vv_norm * 0.7
        ice_overlay[:, :, 1] = vv_norm * 0.7
        ice_overlay[:, :, 2] = vv_norm * 0.7
        
        # Ice areas in blue
        ice_overlay[latest_entry['ice_signature']] = [0, 0.7, 1]
        
        axes[1, 1].imshow(ice_overlay)
        axes[1, 1].set_title('Ice Overlay on Radar Image')
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Detailed ice maps for multiple dates
    n_dates = len(timeline)
    if n_dates > 1:
        n_cols = min(4, n_dates)
        fig, axes = plt.subplots(2, n_cols, figsize=(4*n_cols, 8))
        if n_dates == 1:
            axes = axes.reshape(2, 1)
        
        for i, entry in enumerate(timeline[:n_cols]):
            # VV backscatter
            im1 = axes[0, i].imshow(entry['vv'], cmap='viridis', vmin=-25, vmax=0)
            axes[0, i].set_title(f'VV {entry["date"]}')
            axes[0, i].axis('off')
            
            # Ice detection
            axes[1, i].imshow(entry['ice_signature'], cmap='Blues')
            axes[1, i].set_title(f'Ice {entry["statistics"]["ice_percentage"]:.1f}%')
            axes[1, i].axis('off')
        
        # Hide unused subplots
        for j in range(n_dates, n_cols):
            axes[0, j].axis('off')
            axes[1, j].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    # Ice detection summary
    print(f"\n=== ICE DETECTION SUMMARY ===")
    print(f"Analysis period: {dates[0]} to {dates[-1]}")
    print(f"Detection thresholds: VV > {ice_results['parameters']['ice_threshold_vv']} dB, Ratio > {ice_results['parameters']['ice_threshold_ratio']} dB")
    
    max_ice_idx = ice_percentages.index(max(ice_percentages))
    max_ice_date = dates[max_ice_idx]
    max_ice_percentage = ice_percentages[max_ice_idx]
    
    print(f"\nIce Formation:")
    print(f"  Peak ice coverage: {max_ice_percentage:.2f}% on {max_ice_date}")
    
    if ice_results['water_bodies'] is not None:
        max_water_ice = max(ice_in_water_percentages)
        print(f"  Peak ice in water bodies: {max_water_ice:.1f}%")
    
    # Ice formation analysis
    if max_ice_percentage > 5:
        print("❄️  Significant ice formation detected")
        if max_ice_percentage > 20:
            print("   🧊 Severe freezing conditions")
            print("   • Canal navigation may be affected")
            print("   • Monitor ice thickness for safety")
        else:
            print("   ❄️  Moderate ice formation")
            print("   • Light freezing conditions")
    else:
        print("✅ Minimal ice formation detected")
        print("   • Canals likely remain navigable")
    
    # Trend analysis
    if len(ice_percentages) > 2:
        ice_trend = np.polyfit(range(len(ice_percentages)), ice_percentages, 1)[0]
        if ice_trend > 1:
            print("📈 Ice coverage is increasing - strengthening freeze")
        elif ice_trend < -1:
            print("📉 Ice coverage is decreasing - thawing conditions")
        else:
            print("📊 Stable ice conditions")

# %%
# === 5. ADVANCED TIME SERIES ANALYSIS FOR INFRASTRUCTURE ===

def advanced_infrastructure_monitoring(
    start_date,
    end_date,
    infrastructure_areas=None,
    analysis_interval_days=12,  # Sentinel-1 repeat cycle
    coherence_threshold=0.4,
    displacement_threshold=0.5,  # dB change threshold
    orbit_direction="DESCENDING"
):
    """
    Advanced time series analysis for infrastructure monitoring using coherence and phase.
    
    Args:
        start_date: str - Start of monitoring period
        end_date: str - End of monitoring period
        infrastructure_areas: GeoDataFrame of infrastructure polygons (optional)
        analysis_interval_days: int - Days between analysis points
        coherence_threshold: float - Minimum coherence for reliable measurements
        displacement_threshold: float - Threshold for significant changes
        orbit_direction: str - Satellite orbit direction
    
    Returns:
        dict with comprehensive infrastructure analysis
    """
    print("=== ADVANCED INFRASTRUCTURE MONITORING ===")
    print(f"Time series analysis from {start_date} to {end_date}...")
    
    try:
        # Generate high-frequency time series (every Sentinel-1 cycle)
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)
        
        dates = []
        current_date = start_dt
        while current_date <= end_dt:
            dates.append(current_date.strftime('%Y-%m-%d'))
            current_date += timedelta(days=analysis_interval_days)
        
        print(f"Analysis dates: {len(dates)} acquisitions")
        
        # Fetch time series data
        vv_series = []
        vh_series = []
        ratio_series = []
        valid_dates = []
        
        for date in dates:
            try:
                print(f"Fetching infrastructure data for {date}...", end=' ')
                vv_data = fetch_sentinel1_product('VV', date, orbit_direction=orbit_direction)
                vh_data = fetch_sentinel1_product('VH', date, orbit_direction=orbit_direction)
                ratio_data = fetch_sentinel1_product('VV_VH_RATIO', date, orbit_direction=orbit_direction)
                
                vv_series.append(vv_data)
                vh_series.append(vh_data)
                ratio_series.append(ratio_data)
                valid_dates.append(date)
                print("✓")
                
            except Exception as e:
                print(f"✗ ({e})")
                continue
        
        if len(vv_series) < 4:
            print("Insufficient data for time series analysis")
            return None
        
        print(f"Successfully acquired {len(valid_dates)} time series points")
        
        # Advanced time series analysis
        reference_vv = vv_series[0]
        reference_vh = vh_series[0]
        
        # Calculate time series metrics
        time_series_analysis = []
        displacement_maps = []
        coherence_maps = []
        
        for i, (vv, vh, ratio, date) in enumerate(zip(vv_series[1:], vh_series[1:], ratio_series[1:], valid_dates[1:]), 1):
            # Phase change analysis (proxy for displacement)
            vv_change = vv - reference_vv
            vh_change = vh - reference_vh
            
            # Coherence estimation (stability indicator)
            # Simple coherence approximation using correlation
            coherence_estimate = np.abs(vv + 1j * vh) / (np.abs(reference_vv + 1j * reference_vh) + 0.001)
            coherence_estimate = np.clip(coherence_estimate, 0, 1)
            
            # Infrastructure stability analysis
            high_coherence_mask = coherence_estimate > coherence_threshold
            significant_change_mask = (np.abs(vv_change) > displacement_threshold) & high_coherence_mask
            
            # Calculate displacement indicators
            mean_displacement = np.nanmean(vv_change[high_coherence_mask])
            std_displacement = np.nanstd(vv_change[high_coherence_mask])
            max_displacement = np.nanmax(np.abs(vv_change[high_coherence_mask]))
            
            # Infrastructure health indicators
            coherent_pixels = np.sum(high_coherence_mask)
            total_pixels = vv.size
            coherence_percentage = (coherent_pixels / total_pixels) * 100
            
            unstable_pixels = np.sum(significant_change_mask)
            instability_percentage = (unstable_pixels / max(coherent_pixels, 1)) * 100
            
            # Temporal correlation analysis
            temporal_correlation = np.corrcoef(reference_vv.flatten(), vv.flatten())[0, 1]
            
            # Identify regions of concern
            unstable_regions = measure.regionprops(measure.label(significant_change_mask))
            
            analysis_entry = {
                'date': date,
                'days_from_reference': (datetime.strptime(date, '%Y-%m-%d') - datetime.strptime(valid_dates[0], '%Y-%m-%d')).days,
                'vv': vv,
                'vh': vh,
                'vv_change': vv_change,
                'vh_change': vh_change,
                'coherence_estimate': coherence_estimate,
                'high_coherence_mask': high_coherence_mask,
                'significant_change_mask': significant_change_mask,
                'unstable_regions': unstable_regions,
                'statistics': {
                    'mean_displacement': mean_displacement,
                    'std_displacement': std_displacement,
                    'max_displacement': max_displacement,
                    'coherence_percentage': coherence_percentage,
                    'instability_percentage': instability_percentage,
                    'temporal_correlation': temporal_correlation,
                    'num_unstable_regions': len(unstable_regions),
                    'coherent_pixels': coherent_pixels
                }
            }
            
            time_series_analysis.append(analysis_entry)
            displacement_maps.append(vv_change)
            coherence_maps.append(coherence_estimate)
            
            print(f"  {date}: Coherence {coherence_percentage:.1f}%, Instability {instability_percentage:.2f}%")
        
        # Long-term trend analysis
        days_series = [entry['days_from_reference'] for entry in time_series_analysis]
        displacement_series = [entry['statistics']['mean_displacement'] for entry in time_series_analysis]
        coherence_series = [entry['statistics']['coherence_percentage'] for entry in time_series_analysis]
        
        # Calculate trends
        if len(displacement_series) > 3:
            displacement_trend = np.polyfit(days_series, displacement_series, 1)[0]  # dB/day
            coherence_trend = np.polyfit(days_series, coherence_series, 1)[0]  # %/day
        else:
            displacement_trend = 0
            coherence_trend = 0
        
        # Infrastructure health assessment
        recent_instability = np.mean([entry['statistics']['instability_percentage'] for entry in time_series_analysis[-3:]])
        recent_coherence = np.mean([entry['statistics']['coherence_percentage'] for entry in time_series_analysis[-3:]])
        
        # Critical infrastructure zones (persistent instability)
        critical_zones = np.zeros_like(reference_vv, dtype=bool)
        for entry in time_series_analysis[-5:]:  # Last 5 measurements
            critical_zones |= entry['significant_change_mask']
        
        results = {
            'time_series_analysis': time_series_analysis,
            'displacement_maps': displacement_maps,
            'coherence_maps': coherence_maps,
            'critical_zones': critical_zones,
            'reference_date': valid_dates[0],
            'valid_dates': valid_dates,
            'trends': {
                'displacement_trend': displacement_trend,  # dB/day
                'coherence_trend': coherence_trend,  # %/day
                'displacement_trend_annual': displacement_trend * 365,  # dB/year
                'coherence_trend_annual': coherence_trend * 365  # %/year
            },
            'health_assessment': {
                'recent_instability': recent_instability,
                'recent_coherence': recent_coherence,
                'overall_stability': 'STABLE' if recent_instability < 5 else 'UNSTABLE' if recent_instability > 15 else 'MONITORING',
                'data_quality': 'EXCELLENT' if recent_coherence > 70 else 'GOOD' if recent_coherence > 50 else 'POOR'
            },
            'parameters': {
                'coherence_threshold': coherence_threshold,
                'displacement_threshold': displacement_threshold,
                'analysis_interval_days': analysis_interval_days
            }
        }
        
        print(f"✓ Infrastructure monitoring completed!")
        print(f"  Time series: {len(valid_dates)} points over {(datetime.strptime(valid_dates[-1], '%Y-%m-%d') - datetime.strptime(valid_dates[0], '%Y-%m-%d')).days} days")
        print(f"  Overall stability: {results['health_assessment']['overall_stability']}")
        print(f"  Data quality: {results['health_assessment']['data_quality']}")
        
        return results
        
    except Exception as e:
        print(f"Error in infrastructure monitoring: {e}")
        return None

def plot_infrastructure_monitoring(infra_results):
    """Plot comprehensive infrastructure monitoring results."""
    if not infra_results:
        return
    
    time_series = infra_results['time_series_analysis']
    dates = [entry['date'] for entry in time_series]
    days = [entry['days_from_reference'] for entry in time_series]
    
    # Time series metrics
    displacement_means = [entry['statistics']['mean_displacement'] for entry in time_series]
    displacement_stds = [entry['statistics']['std_displacement'] for entry in time_series]
    coherence_percentages = [entry['statistics']['coherence_percentage'] for entry in time_series]
    instability_percentages = [entry['statistics']['instability_percentage'] for entry in time_series]
    correlations = [entry['statistics']['temporal_correlation'] for entry in time_series]
    
    # Create comprehensive dashboard
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    
    # 1. Displacement time series
    axes[0, 0].errorbar(days, displacement_means, yerr=displacement_stds, fmt='o-', capsize=3, linewidth=2)
    axes[0, 0].set_xlabel('Days from Reference')
    axes[0, 0].set_ylabel('Mean Displacement (dB)')
    axes[0, 0].set_title('Infrastructure Displacement Timeline')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # Add trend line
    if len(days) > 3:
        trend_line = np.poly1d(np.polyfit(days, displacement_means, 1))
        axes[0, 0].plot(days, trend_line(days), 'r--', alpha=0.7, 
                       label=f'Trend: {infra_results["trends"]["displacement_trend"]*365:.3f} dB/year')
        axes[0, 0].legend()
    
    # 2. Coherence quality over time
    axes[0, 1].plot(days, coherence_percentages, 'o-', color='green', linewidth=2, markersize=6)
    axes[0, 1].set_xlabel('Days from Reference')
    axes[0, 1].set_ylabel('Coherence (%)')
    axes[0, 1].set_title('Data Quality (Coherence) Timeline')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=infra_results['parameters']['coherence_threshold']*100, 
                      color='red', linestyle='--', alpha=0.5, label='Quality Threshold')
    axes[0, 1].legend()
    
    # 3. Instability percentage
    axes[0, 2].plot(days, instability_percentages, 'o-', color='red', linewidth=2, markersize=6)
    axes[0, 2].set_xlabel('Days from Reference')
    axes[0, 2].set_ylabel('Instability (%)')
    axes[0, 2].set_title('Infrastructure Instability Timeline')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Latest displacement map
    if time_series:
        latest_entry = time_series[-1]
        im1 = axes[1, 0].imshow(latest_entry['vv_change'], cmap='RdBu_r', vmin=-2, vmax=2)
        axes[1, 0].set_title(f'Latest Displacement Map\n{latest_entry["date"]}')
        axes[1, 0].axis('off')
        plt.colorbar(im1, ax=axes[1, 0], shrink=0.8, label='Displacement (dB)')
        
        # 5. Latest coherence map
        im2 = axes[1, 1].imshow(latest_entry['coherence_estimate'], cmap='viridis', vmin=0, vmax=1)
        axes[1, 1].set_title(f'Latest Coherence Map\n{latest_entry["date"]}')
        axes[1, 1].axis('off')
        plt.colorbar(im2, ax=axes[1, 1], shrink=0.8, label='Coherence')
        
        # 6. Instability map
        axes[1, 2].imshow(latest_entry['significant_change_mask'], cmap='Reds')
        axes[1, 2].set_title(f'Instability Areas\n{latest_entry["statistics"]["instability_percentage"]:.2f}%')
        axes[1, 2].axis('off')
    
    # 7. Critical infrastructure zones
    axes[2, 0].imshow(infra_results['critical_zones'], cmap='Reds')
    axes[2, 0].set_title('Critical Infrastructure Zones\n(Persistent Instability)')
    axes[2, 0].axis('off')
    
    # 8. Correlation analysis
    axes[2, 1].plot(days, correlations, 'o-', color='purple', linewidth=2, markersize=6)
    axes[2, 1].set_xlabel('Days from Reference')
    axes[2, 1].set_ylabel('Temporal Correlation')
    axes[2, 1].set_title('Image Correlation Over Time')
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Good Correlation')
    axes[2, 1].legend()
    
    # 9. Infrastructure health summary
    health = infra_results['health_assessment']
    trends = infra_results['trends']
    
    # Create health status visualization
    status_colors = {'STABLE': 'green', 'MONITORING': 'orange', 'UNSTABLE': 'red'}
    quality_colors = {'EXCELLENT': 'green', 'GOOD': 'orange', 'POOR': 'red'}
    
    axes[2, 2].text(0.1, 0.8, 'INFRASTRUCTURE HEALTH', fontsize=14, fontweight='bold')
    axes[2, 2].text(0.1, 0.7, f'Status: {health["overall_stability"]}', 
                   fontsize=12, color=status_colors.get(health["overall_stability"], 'black'))
    axes[2, 2].text(0.1, 0.6, f'Data Quality: {health["data_quality"]}', 
                   fontsize=12, color=quality_colors.get(health["data_quality"], 'black'))
    axes[2, 2].text(0.1, 0.5, f'Recent Instability: {health["recent_instability"]:.1f}%', fontsize=10)
    axes[2, 2].text(0.1, 0.4, f'Recent Coherence: {health["recent_coherence"]:.1f}%', fontsize=10)
    axes[2, 2].text(0.1, 0.3, f'Displacement Trend: {trends["displacement_trend_annual"]:.3f} dB/year', fontsize=10)
    axes[2, 2].text(0.1, 0.2, f'Coherence Trend: {trends["coherence_trend_annual"]:.1f} %/year', fontsize=10)
    axes[2, 2].set_xlim(0, 1)
    axes[2, 2].set_ylim(0, 1)
    axes[2, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Displacement evolution animation-style plot
    n_maps = min(6, len(infra_results['displacement_maps']))
    if n_maps > 1:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i in range(n_maps):
            entry = time_series[-(n_maps-i)]  # Show recent maps
            im = axes[i].imshow(entry['vv_change'], cmap='RdBu_r', vmin=-2, vmax=2)
            axes[i].set_title(f'{entry["date"]}\n{entry["statistics"]["instability_percentage"]:.1f}% unstable')
            axes[i].axis('off')
        
        # Hide unused subplots
        for j in range(n_maps, 6):
            axes[j].axis('off')
        
        plt.suptitle('Infrastructure Displacement Evolution', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    # Detailed analysis report
    print(f"\n=== INFRASTRUCTURE MONITORING REPORT ===")
    print(f"Monitoring period: {infra_results['reference_date']} to {dates[-1]}")
    print(f"Analysis interval: {infra_results['parameters']['analysis_interval_days']} days")
    print(f"Total time series points: {len(dates)}")
    
    print(f"\n--- STABILITY ASSESSMENT ---")
    print(f"Overall Status: {health['overall_stability']}")
    print(f"Data Quality: {health['data_quality']}")
    print(f"Recent Instability: {health['recent_instability']:.2f}%")
    print(f"Recent Coherence: {health['recent_coherence']:.1f}%")
    
    print(f"\n--- TREND ANALYSIS ---")
    print(f"Displacement Trend: {trends['displacement_trend_annual']:.4f} dB/year")
    print(f"Coherence Trend: {trends['coherence_trend_annual']:.2f} %/year")
    
    if abs(trends['displacement_trend_annual']) > 0.1:
        if trends['displacement_trend_annual'] > 0:
            print("⚠️  POSITIVE DISPLACEMENT TREND - Infrastructure may be experiencing systematic changes")
        else:
            print("⚠️  NEGATIVE DISPLACEMENT TREND - Potential subsidence or structural settling")
    else:
        print("✅ Stable displacement trend - no systematic movement detected")
    
    if trends['coherence_trend_annual'] < -5:
        print("📉 DECLINING DATA QUALITY - Infrastructure may be becoming less stable")
    elif trends['coherence_trend_annual'] > 5:
        print("📈 IMPROVING DATA QUALITY - Infrastructure stability increasing")
    else:
        print("📊 Stable data quality over time")
    
    print(f"\n--- CRITICAL ZONES ---")
    critical_area = np.sum(infra_results['critical_zones'])
    total_area = infra_results['critical_zones'].size
    critical_percentage = (critical_area / total_area) * 100
    print(f"Critical infrastructure zones: {critical_percentage:.2f}% of area")
    
    if critical_percentage > 5:
        print("🚨 SIGNIFICANT CRITICAL ZONES DETECTED")
        print("   Recommendations:")
        print("   • Conduct detailed ground survey of critical zones")
        print("   • Increase monitoring frequency for unstable areas")
        print("   • Consider structural health monitoring systems")
    elif critical_percentage > 1:
        print("⚠️  Minor critical zones detected - continue monitoring")
    else:
        print("✅ No significant critical zones - infrastructure appears stable")
    
    print(f"\n--- RECOMMENDATIONS ---")
    if health['overall_stability'] == 'UNSTABLE':
        print("🔴 IMMEDIATE ACTION REQUIRED")
        print("   • Emergency structural inspection")
        print("   • Implement continuous monitoring")
        print("   • Restrict access if necessary")
    elif health['overall_stability'] == 'MONITORING':
        print("🟡 ENHANCED MONITORING RECOMMENDED")
        print("   • Increase inspection frequency")
        print("   • Monitor trend development")
        print("   • Prepare contingency plans")
    else:
        print("🟢 CONTINUE ROUTINE MONITORING")
        print("   • Maintain current monitoring schedule")
        print("   • Annual structural assessments")

# %%
# === EXAMPLE USAGE AND TESTING ===

print("=== ADVANCED RADAR APPLICATIONS FOR ALKMAAR ===")
print("Testing comprehensive radar monitoring capabilities...\n")

# Test 1: Flood Detection
print("1. TESTING FLOOD DETECTION...")
try:
    flood_results = detect_flood_extent(
        baseline_dates=['2024-07-01', '2024-08-01'],  # Summer baseline
        flood_dates=['2024-01-15', '2024-02-15'],     # Winter potential flooding
        water_threshold_vv=-18,
        water_threshold_vh=-22,
        min_flood_area=30
    )
    
    if flood_results:
        plot_flood_analysis(flood_results)
    else:
        print("⚠️  No flood data available for analysis")
        
except Exception as e:
    print(f"✗ Error in flood detection: {e}")

print("\n" + "="*60 + "\n")

# Test 2: Construction Timeline
print("2. TESTING CONSTRUCTION TIMELINE...")
try:
    construction_results = analyze_construction_timeline(
        start_date='2024-01-01',
        end_date='2024-08-01',
        time_interval_months=2,
        change_threshold=1.2
    )
    
    if construction_results:
        plot_construction_timeline(construction_results)
    else:
        print("⚠️  Insufficient data for construction timeline")
        
except Exception as e:
    print(f"✗ Error in construction timeline: {e}")

print("\n" + "="*60 + "\n")

# Test 3: Agricultural Monitoring
print("3. TESTING AGRICULTURAL MONITORING...")
try:
    agricultural_results = monitor_agricultural_cycles(
        start_date='2024-03-01',
        end_date='2024-08-01',
        time_interval_months=1
    )
    
    if agricultural_results:
        plot_agricultural_monitoring(agricultural_results)
    else:
        print("⚠️  Insufficient data for agricultural monitoring")
        
except Exception as e:
    print(f"✗ Error in agricultural monitoring: {e}")

print("\n" + "="*60 + "\n")

# Test 4: Ice Detection
print("4. TESTING ICE DETECTION...")
try:
    ice_results = detect_ice_on_canals(
        winter_dates=['2024-01-15', '2024-02-01', '2024-02-15'],
        baseline_dates=['2024-07-01', '2024-08-01'],  # Summer baseline
        ice_threshold_vv=-12,
        ice_threshold_ratio=15
    )
    
    if ice_results:
        plot_ice_detection(ice_results)
    else:
        print("⚠️  No ice detection data available")
        
except Exception as e:
    print(f"✗ Error in ice detection: {e}")

print("\n" + "="*60 + "\n")

# Test 5: Infrastructure Time Series
print("5. TESTING ADVANCED INFRASTRUCTURE MONITORING...")
try:
    infrastructure_results = advanced_infrastructure_monitoring(
        start_date='2024-03-01',
        end_date='2024-08-01',
        analysis_interval_days=12,  # Every Sentinel-1 cycle
        coherence_threshold=0.4,
        displacement_threshold=0.8
    )
    
    if infrastructure_results:
        plot_infrastructure_monitoring(infrastructure_results)
    else:
        print("⚠️  Insufficient data for infrastructure monitoring")
        
except Exception as e:
    print(f"✗ Error in infrastructure monitoring: {e}")

print("\n" + "="*80)
print("=== ADVANCED RADAR APPLICATIONS COMPLETE ===")
print("="*80)

print(f"\n🎯 SUMMARY OF CAPABILITIES:")
print(f"✅ Flood Detection - Monitor water management systems")
print(f"✅ Construction Timeline - Track urban development") 
print(f"✅ Agricultural Monitoring - Analyze crop cycles")
print(f"✅ Ice Detection - Winter canal monitoring")
print(f"✅ Infrastructure Health - Advanced time series analysis")

print(f"\n📊 APPLICATIONS FOR ALKMAAR:")
print(f"• Polder management and flood risk assessment")
print(f"• Urban planning and construction oversight")
print(f"• Agricultural productivity monitoring")
print(f"• Winter navigation safety on canals")
print(f"• Bridge and infrastructure health monitoring")
print(f"• Real-time environmental monitoring dashboard")

print(f"\n🔧 NEXT STEPS:")
print(f"• Integrate with weather data for improved accuracy")
print(f"• Set up automated alert systems for critical changes")
print(f"• Combine with Sentinel-2 optical data")
print(f"• Create operational monitoring dashboard")
print(f"• Validate results with ground truth measurements")