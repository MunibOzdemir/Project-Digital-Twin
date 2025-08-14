# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: satellite_nso_test_env
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import os
import json
from tools import *
from illegal import *
from radar_copernicus import fetch_sentinel1_product, get_token

# %%
# Fetch the geojson file for the region of interest
path_geojson = get_geojson_path('alkmaar.geojson')

# Get the current working directory (useful in Jupyter notebooks)
current_dir = os.getcwd()  # This will give you the current working directory
parent_dir = os.path.join(current_dir, '..')  # Parent directory
folder_data = str(os.path.join(parent_dir, 'data'))  # Point to the 'data' folder

# %% [markdown]
# # BGT

bgt_data = download_bgt_for_alkmaar(path_geojson, folder_data)

# Use the clipped data for plotting
bgt_data_between_2019_2022 = filter_bgt_data(bgt_data, path_geojson)

# %% [Using radar data instead]

# Combined plot of radar-based change detection and BGT data - ALIGNED CRS APPROACH
if not bgt_data_between_2019_2022.empty:
    # Refresh token before radar detection
    print("🔄 Refreshing Sentinel Hub token...")
    token = get_token()
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    
    # Convert BGT data to WGS84 (same as radar data) instead of satellite CRS
    print("🔄 Converting BGT data to WGS84 (same as radar data)...")
    if bgt_data_between_2019_2022.crs.to_epsg() != 4326:
        print(f"Converting BGT data from {bgt_data_between_2019_2022.crs} to EPSG:4326")
        bgt_converted = bgt_data_between_2019_2022.to_crs('EPSG:4326')
    else:
        bgt_converted = bgt_data_between_2019_2022.copy()
    
    # Use the existing radar change detection from cell 32 - REUSE EXACTLY
    print("🔄 Using radar change detection from previous cell...")
    try:
        # Fetch radar data using the EXISTING function from cell 32
        print("📡 Fetching reference radar data (2019)...")
        ref_data = fetch_sentinel1_product(
            product_type='VV',
            start_date='2019-12-01',
            orbit_direction='DESCENDING'
        )
        
        print("📡 Fetching comparison radar data (2022)...")
        comp_data = fetch_sentinel1_product(
            product_type='VV',
            start_date='2022-12-01',
            orbit_direction='DESCENDING'
        )
        
        # Calculate change using same logic as cell 32
        change_map = comp_data - ref_data
        positive_changes_mask = change_map > 0.05  # Same threshold as cell 32
        
        print(f"✅ Radar data shapes - Ref: {ref_data.shape}, Comp: {comp_data.shape}")
        print(f"✅ Positive changes: {np.sum(positive_changes_mask)} pixels")
        
        # Get the radar bounds from the GeoJSON (which is in WGS84)
        with open(path_geojson) as f:
            gj = json.load(f)
        geom = gj["features"][0]["geometry"] if "features" in gj else gj.get("geometry", gj)
        
        # Extract bounds from geometry
        if geom['type'] == 'Polygon':
            coords = geom['coordinates'][0]
        else:
            coords = []
            for poly in geom['coordinates']:
                coords.extend(poly[0])
        
        lons = [coord[0] for coord in coords]
        lats = [coord[1] for coord in coords]
        radar_bounds = (min(lons), min(lats), max(lons), max(lats))
        
        print(f"🗺️ Radar bounds (WGS84): {radar_bounds}")
        
        # Create the plot with radar bounds
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Plot radar changes as base layer with proper extent
        radar_overlay = ax.imshow(
            positive_changes_mask.astype(float), 
            extent=[radar_bounds[0], radar_bounds[2], radar_bounds[1], radar_bounds[3]], # [minx, maxx, miny, maxy]
            cmap='Reds', 
            alpha=0.8,
            vmin=0, 
            vmax=1,
            aspect='equal',
            origin='upper'
        )
        
        # Layer 2: BGT legal changes on top (now both in WGS84)
        if 'source_file' in bgt_converted.columns:
            unique_sources = bgt_converted['source_file'].unique()
            
            print(f"📋 Found {len(unique_sources)} different BGT source files:")
            for source in unique_sources:
                count = len(bgt_converted[bgt_converted['source_file'] == source])
                print(f"   {source}: {count} features")
            
            # Define colors for BGT features (bright colors to stand out)
            bgt_colors = ['green', 'cyan', 'blue', 'purple', 'orange']
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
            
            # Add radar change detection to legend
            legend_handles.append(
                Patch(facecolor='red', alpha=0.8, label='Radar Changes (Potential Illegal Construction)')
            )
            
            # Add the legend
            if legend_handles:
                ax.legend(handles=legend_handles, loc='upper left', framealpha=0.9, 
                         fontsize=10, bbox_to_anchor=(0, 1))
        
        # Set plot properties to radar bounds
        ax.set_xlim(radar_bounds[0], radar_bounds[2])
        ax.set_ylim(radar_bounds[1], radar_bounds[3])
        
        print(f"✅ Plotted radar changes at {positive_changes_mask.shape} resolution")
        print(f"✅ Plotted {len(bgt_converted)} BGT features in same coordinate system")
        
        # Add title and format
        ax.set_title("Legal vs Potential Illegal Changes (2019-2022) - Radar + BGT Analysis", 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        
        # Add colorbar
        cbar = plt.colorbar(radar_overlay, ax=ax, shrink=0.6, aspect=20, pad=0.02)
        cbar.set_label('Radar Change Detection\n(1 = Change Detected)', rotation=270, labelpad=20)
        
        plt.tight_layout()
        plt.show()
        
        # Print summary using same approach as cell 32
        total_positive = np.sum(positive_changes_mask)
        total_pixels = positive_changes_mask.size
        positive_percentage = (total_positive / total_pixels) * 100
        
        print(f"\n📊 Combined Radar & BGT Analysis:")
        print(f"   Legal changes (BGT): {len(bgt_converted)} features")
        print(f"   Radar positive changes: {positive_percentage:.3f}% of area ({total_positive} pixels)")
        print(f"   🔍 Red areas show radar-detected construction/surface changes")
        print(f"   🟢 Colored polygons show legal BGT changes")
        print(f"   ⚖️ Areas with both may indicate legal construction projects")
        print(f"   🚨 Red-only areas may indicate unauthorized construction")
        
        # Additional radar insights (same as cell 32)
        print(f"\n🛰️ Radar Change Detection Insights:")
        if positive_percentage > 0.5:
            print("   📈 Significant construction activity detected")
        if positive_percentage > 2.0:
            print("   🏗️ Major construction/development activity")
        if positive_percentage < 0.1:
            print("   ✅ Minimal unauthorized construction detected")
            
    except Exception as e:
        print(f"❌ Failed to generate radar change detection data: {e}")
        print(f"Error details: {str(e)}")

else:
    print("⚠️ No BGT data to plot")