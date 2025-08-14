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
import satellite_images_nso.api.nso_georegion as nso
import satellite_images_nso.api.sat_manipulator as sat_manipulator
import matplotlib.pyplot as plt
import rasterio
import numpy as np
from dotenv import load_dotenv
import os
import json
import time
import zipfile
import requests
import geopandas as gpd
from tools import *
from illegal import *
from radar_copernicus import fetch_sentinel1_product, get_token

# %%
load_dotenv()

user_name = os.getenv('API_USERNAME')
user_password = os.getenv('API_PASSWORD')

# Fetch the geojson file for the region of interest
path_geojson = get_geojson_path('alkmaar.geojson')

# Get the current working directory (useful in Jupyter notebooks)
current_dir = os.getcwd()  # This will give you the current working directory
parent_dir = os.path.join(current_dir, '..')  # Parent directory
folder_data = str(os.path.join(parent_dir, 'data'))  # Point to the 'data' folder

# %%
# The first parameter is the path to the geojson, the second the map where the cropped satellite data will be downloaded, the third is your NSO username and the last your NSO password.
georegion = nso.nso_georegion(
    path_to_geojson=path_geojson,
    output_folder=folder_data,
    username=user_name,
    password=user_password,
)

# %% [markdown]
# # 2019

# %%
# This method fetches all the download links with all the satellite images the NSO has which contain the region in the given geojson.
# Max_diff parameters represents the amount of percentage the selected region has to be in the satellite image.
# So 1 is the the selected region has to be fully in the satellite images while 0.7 donates only 70% of the selected region is in the
links = georegion.retrieve_download_links(
    max_diff=0.5, start_date="2019-03-01", end_date="2019-10-01"
)

# %%
# This example filters out only 200 cm RGB Infrared Superview satellite imagery in the spring from all the links
season = "Spring"
links_group = []
for link in links:
    # Use 200 cm RGB Infrared Superview satellite imagery to get faster download links
    if "SV" in link and "200cm" in link and "RGBI" in link:
        if (
            sat_manipulator.get_season_for_month(
                int(link.split("/")[len(link.split("/")) - 1][4:6])
            )[0]
            == season
        ):
            links_group.append(link)

# %%
# Downloads a satellite image from the NSO, makes a crop out of it so it fits the geojson region and calculates the NVDI index.
# The output will stored in the output folder.
# The parameters are : link, delete_zip_file = False, delete_source_files = True,  plot=True, in_image_cloud_percentage = False,  add_ndvi_band = False, add_height_band = False
# description of these parameters can be found in the code.
georegion.execute_link(links_group[0],  delete_zip_file=True, plot=False, add_ndvi_band=True)

# %%
tif_path_2019 = get_tif_path(filename= "20190308_111637_SV1-01_200cm_RD_11bit_RGBI_Heiloo_alkmaar_cropped_ndvi.tif")

# %% [markdown]
# # 2022

# %%
# This method fetches all the download links with all the satellite images the NSO has which contain the region in the given geojson.
# Max_diff parameters represents the amount of percentage the selected region has to be in the satellite image.
# So 1 is the the selected region has to be fully in the satellite images while 0.7 donates only 70% of the selected region is in the
links = georegion.retrieve_download_links(
    max_diff=0.5, start_date="2022-01-01", end_date="2022-10-01"
)

# %%
# This example filters out only 50 cm RGB Infrared Superview satellite imagery in the summer from all the links
season = "Spring"
links_group = []
for link in links:
    # Gebruik hier 200cm om snel een beeld te krijgen
    if "SV" in link and "200cm" in link and "RGBI" in link:
        if (
            sat_manipulator.get_season_for_month(
                int(link.split("/")[len(link.split("/")) - 1][4:6])
            )[0]
            == season
        ):
            links_group.append(link)

# %%
# Downloads a satellite image from the NSO, makes a crop out of it so it fits the geojson region and calculates the NVDI index.
# The output will stored in the output folder.
# The parameters are : link, delete_zip_file = False, delete_source_files = True,  plot=True, in_image_cloud_percentage = False,  add_ndvi_band = False, add_height_band = False
# description of these parameters can be found in the code.
georegion.execute_link(links_group[0],  delete_zip_file=True, plot=False, add_ndvi_band=True, )

# %%
tif_path_2022 = get_tif_path(filename="20220308_103948_SV1-01_SV_RD_11bit_RGBI_200cm_Driehuizen_alkmaar_cropped_ndvi.tif")

# %% [markdown]
# # showing changes

# %%
# Run change detection to get the raw change magnitude
change_data = detect_visual_changes_proper(tif_path_2019, tif_path_2022)

analyze_change_thresholds(change_data)

# %%
# Simple change detection 
print("Comparing 2019 vs 2022 satellite images...")

# Plot the changes
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(np.stack([change_data['red1_norm'], change_data['green1_norm'], change_data['blue1_norm']], axis=-1))
plt.title('2019 Satellietbeeld')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(np.stack([change_data['red2_norm'], change_data['green2_norm'], change_data['blue2_norm']], axis=-1))
plt.title('2022 Satellietbeeld')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(change_data['change_magnitude_thresholded'], cmap='Reds', vmin=0, vmax=0.5)
plt.colorbar(label='Verandering')
plt.title('Veranderingsdetectie\n(Rood = meer verandering)')
plt.axis('off')

plt.tight_layout()
plt.show()

print(f"Gemiddelde verandering: {change_data['change_magnitude'].mean():.3f}")
print(f"Maximale verandering: {change_data['change_magnitude'].max():.3f}")

# %% [markdown]
# # BGT

# %%
bgt_data = download_bgt_for_alkmaar(path_geojson, folder_data)

# %%
# Use the clipped data for plotting
bgt_data_between_2019_2022 = filter_bgt_data(bgt_data, path_geojson)

# %%
# Use the updated function
if not bgt_data_between_2019_2022.empty:
    print("🎨 Creating BGT overlay grouped by source file...")
    plot_bgt_on_satellite_with_boundary(
        tif_path_2022, 
        bgt_data_between_2019_2022
    )
else:
    print("⚠️ No BGT data available for plotting")

# %%
# Execute the combined analysis
if not bgt_data_between_2019_2022.empty:
    print("🎨 Creating combined legal and visual changes analysis...")
    plot_combined_changes_analysis(
        tif_path_2022, tif_path_2019,
        bgt_data_between_2019_2022,
        change_data,  # This parameter is now ignored, we regenerate fresh data
        title="Legal vs Potential Illegal Changes (2019-2022) on 2022 Satellite Image"
    )
else:
    print("⚠️ Missing BGT data for combined analysis")

# %%
# Combined plot of change detection and BGT data without satellite background
if not bgt_data_between_2019_2022.empty:
    print("🎨 Creating combined changes overlay (no satellite background)...")
    plot_changes_and_bgt_overlay(
        tif_path_2022, 
        bgt_data_between_2019_2022,
        tif_path_2019,
        title="Legal vs Potential Illegal Changes (2019-2022)"
    )
else:
    print("⚠️ No BGT data to plot")

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
        with open(GEOJSON_PATH) as f:
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