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
