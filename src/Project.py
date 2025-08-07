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
from rasterio.warp import reproject, Resampling
from tools import *

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
    max_diff=0.5, start_date="2019-05-01", end_date="2019-10-01"
)

# Inspect the links
print(f"Found links: {len(links)}")
for link in links:
    print(link)

# %%
# This example filters out only 200 cm RGB Infrared Superview satellite imagery in the spring from all the links
season = "Summer"
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

# Inspect the filtered links
print(f"Found links in the group: {len(links_group)}")
for link in links_group:
    print(link)

# %%
# Downloads a satellite image from the NSO, makes a crop out of it so it fits the geojson region and calculates the NVDI index.
# The output will stored in the output folder.
# The parameters are : link, delete_zip_file = False, delete_source_files = True,  plot=True, in_image_cloud_percentage = False,  add_ndvi_band = False, add_height_band = False
# description of these parameters can be found in the code.
georegion.execute_link(links_group[0],  delete_zip_file=True, plot=False, add_ndvi_band=True)

# %%
tif_path_2019 = get_tif_path()
print(f"Most recent TIFF file: {tif_path_2019}")

# Open the TIFF file and read the bands
with rasterio.open(tif_path_2019) as src:
    red = src.read(3).astype(np.float32)
    green = src.read(2).astype(np.float32)
    blue = src.read(1).astype(np.float32)
    nir= src.read(4).astype(np.float32)

    # Normalize the bands to the range [0, 1]
    max_val = max(red.max(), green.max(), blue.max())
    red /= max_val
    green /= max_val
    blue /= max_val
    nir /= max_val

    # Stack the bands to create an RGB image
    rgb = np.stack((red, green, blue), axis=-1)

    print(f"Number of bands: {src.count}")

# Visualize the RGB image
plt.figure(figsize=(10, 10))
plt.imshow(rgb)
plt.title("RGB Satellietbeeld (geschaald)")
plt.axis("off")
plt.show()

# %% [markdown]
# # 2022

# %%
# This method fetches all the download links with all the satellite images the NSO has which contain the region in the given geojson.
# Max_diff parameters represents the amount of percentage the selected region has to be in the satellite image.
# So 1 is the the selected region has to be fully in the satellite images while 0.7 donates only 70% of the selected region is in the
links = georegion.retrieve_download_links(
    max_diff=0.5, start_date="2022-01-01", end_date="2022-10-01"
)
print(f"Aantal gevonden links: {len(links)}")
for link in links:
    print(link)

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

print(f"Aantal gevonden links in de groep: {len(links_group)}")
for link in links_group:
    print(link)

# %%
# Downloads a satellite image from the NSO, makes a crop out of it so it fits the geojson region and calculates the NVDI index.
# The output will stored in the output folder.
# The parameters are : link, delete_zip_file = False, delete_source_files = True,  plot=True, in_image_cloud_percentage = False,  add_ndvi_band = False, add_height_band = False
# description of these parameters can be found in the code.
georegion.execute_link(links_group[0],  delete_zip_file=True, plot=False, add_ndvi_band=True, )

# %%
tif_path_2022 = get_tif_path()
print(f"Most recent TIFF file: {tif_path_2022}")

# Open the TIFF file and read the bands
with rasterio.open(tif_path_2022) as src:
    red = src.read(3).astype(np.float32)
    green = src.read(2).astype(np.float32)
    blue = src.read(1).astype(np.float32)
    nir= src.read(4).astype(np.float32)

    # Normalize the bands to the range [0, 1]
    max_val = max(red.max(), green.max(), blue.max())
    red /= max_val
    green /= max_val
    blue /= max_val
    nir /= max_val

    # Stack the bands to create an RGB image
    rgb = np.stack((red, green, blue), axis=-1)

    print(f"Number of bands: {src.count}")

# Visualize the RGB image
plt.figure(figsize=(10, 10))
plt.imshow(rgb)
plt.title("RGB Satellietbeeld (geschaald)")
plt.axis("off")
plt.show()

# %% [markdown]
# # showing changes

# %%
# Simple change detection - add this to your "Next step" cell
print("Comparing 2019 vs 2022 satellite images...")

change_data = detect_visual_changes_proper(tif_path_2019, tif_path_2022)

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
plt.imshow(change_data['change_magnitude'], cmap='Reds', vmin=0, vmax=0.5)
plt.colorbar(label='Verandering')
plt.title('Veranderingsdetectie\n(Rood = meer verandering)')
plt.axis('off')

plt.tight_layout()
plt.show()

print(f"Gemiddelde verandering: {change_data['change_magnitude'].mean():.3f}")
print(f"Maximale verandering: {change_data['change_magnitude'].max():.3f}")

# %% [markdown]
# ## Save as tiff file

# %%
# # Save the change magnitude as a TIFF file
# changes_tif_path = os.path.join(folder_data, 'change_detection_2019_2022.tif')

# # Use the 2019 image as reference for geospatial properties
# with rasterio.open(tif_path_2019) as src:
#     # Copy metadata from the reference image
#     profile = src.profile.copy()
    
#     # Update for single band output
#     profile.update({
#         'count': 1,
#         'dtype': 'float32'
#     })
    
#     # Write the change magnitude to a new TIFF
#     with rasterio.open(changes_tif_path, 'w', **profile) as dst:
#         dst.write(change_data['change_magnitude'].astype('float32'), 1)

# print(f"Change detection layer saved to: {changes_tif_path}")


# %% [markdown]
# # export to web (leave for now)

# %%
from web_export import export_for_web, create_simple_server

# Option 1: Export with change detection (both images) - most new one first
#bounds_info = export_for_web(tif_path_2022, path_geojson, tif_path_2019)

# Option 2: Export just one image (original functionality)
bounds_info = export_for_web(tif_path_2022, path_geojson)

server_url = create_simple_server(folder_data)
print(f"Server running at: {server_url}")

# %% [markdown]
# # Misschien niet nodig

# %%
# Calculate NDVI and visualize it
# NDVI = (NIR - Red) / (NIR + Red)
# NIR is the near-infrared band, and Red is the red band.
# The NDVI value ranges from -1 to 1, where higher values indicate healthier vegetation.
ndvi = (nir - red) / (nir + red + 1e-10)  # +1e-10 to avoid division by zero
plt.figure(figsize=(10, 10))
plt.imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
plt.colorbar(label='NDVI waarde')
plt.title('NDVI berekening')
plt.axis('off')
plt.show()
