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
from illegal import *

# %%
load_dotenv()

user_name = os.getenv("API_USERNAME")
user_password = os.getenv("API_PASSWORD")

# Fetch the geojson file for the region of interest
path_geojson = get_geojson_path("alkmaar.geojson")

# Get the current working directory (useful in Jupyter notebooks)
current_dir = os.getcwd()  # This will give you the current working directory
parent_dir = os.path.join(current_dir, "..")  # Parent directory
folder_data = str(os.path.join(parent_dir, "data"))  # Point to the 'data' folder

# %%
# The first parameter is the path to the geojson, the second the map where the cropped satellite data will be downloaded, the third is your NSO username and the last your NSO password.
georegion = nso.nso_georegion(
    path_to_geojson=path_geojson,
    output_folder=folder_data,
    username=user_name,
    password=user_password,
)

# %%
# This method fetches all the download links with all the satellite images the NSO has which contain the region in the given geojson.
# Max_diff parameters represents the amount of percentage the selected region has to be in the satellite image.
# So 1 is the the selected region has to be fully in the satellite images while 0.7 donates only 70% of the selected region is in the
links = georegion.retrieve_download_links(
    max_diff=0.5, start_date="2022-01-01", end_date="2022-04-01"
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
georegion.execute_link(
    links_group[0], delete_zip_file=True, plot=False, add_ndvi_band=True
)

# %%
tif_path = get_tif_path()

# %%
from web_export import export_for_web, create_simple_server

# Option 2: Export just one image (original functionality)
bounds_info = export_for_web(tif_path, path_geojson)

server_url = create_simple_server(folder_data)
print(f"Server running at: {server_url}")


# %%
