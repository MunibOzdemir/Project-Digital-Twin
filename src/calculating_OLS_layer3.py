# %%
import json
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import statsmodels.api as sm
import warnings
import requests
from io import BytesIO
from tools import get_geojson_path

warnings.filterwarnings("ignore")

# Path to your GeoJSON files
GEOJSON_PATH = get_geojson_path("alkmaar.geojson")

# Read GeoJSON and extract correct geometry
with open(GEOJSON_PATH) as f:
    gj = json.load(f)

# %% [markdown]
# # Using PDOK data to measure green


# %%
def convert_gml_to_geojson_and_clip_robust(
    gml_path, alkmaar_geojson_path, output_path=None
):
    """
    Convert GML file to GeoJSON with robust CRS handling, then clip to Alkmaar boundary.

    Args:
        gml_path: Path to the input GML file
        alkmaar_geojson_path: Path to the Alkmaar boundary GeoJSON
        output_path: Output path for clipped GeoJSON (optional)

    Returns:
        Clipped GeoDataFrame
    """
    print(f"🔄 Loading GML file: {gml_path}")

    try:
        # Load the GML file
        gml_gdf = gpd.read_file(gml_path)
        print(f"✅ Loaded GML with {len(gml_gdf)} features")
        print(f"   Original CRS: {gml_gdf.crs}")

        # Handle missing CRS - common issue with GML files
        if gml_gdf.crs is None:
            print(
                "⚠️  GML file has no CRS defined. Attempting to detect or assign CRS..."
            )

            # Check the coordinate values to guess the CRS
            sample_bounds = gml_gdf.total_bounds
            print(f"   Sample bounds: {sample_bounds}")

            # If coordinates look like they're in the Netherlands and in meters, likely RD New (EPSG:28992)
            if (
                sample_bounds[0] > 10000
                and sample_bounds[0] < 300000
                and sample_bounds[1] > 300000
                and sample_bounds[1] < 700000
            ):
                print("   Coordinates suggest Dutch RD New projection (EPSG:28992)")
                gml_gdf.set_crs("EPSG:28992", inplace=True)
            # If coordinates look like lat/lon
            elif (
                sample_bounds[0] > -180
                and sample_bounds[0] < 180
                and sample_bounds[1] > -90
                and sample_bounds[1] < 90
            ):
                print("   Coordinates suggest WGS84 (EPSG:4326)")
                gml_gdf.set_crs("EPSG:4326", inplace=True)
            else:
                print(
                    "   ❌ Cannot determine CRS automatically. Assuming EPSG:28992 (Dutch standard)"
                )
                gml_gdf.set_crs("EPSG:28992", inplace=True)

        print(f"   Working CRS: {gml_gdf.crs}")

        # Load Alkmaar boundary
        print(f"🔄 Loading Alkmaar boundary: {alkmaar_geojson_path}")
        alkmaar_gdf = gpd.read_file(alkmaar_geojson_path)
        print(f"✅ Loaded Alkmaar boundary")
        print(f"   Alkmaar CRS: {alkmaar_gdf.crs}")

        # Convert GML to same CRS as Alkmaar if needed
        if gml_gdf.crs != alkmaar_gdf.crs:
            print(f"🔄 Converting GML from {gml_gdf.crs} to {alkmaar_gdf.crs}")
            gml_gdf = gml_gdf.to_crs(alkmaar_gdf.crs)
            print("✅ CRS conversion complete")
        else:
            print("✅ CRS already matches")

        # Clip GML data to Alkmaar boundary
        print("🔄 Clipping GML data to Alkmaar boundary...")
        clipped_gdf = gpd.clip(gml_gdf, alkmaar_gdf)
        print(
            f"✅ Clipping complete: {len(clipped_gdf)} features remain after clipping"
        )

        # Save to GeoJSON if output path provided
        if output_path:
            print(f"💾 Saving to: {output_path}")
            clipped_gdf.to_file(output_path, driver="GeoJSON")
            print("✅ GeoJSON saved successfully")

        return clipped_gdf

    except Exception as e:
        print(f"❌ Error processing GML file: {str(e)}")
        print("🔧 Trying alternative approach...")
        return convert_gml_step_by_step(gml_path, alkmaar_geojson_path, output_path)


def convert_gml_step_by_step(gml_path, alkmaar_geojson_path, output_path=None):
    """
    Alternative approach: Convert GML to GeoJSON first, then handle clipping.
    """
    try:
        print("🔧 Step 1: Converting GML to temporary GeoJSON...")

        # First, just convert to GeoJSON without CRS operations
        gml_gdf = gpd.read_file(gml_path)

        # Create temporary GeoJSON file
        temp_geojson = str(gml_path).replace(".gml", "_temp.geojson")

        # If no CRS, try to assign one based on data inspection
        if gml_gdf.crs is None:
            sample_bounds = gml_gdf.total_bounds
            print(f"   Inspecting coordinates: {sample_bounds}")

            # Dutch data is often in RD New (EPSG:28992)
            if sample_bounds[0] > 10000 and sample_bounds[1] > 300000:
                print("   Assigning EPSG:28992 (Dutch RD New)")
                gml_gdf.set_crs("EPSG:28992", inplace=True)
            else:
                print("   Assigning EPSG:4326 (WGS84)")
                gml_gdf.set_crs("EPSG:4326", inplace=True)

        # Save as temporary GeoJSON
        gml_gdf.to_file(temp_geojson, driver="GeoJSON")
        print(f"✅ Temporary GeoJSON saved: {temp_geojson}")

        print("🔧 Step 2: Loading and processing GeoJSON...")

        # Load the temporary GeoJSON
        temp_gdf = gpd.read_file(temp_geojson)
        print(f"   Loaded temp GeoJSON with CRS: {temp_gdf.crs}")

        # Load Alkmaar boundary
        alkmaar_gdf = gpd.read_file(alkmaar_geojson_path)

        # Convert to same CRS if needed
        if temp_gdf.crs != alkmaar_gdf.crs:
            print(f"   Converting from {temp_gdf.crs} to {alkmaar_gdf.crs}")
            temp_gdf = temp_gdf.to_crs(alkmaar_gdf.crs)

        # Clip to Alkmaar
        clipped_gdf = gpd.clip(temp_gdf, alkmaar_gdf)
        print(f"✅ Clipped to {len(clipped_gdf)} features")

        # Save final result
        if output_path:
            clipped_gdf.to_file(output_path, driver="GeoJSON")
            print(f"✅ Final GeoJSON saved: {output_path}")

        # Clean up temporary file
        import os

        if os.path.exists(temp_geojson):
            os.remove(temp_geojson)
            print("🗑️ Cleaned up temporary file")

        return clipped_gdf

    except Exception as e:
        print(f"❌ Alternative approach also failed: {str(e)}")
        return None


def convert_and_clip_with_auto_paths_robust(gml_filename, output_filename=None):
    """
    Robust version of the conversion function.

    Args:
        gml_filename: Name of GML file in data directory (can be full path)
        output_filename: Name for output file (optional)
    """
    from tools import get_geojson_path
    from pathlib import Path

    # Handle full path or just filename
    if gml_filename.startswith("/"):
        gml_path = Path(gml_filename)
        data_dir = gml_path.parent
    else:
        current_dir = (
            Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
        )
        data_dir = current_dir.parent / "data"
        gml_path = data_dir / gml_filename

    alkmaar_path = get_geojson_path("alkmaar.geojson")

    if output_filename:
        output_path = data_dir / output_filename
    else:
        # Auto-generate output filename
        base_name = gml_path.stem
        output_path = data_dir / f"{base_name}_alkmaar_clipped.geojson"

    print(f"📁 Input GML: {gml_path}")
    print(f"📁 Alkmaar boundary: {alkmaar_path}")
    print(f"📁 Output: {output_path}")

    # Check if GML file exists
    if not gml_path.exists():
        print(f"❌ GML file not found: {gml_path}")
        return None

    return convert_gml_to_geojson_and_clip_robust(gml_path, alkmaar_path, output_path)


# %%
GML_GREEN = get_geojson_path("top10nl_terrein.gml")

# Convert your specific GML file with robust handling
result = convert_and_clip_with_auto_paths_robust(
    GML_GREEN, "top10nl_terrein_alkmaar_clipped.geojson"
)

if result is not None:
    print(f"\n🎉 SUCCESS!")
    print(f"   Features after clipping: {len(result)}")
    print(f"   Final CRS: {result.crs}")
    print(f"   Columns available: {list(result.columns)}")

    # Show a sample of the data
    if len(result) > 0:
        print(f"\n📊 Sample data:")
        print(result.head())
else:
    print("❌ Conversion failed")


# %%
def calculate_green_percentage_per_wijk_clean(wijken_path, gdf):
    """
    Clean function to calculate green percentage per wijk using TOP10NL data.
    Returns only green areas with wijk boundaries and green percentage per wijk visualizations.
    """

    # Identify green areas based on land use classification
    green_keywords = [
        "grasland",
        "bos: loofbos",
        "dodenakker",
        "bos: griend",
        "boomgaard",
        "akkerland",
        "fruitkwekerij",
        "boomkwekerij",
    ]

    green_areas = []
    for col in ["typeLandgebruik", "fysiekVoorkomen"]:
        if col in gdf.columns:
            for keyword in green_keywords:
                matches = gdf[gdf[col].str.contains(keyword, case=False, na=False)]
                if len(matches) > 0:
                    green_areas.append(matches)

    if not green_areas:
        print("❌ No green areas found!")
        return None

    # Combine green areas and remove duplicates
    green_gdf = gpd.GeoDataFrame(pd.concat(green_areas, ignore_index=True))
    green_gdf = green_gdf.drop_duplicates(subset=["gml_id"])

    # Load wijken/buurten boundaries
    wijken_gdf = gpd.read_file(wijken_path)

    drop_names = [
        "Zuid",
        "Oudorp",
        "Overdie",
        "West",
        "Huiswaard",
        "De Mare",
        "Daalmeer/Koedijk",
        "Centrum",
        "Schermer",
        "Graft-De Rijp",
        "Vroonermeer",
    ]

    # drop wijken from dataframes
    wijken_gdf = wijken_gdf[~wijken_gdf["naam"].isin(drop_names)].copy()

    # Ensure same CRS
    if wijken_gdf.crs != green_gdf.crs:
        wijken_gdf = wijken_gdf.to_crs(green_gdf.crs)

    # Find name column
    name_col = None
    for col in ["naam", "wijknaam", "buurtnaam", "gebiedsnaam"]:
        if col in wijken_gdf.columns:
            name_col = col
            break

    # Calculate green percentage for each wijk
    results = []
    for idx, wijk_row in wijken_gdf.iterrows():
        wijk_name = (
            str(wijk_row[name_col])
            if name_col and pd.notna(wijk_row[name_col])
            else f"Wijk_{idx}"
        )
        wijk_geometry = wijk_row.geometry
        wijk_area = wijk_geometry.area

        # Find intersecting green areas
        intersecting_green = green_gdf[green_gdf.intersects(wijk_geometry)]

        if len(intersecting_green) == 0:
            green_area = 0
            green_percentage = 0
            green_features = 0
        else:
            # Calculate intersection areas
            intersection_areas = []
            for _, green_feature in intersecting_green.iterrows():
                try:
                    intersection = green_feature.geometry.intersection(wijk_geometry)
                    if hasattr(intersection, "area"):
                        intersection_areas.append(intersection.area)
                    else:
                        intersection_areas.append(0)
                except:
                    intersection_areas.append(0)

            green_area = sum(intersection_areas)
            green_percentage = (green_area / wijk_area) * 100 if wijk_area > 0 else 0
            green_features = len(intersecting_green)

        # Convert to hectares
        wijk_area_ha = wijk_area / 10000
        green_area_ha = green_area / 10000

        results.append(
            {
                "wijk_name": wijk_name,
                "green_percentage": green_percentage,
                "total_area_ha": wijk_area_ha,
                "green_area_ha": green_area_ha,
                "green_features_count": green_features,
                "geometry": wijk_geometry,
            }
        )

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Create only the two visualizations you want
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # 1. Green areas with wijk boundaries
    wijken_gdf.plot(ax=ax1, facecolor="none", edgecolor="black", linewidth=1)
    green_gdf.plot(
        ax=ax1, color="green", edgecolor="darkgreen", linewidth=0.2, alpha=0.7
    )
    ax1.set_title("Green Areas with Wijk Boundaries", fontsize=14, fontweight="bold")
    ax1.axis("off")

    # 2. Green percentage per wijk
    wijken_with_results = wijken_gdf.copy()
    wijken_with_results["green_percentage"] = results_df["green_percentage"].values

    if wijken_with_results["green_percentage"].max() > 0:
        wijken_with_results.plot(
            column="green_percentage",
            cmap="RdYlGn",
            legend=True,
            ax=ax2,
            edgecolor="black",
            linewidth=0.5,
        )
        ax2.set_title("Green Percentage per Wijk/Buurt", fontsize=14, fontweight="bold")
    else:
        wijken_with_results.plot(
            ax=ax2, facecolor="lightgray", edgecolor="black", linewidth=0.5
        )
        ax2.set_title("Wijken/Buurten (No Green Data)", fontsize=14, fontweight="bold")
    ax2.axis("off")

    plt.tight_layout()
    plt.show()

    return results_df


# Load TOP10NL data
geojson_green = get_geojson_path("top10nl_terrein_alkmaar_clipped.geojson")

gdf = gpd.read_file(geojson_green)


# Execute the clean analysis
wijken_path = get_geojson_path("alkmaar_wijken_buurten.geojson")
# Run the analysis and get results
top10nl_green_results_fixed = calculate_green_percentage_per_wijk_clean(
    wijken_path, gdf
)


# %%
def load_cbs_data(year):
    """
    Load CBS kerncijfers wijken en buurten data for specified year

    Parameters:
    year (int): Year of the data to download (e.g., 2022, 2021, 2020, etc.)

    Returns:
    pandas.DataFrame: CBS data for the specified year
    """

    # Construct the download URL
    url = f"https://download.cbs.nl/regionale-kaarten/kwb-{year}.xlsx"

    print(f"Downloading CBS data for year {year}...")
    print(f"URL: {url}")

    try:
        # Download the file
        response = requests.get(url)
        response.raise_for_status()  # Raises an HTTPError for bad responses

        # Read Excel file from memory
        excel_data = BytesIO(response.content)
        df_cbs = pd.read_excel(excel_data)

        print(f"✅ Successfully loaded CBS data for {year}")
        print(f"   Shape: {df_cbs.shape}")
        print(f"   Columns: {list(df_cbs.columns)}")

        return df_cbs

    except requests.exceptions.HTTPError as e:
        if response.status_code == 404:
            print(f"❌ Data for year {year} not found. URL: {url}")
            print("   This year might not be available. Try a different year.")
        else:
            print(f"❌ HTTP Error {response.status_code}: {e}")
        return None

    except Exception as e:
        print(f"❌ Error downloading data: {e}")
        return None


def load_multiple_years_cbs_data(years):
    """
    Load CBS data for multiple years and combine them

    Parameters:
    years (list): List of years to download

    Returns:
    pandas.DataFrame: Combined CBS data for all years
    """

    print(f"🔄 LOADING CBS DATA FOR MULTIPLE YEARS: {years}")
    print("=" * 60)

    all_years_data = []

    for year in years:
        df_year = load_cbs_data(year)
        if df_year is not None:
            # Add year column to track which year the data comes from
            df_year["data_year"] = year
            all_years_data.append(df_year)
            print(f"✅ Added {year} data: {len(df_year)} rows")
        else:
            print(f"❌ Failed to load data for {year}")

    if not all_years_data:
        print("❌ No data loaded for any year!")
        return None

    # Combine all years
    df_combined = pd.concat(all_years_data, ignore_index=True)
    print(f"\n🎉 COMBINED DATA SUMMARY:")
    print(f"   Total rows: {len(df_combined)}")
    print(f"   Years included: {sorted(df_combined['data_year'].unique())}")
    print(f"   Rows per year: {df_combined['data_year'].value_counts().sort_index()}")

    return df_combined


# Load data for three years: 2022, 2023, 2024
years_to_analyze = [2019, 2022, 2023, 2024]
df_cbs_multi = load_multiple_years_cbs_data(years_to_analyze)

# Define the columns to keep
columns_to_keep = [
    "regio",
    "gm_naam",
    "recs",
    "a_inw",
    "a_65_oo",
    "data_year",  # Keep the year column
]

# Filter the dataframe to keep only specified columns
if df_cbs_multi is not None:
    df_cbs_multi = df_cbs_multi[columns_to_keep]

    # Convert numeric columns
    for col in df_cbs_multi.columns:
        if col not in ["regio", "gm_naam", "recs", "data_year"]:
            try:
                df_cbs_multi[col] = pd.to_numeric(df_cbs_multi[col], errors="coerce")
            except:
                pass

    print(f"\n📊 FINAL MULTI-YEAR CBS DATA:")
    print(f"   Shape: {df_cbs_multi.shape}")
    print(f"   Data years: {sorted(df_cbs_multi['data_year'].unique())}")
else:
    print("❌ No multi-year CBS data available")

# %%
# Filter for Alkmaar municipality for all years
df_alkmaar_multi = df_cbs_multi[
    (df_cbs_multi["gm_naam"] == "Alkmaar") & (df_cbs_multi["recs"] == "Buurt")
].copy()

print(f"📍 MULTI-YEAR ALKMAAR DATA:")
print("=" * 40)
print(f"Shape: {df_alkmaar_multi.shape}")
print(f"Years: {sorted(df_alkmaar_multi['data_year'].unique())}")
print(f"Wijken per year: {df_alkmaar_multi['data_year'].value_counts().sort_index()}")


print(f"Shape: {df_alkmaar_multi.shape}")
print(f"Wijken per year: {df_alkmaar_multi['data_year'].value_counts().sort_index()}")

# Calculate percentage 65+ and other derived variables
df_alkmaar_multi["percent_old"] = (
    df_alkmaar_multi["a_65_oo"] / df_alkmaar_multi["a_inw"]
) * 100


# Display summary of the multi-year dataset
print(f"\n📊 MULTI-YEAR ALKMAAR DATASET SUMMARY:")
print("=" * 50)
print(f"Total observations: {len(df_alkmaar_multi)}")
print(f"Years: {sorted(df_alkmaar_multi['data_year'].unique())}")
print(f"Unique wijken: {df_alkmaar_multi['regio'].nunique()}")

# Check if we have roughly 66 wijken * 3 years = 198 observations
expected_rows = df_alkmaar_multi["regio"].nunique() * len(
    df_alkmaar_multi["data_year"].unique()
)
print(f"Expected rows: {expected_rows}")
print(f"Actual rows: {len(df_alkmaar_multi)}")


# %%
# Path to the KNMI file
Path_KNMI = get_geojson_path("result.txt")

# Find the header line to know where data starts
with open(Path_KNMI, "r") as f:
    lines = f.readlines()

skip_rows = 0
for i, line in enumerate(lines):
    if "STN,YYYYMMDD" in line:
        skip_rows = i
        break

# Read the KNMI file
df_knmi = pd.read_csv(Path_KNMI, skiprows=skip_rows, skipinitialspace=True)

# Convert numeric columns
for col in df_knmi.columns:
    if col != "STN":
        df_knmi[col] = pd.to_numeric(df_knmi[col], errors="coerce")

# Add date column
df_knmi["date"] = pd.to_datetime(df_knmi["YYYYMMDD"], format="%Y%m%d")

# Convert temperatures to Celsius (from 0.1°C)
df_knmi["temp_avg"] = df_knmi["TG"] / 10.0
df_knmi["temp_min"] = df_knmi["TN"] / 10.0
df_knmi["temp_max"] = df_knmi["TX"] / 10.0

# create dataframe with number of hot days for every year
df_hot_days = (
    df_knmi[df_knmi["temp_max"] > 27.0]
    .groupby(df_knmi["date"].dt.year)
    .size()
    .reset_index(name="hot_days")
)

# Merge hot days data with CBS data
df_alkmaar_multi = df_alkmaar_multi.merge(
    df_hot_days, left_on="data_year", right_on="date", how="left"
)


# %%
def clean_and_match_names(df_alkmaar, top10nl_green_results_fixed):
    """
    Clean and match regio names from df_alkmaar with wijk_name from top10nl_green_results_fixed.
    Handles spaces, hyphens, case differences, and common name variations.

    Args:
        df_alkmaar: DataFrame with 'regio' column
        top10nl_green_results_fixed: DataFrame with 'wijk_name' column

    Returns:
        df_alkmaar with updated 'regio' column and merge statistics
    """

    print("🔍 CLEANING AND MATCHING NAMES")
    print("=" * 50)

    # Create copies to avoid modifying originals
    df_alkmaar_clean = df_alkmaar.copy()

    # Get unique names from both datasets
    alkmaar_names = set(df_alkmaar["regio"].astype(str).unique())
    wijk_names = set(top10nl_green_results_fixed["wijk_name"].astype(str).unique())

    print(f"📊 Original names:")
    print(f"   df_alkmaar regions: {len(alkmaar_names)}")
    print(f"   top10nl wijk names: {len(wijk_names)}")

    # Manual mappings for cases that fuzzy matching can't handle
    manual_mappings = {
        "'t Rak-Noord": "Het Rak Noord",
        "Bloemwijk en Zocherkwartier": "Bloemwijk - Zocherstraat",
        "De Mare": "De Mare Centrum",
    }

    print(f"\n🔧 Manual mappings applied:")
    for original, mapped in manual_mappings.items():
        if original in alkmaar_names and mapped in wijk_names:
            print(f"   ✅ '{original}' → '{mapped}'")
        else:
            print(f"   ❌ '{original}' → '{mapped}' (one or both names not found)")

    # Function to normalize names for matching
    def normalize_name(name):
        """Normalize name for matching: lowercase, remove spaces/hyphens, etc."""
        if pd.isna(name):
            return ""
        name = str(name).lower()
        # Remove common variations
        name = name.replace(" ", "").replace("-", "").replace("_", "")
        name = name.replace("(", "").replace(")", "")
        name = name.replace("wijk", "").replace("buurt", "")
        name = name.strip()
        return name

    # Create normalized lookup dictionaries
    alkmaar_normalized = {normalize_name(name): name for name in alkmaar_names}
    wijk_normalized = {normalize_name(name): name for name in wijk_names}

    # Create mapping dictionary starting with manual mappings
    name_mapping = manual_mappings.copy()
    exact_matches = 0
    fuzzy_matches = 0
    manual_matches = len([k for k in manual_mappings.keys() if k in alkmaar_names])
    no_matches = []

    print(f"\n🔍 Automatic matching process:")

    for alkmaar_name in alkmaar_names:
        # Skip if already manually mapped
        if alkmaar_name in manual_mappings:
            continue

        normalized_alkmaar = normalize_name(alkmaar_name)

        # Try exact normalized match first
        if normalized_alkmaar in wijk_normalized:
            name_mapping[alkmaar_name] = wijk_normalized[normalized_alkmaar]
            exact_matches += 1
            continue

        # Try fuzzy matching
        best_match = None
        best_score = 0

        for wijk_name in wijk_names:
            normalized_wijk = normalize_name(wijk_name)

            # Calculate similarity (simple approach)
            if len(normalized_alkmaar) == 0 or len(normalized_wijk) == 0:
                continue

            # Check if one is contained in the other
            if (
                normalized_alkmaar in normalized_wijk
                or normalized_wijk in normalized_alkmaar
            ):
                score = min(len(normalized_alkmaar), len(normalized_wijk)) / max(
                    len(normalized_alkmaar), len(normalized_wijk)
                )
                if score > best_score:
                    best_score = score
                    best_match = wijk_name

            # Check for partial matches (at least 70% similarity)
            elif len(normalized_alkmaar) > 3 and len(normalized_wijk) > 3:
                # Simple character overlap scoring
                overlap = len(set(normalized_alkmaar) & set(normalized_wijk))
                total_chars = len(set(normalized_alkmaar) | set(normalized_wijk))
                score = overlap / total_chars if total_chars > 0 else 0

                if score > 0.7 and score > best_score:
                    best_score = score
                    best_match = wijk_name

        if best_match and best_score > 0.5:  # Only accept matches with >50% confidence
            name_mapping[alkmaar_name] = best_match
            fuzzy_matches += 1
            print(
                f"   🔗 Fuzzy match: '{alkmaar_name}' → '{best_match}' (score: {best_score:.2f})"
            )
        else:
            no_matches.append(alkmaar_name)
            print(f"   ❌ No match found: '{alkmaar_name}'")

    # Apply the mapping
    df_alkmaar_clean["regio"] = (
        df_alkmaar_clean["regio"].map(name_mapping).fillna(df_alkmaar_clean["regio"])
    )

    # Print matching statistics
    print(f"\n📈 MATCHING RESULTS:")
    print(f"   ✅ Exact matches: {exact_matches}")
    print(f"   🔗 Fuzzy matches: {fuzzy_matches}")
    print(f"   🔧 Manual matches: {manual_matches}")
    print(f"   ❌ No matches: {len(no_matches)}")
    total_matches = exact_matches + fuzzy_matches + manual_matches
    print(f"   📊 Total match rate: {(total_matches / len(alkmaar_names)) * 100:.1f}%")

    if no_matches:
        print(f"\n⚠️  Unmatched names:")
        for name in no_matches:
            print(f"     • {name}")

    return df_alkmaar_clean, name_mapping


def merge_datasets_with_matching(df_alkmaar, top10nl_green_results_fixed):
    """
    Complete function to clean names and merge the datasets.

    Args:
        df_alkmaar: DataFrame with CBS data
        top10nl_green_results_fixed: DataFrame with green space data

    Returns:
        Merged DataFrame
    """

    print("🔄 STARTING DATASET MERGE PROCESS")
    print("=" * 60)

    # Step 1: Clean and match names
    df_alkmaar_matched, mapping = clean_and_match_names(
        df_alkmaar, top10nl_green_results_fixed
    )

    # Step 2: Perform the merge
    print(f"\n🔗 MERGING DATASETS")
    print("=" * 30)

    print(f"Before merge:")
    print(f"   df_alkmaar rows: {len(df_alkmaar_matched)}")
    print(f"   top10nl rows: {len(top10nl_green_results_fixed)}")

    # Merge on the matched names
    merged_df = pd.merge(
        df_alkmaar_matched,
        top10nl_green_results_fixed,
        left_on="regio",
        right_on="wijk_name",
        how="outer",  # Use outer join to see all records
        indicator=True,  # Add column showing merge source
    )

    print(f"\nAfter merge:")
    print(f"   Total rows: {len(merged_df)}")
    print(f"   Both datasets: {sum(merged_df['_merge'] == 'both')}")
    print(f"   Only CBS (left): {sum(merged_df['_merge'] == 'left_only')}")
    print(f"   Only Green (right): {sum(merged_df['_merge'] == 'right_only')}")

    # Show merge details
    if sum(merged_df["_merge"] == "left_only") > 0:
        print(f"\n⚠️  CBS records without green data:")
        unmatched_cbs = merged_df[merged_df["_merge"] == "left_only"]["regio"].unique()
        for name in unmatched_cbs:
            print(f"     • {name}")

    if sum(merged_df["_merge"] == "right_only") > 0:
        print(f"\n⚠️  Green records without CBS data:")
        unmatched_green = merged_df[merged_df["_merge"] == "right_only"][
            "wijk_name"
        ].unique()
        for name in unmatched_green:
            print(f"     • {name}")

    # Keep only successfully merged records for analysis
    merged_df_clean = merged_df[merged_df["_merge"] == "both"].copy()
    merged_df_clean.drop("_merge", axis=1, inplace=True)

    print(f"\n✅ Final clean merged dataset: {len(merged_df_clean)} rows")

    return merged_df_clean, merged_df  # Return both clean and full versions


# Execute the merge
print("Merging df_alkmaar and top10nl_green_results_fixed...")

# Perform the merge with name matching
merged_data_multi_clean, merged_data_multi_full = merge_datasets_with_matching(
    df_alkmaar_multi, top10nl_green_results_fixed
)

# Display the results
print(f"\n📊 FINAL MERGED DATASET PREVIEW:")
print("=" * 40)
print(f"Shape: {merged_data_multi_clean.shape}")
print(f"Columns: {list(merged_data_multi_clean.columns)}")

if len(merged_data_multi_clean) > 0:
    print(f"\nFirst few rows:")
    print(
        merged_data_multi_clean[
            ["regio", "wijk_name", "green_percentage", "a_65_oo", "p_ste"]
        ].head()
    )
else:
    print("❌ No successful merges found!")

# %%
# make all NAs zero
merged_data_multi_clean = merged_data_multi_clean.fillna(0)

# Make all zeros very slightly more for log
merged_data_multi_clean = merged_data_multi_clean.replace(0, 10**-5)

# training data
training_data = merged_data_multi_clean[
    merged_data_multi_clean["data_year"].isin([2019, 2022, 2023])
]


# %%
# Multi-year OLS Regression Analysis
print("🔬 MULTI-YEAR OLS REGRESSION ANALYSIS")
print("=" * 60)

y_multi = training_data["a_ste"]

# log needed variables
log_population_multi = np.log(training_data["a_inw"])
log_y_multi = np.log(y_multi)


# Create predictor matrix for multi-year analysis
X_multi = pd.DataFrame(
    {
        "hot_days": training_data["hot_days"],
        "percent_old": training_data["percent_old"],
        "green_percentage": training_data["green_percentage"],
        "hot_days_x_percent_old": training_data["hot_days"]
        * training_data["percent_old"],
        "hot_days_x_green": training_data["hot_days"]
        * training_data["green_percentage"],
        "log_population": log_population_multi,
    }
)


# Add constant
X_multi = sm.add_constant(X_multi)

print(f"\n📊 MULTI-YEAR MODEL MATRIX:")
print("=" * 40)
print(f"Shape: {X_multi.shape}")
print(f"Features: {list(X_multi.columns)}")

# Fit multi-year OLS model
print(f"\n🎯 FITTING MULTI-YEAR OLS MODEL:")
print("=" * 40)

try:
    ols_model_multi = sm.OLS(log_y_multi, X_multi).fit()

    print("Multi-year OLS Results:")
    print(ols_model_multi.summary())

except Exception as e:
    print(f"❌ Error fitting multi-year model: {e}")
    print("\nThis might be due to multicollinearity or other data issues.")
    print("Let's investigate the data structure further...")
