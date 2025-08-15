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
from matplotlib import colors

warnings.filterwarnings("ignore")

# --- Configuration & credentials ---
CLIENT_ID = "sh-93c6fbd0-8c4a-4e40-8c59-d06889413797"
CLIENT_SECRET = "LKVq6MTE0S3kohQjRI1Yuj03aU5frOTm"
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"

# Path to your GeoJSON files
GEOJSON_PATH = path_geojson = get_geojson_path("alkmaar.geojson")

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
    if geometry["type"] == "Polygon":
        coords = geometry["coordinates"][0]
    elif geometry["type"] == "MultiPolygon":
        coords = []
        for poly in geometry["coordinates"]:
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
    mosaicking_order="mostRecent",
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
    if product_type == "VV":
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
    elif product_type == "VH":
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
    elif product_type == "VV_VH_RATIO":
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
    elif product_type == "RGB_VV_VH":
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
                        "orbitDirection": orbit_direction,
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
        if product_type == "VV":
            arr = (arr * 25) - 25  # Convert to dB
        elif product_type == "VH":
            arr = (arr * 25) - 30  # Convert to dB
        elif product_type == "VV_VH_RATIO":
            arr = arr * 15  # Convert to ratio dB

        return arr.squeeze() if arr.ndim == 3 and arr.shape[2] == 1 else arr

    except Exception as e:
        print(f"Error fetching {product_type} data: {e}")
        raise


# %%
import pandas as pd
import requests
from io import BytesIO


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


# Example usage:
year = 2022  # Change this variable to load different years

# Load the data
df_cbs = load_cbs_data(year)

# Define the columns to keep
columns_to_keep = [
    "regio",
    "gm_naam",
    "recs",
    "a_inw",
    "a_65_oo",
    "a_ste",
    "p_ste",
    "a_inkont",
    "g_ink_po",
    "g_ink_pi",
    "p_ink_li",
    "p_ink_hi",
    "g_hh_sti",
    "p_hh_li",
    "p_hh_hi",
    "p_hh_lkk",
    "p_hh_osm",
    "p_hh_110",
    "p_hh_120",
    "m_hh_ver",
    "a_opp_ha",
    "a_lan_ha",
    "a_wat_ha",
]

# Filter the dataframe to keep only specified columns
df_cbs = df_cbs[columns_to_keep]

"""
# Convert numeric columns
# Find the position of 'recs'
recs_idx = df_cbs.columns.get_loc("recs")

# Convert all columns before 'recs' to numeric
for col in df_cbs.columns[recs_idx:]:
    df_cbs[col] = pd.to_numeric(df_cbs[col], errors='coerce')
 """
for col in df_cbs.columns:
    try:
        df_cbs[col] = pd.to_numeric(df_cbs[col], errors="ignore")
    except:
        pass

# df_alkmaar_2 = df_cbs[(df_cbs["gm_naam"] == "Alkmaar")].copy()


# %%
# load greenness data
# Path_green = "/Users/twanzwetsloot/Downloads/Green_percentage_wijk.csv"
Path_green = get_geojson_path("Green_percentage_wijk.csv")

df_green = pd.read_csv(Path_green)

# %%
# Direct WFS URL in WGS84
"""
geojson_url = (
    "https://datalab.alkmaar.nl/geoserver/Alkmaar/wfs"
    "?service=WFS&version=1.0.0&request=GetFeature"
    "&typeName=Alkmaar:GEBIED"
    "&outputFormat=application/json"
    "&srsName=EPSG:4326"
)

# Download GeoJSON
r = requests.get(geojson_url)
r.raise_for_status()

# Save locally so your existing code can still use it
with open(GEOJSON_PATH, "wb") as f:
    f.write(r.content)
"""
gdf_file = get_geojson_path("alkmaar_wijken_buurten.geojson")

# Read with geopandas
gdf = gpd.read_file(gdf_file)

# Plot the geometry
gdf.plot(edgecolor="black", facecolor="none")
plt.title("Wijken & Buurten - Gemeente Alkmaar")
plt.show()

# %%
import pandas as pd
import geopandas as gpd
from difflib import SequenceMatcher
import re
import numpy as np


class RobustNeighborhoodMerger:
    """
    Robust merger for neighborhood data with comprehensive preprocessing and fuzzy matching
    """

    def __init__(self, similarity_threshold=0.8):
        self.similarity_threshold = similarity_threshold
        self.name_mapping = {}
        self.merge_stats = {}

    def normalize_name(self, name):
        """
        Comprehensive name normalization for consistent matching
        """
        if pd.isna(name) or name == "" or name is None:
            return ""

        # Convert to string and lowercase
        normalized = str(name).lower().strip()

        # Remove extra whitespace
        normalized = re.sub(r"\s+", " ", normalized)

        # Standardize directional terms (order matters - do compound directions first)
        direction_replacements = [
            (r"[-\s]*noord[-\s]*oost[-\s]*", "noordoost"),
            (r"[-\s]*noord[-\s]*west[-\s]*", "noordwest"),
            (r"[-\s]*zuid[-\s]*oost[-\s]*", "zuidoost"),
            (r"[-\s]*zuid[-\s]*west[-\s]*", "zuidwest"),
            (r"[-\s]*noord[-\s]*", "noord"),
            (r"[-\s]*zuid[-\s]*", "zuid"),
            (r"[-\s]*oost[-\s]*", "oost"),
            (r"[-\s]*west[-\s]*", "west"),
        ]

        for pattern, replacement in direction_replacements:
            normalized = re.sub(pattern, replacement, normalized)

        # Standardize common connectors
        normalized = re.sub(r"\s+(en|-)\s+", "", normalized)
        normalized = re.sub(r"[-\s]+", "", normalized)

        return normalized

    def create_manual_replacements(self):
        """
        Define manual replacements for known mismatches
        """
        return {
            # Main replacements from original code
            "De Mare": "De Mare-Noord",
            "De Mare Centrum": "De Mare-Noord",  # Additional variation
            "Staatslieden - Landstratenkwartier": "Staatsliedenkwartier en Landstraten",
            "Bloemwijk - Zocherstraat": "Bloemwijk en Zocherkwartier",
        }

    def apply_manual_fixes(self, df, column_name):
        """
        Apply manual name fixes before fuzzy matching
        """
        df = df.copy()
        replacements = self.create_manual_replacements()

        # Apply case-insensitive replacements
        for old_name, new_name in replacements.items():
            # Direct replacement
            df[column_name] = df[column_name].replace(old_name, new_name)

            # Case-insensitive replacement
            mask = df[column_name].str.lower() == old_name.lower()
            df.loc[mask, column_name] = new_name

        return df

    def create_fuzzy_mapping(self, source_names, target_names, threshold=None):
        """
        Create fuzzy matching mapping between two sets of names
        """
        if threshold is None:
            threshold = self.similarity_threshold

        # Normalize names for comparison
        source_normalized = {
            name: self.normalize_name(name) for name in source_names if name
        }
        target_normalized = {
            name: self.normalize_name(name) for name in target_names if name
        }

        mapping = {}
        unmatched_source = set(source_names)
        unmatched_target = set(target_names)

        # Phase 1: Exact matches after normalization
        for source_name, source_norm in source_normalized.items():
            if not source_norm:
                continue

            for target_name, target_norm in target_normalized.items():
                if source_norm == target_norm:
                    mapping[source_name] = target_name
                    unmatched_source.discard(source_name)
                    unmatched_target.discard(target_name)
                    break

        # Phase 2: Fuzzy matches for remaining names
        for source_name in list(unmatched_source):
            source_norm = source_normalized.get(source_name, "")
            if not source_norm:
                continue

            best_match = None
            best_ratio = 0

            for target_name in unmatched_target:
                target_norm = target_normalized.get(target_name, "")
                if not target_norm:
                    continue

                ratio = SequenceMatcher(None, source_norm, target_norm).ratio()
                if ratio >= threshold and ratio > best_ratio:
                    best_ratio = ratio
                    best_match = target_name

            if best_match:
                mapping[source_name] = best_match
                unmatched_source.discard(source_name)
                unmatched_target.discard(best_match)

        # Store statistics
        self.merge_stats = {
            "total_source": len(source_names),
            "total_target": len(target_names),
            "matched": len(mapping),
            "unmatched_source": list(unmatched_source),
            "unmatched_target": list(unmatched_target),
            "match_ratio": len(mapping) / len(source_names) if source_names else 0,
        }

        return mapping

    def prepare_data(self, gdf, df_cbs, df_green):
        """
        Prepare and clean all datasets for merging
        """
        print("🔄 Starting data preparation...")

        # 1. Filter unwanted regions from GDF
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

        gdf_clean = gdf[~gdf["naam"].isin(drop_names)].copy()
        print(
            f"   📊 GDF: {len(gdf)} → {len(gdf_clean)} rows (removed {len(drop_names)} unwanted regions)"
        )

        # 2. Filter CBS data for Alkmaar Buurt level
        df_alkmaar = df_cbs[
            (df_cbs["gm_naam"] == "Alkmaar") & (df_cbs["recs"] == "Buurt")
        ].copy()
        print(
            f"   📊 CBS: {len(df_cbs)} → {len(df_alkmaar)} rows (filtered for Alkmaar Buurt)"
        )

        # 3. Apply manual fixes to all dataframes
        gdf_clean = self.apply_manual_fixes(gdf_clean, "naam")
        df_alkmaar = self.apply_manual_fixes(df_alkmaar, "regio")
        df_green_clean = self.apply_manual_fixes(df_green, "wijk_name")

        print(f"   ✅ Applied manual name fixes to all datasets")

        return gdf_clean, df_alkmaar, df_green_clean

    def merge_datasets(self, gdf, df_cbs, df_green, verbose=True):
        """
        Complete pipeline to merge all datasets with robust name matching
        """
        if verbose:
            print("🚀 Starting robust neighborhood data merging...")
            print("=" * 60)

        # Step 1: Prepare data
        gdf_clean, df_alkmaar, df_green_clean = self.prepare_data(gdf, df_cbs, df_green)

        # Step 2: Create name mapping between CBS and GDF
        if verbose:
            print("\n🔍 Creating name mapping between CBS and GeoDataFrame...")

        cbs_names = set(df_alkmaar["regio"].dropna().unique())
        gdf_names = set(gdf_clean["naam"].dropna().unique())

        name_mapping = self.create_fuzzy_mapping(cbs_names, gdf_names)

        if verbose:
            print(
                f"   🎯 Mapped {len(name_mapping)}/{len(cbs_names)} CBS regions to GDF names"
            )
            print(f"   📈 Match ratio: {self.merge_stats['match_ratio']:.1%}")

            if self.merge_stats["unmatched_source"]:
                print(
                    f"   ⚠️ Unmatched CBS regions: {self.merge_stats['unmatched_source']}"
                )
            if self.merge_stats["unmatched_target"]:
                print(
                    f"   ⚠️ Unmatched GDF regions: {self.merge_stats['unmatched_target']}"
                )

        # Step 3: Apply mapping and merge CBS data
        df_alkmaar["regio_mapped"] = df_alkmaar["regio"].map(name_mapping)

        # Merge GeoDataFrame with CBS data
        gdf_merged = gdf_clean.merge(
            df_alkmaar, left_on="naam", right_on="regio_mapped", how="left"
        )

        # Remove any records that might have been matched to 'Wijk' level (safety check)
        gdf_merged = gdf_merged[gdf_merged["recs"] != "Wijk"].copy()

        if verbose:
            print(f"\n📊 After CBS merge: {len(gdf_merged)} rows")
            cbs_matched = gdf_merged["regio_mapped"].notna().sum()
            print(
                f"   ✅ {cbs_matched}/{len(gdf_clean)} regions matched with CBS data ({cbs_matched / len(gdf_clean):.1%})"
            )

        # Step 4: Create mapping for green data
        if verbose:
            print("\n🌱 Merging green space data...")

        green_names = set(df_green_clean["wijk_name"].dropna().unique())
        green_mapping = self.create_fuzzy_mapping(green_names, gdf_names)

        if verbose:
            print(
                f"   🎯 Mapped {len(green_mapping)}/{len(green_names)} green regions to GDF names"
            )
            print(f"   📈 Match ratio: {self.merge_stats['match_ratio']:.1%}")

        # Step 5: Merge green data
        gdf_final = gdf_merged.merge(
            df_green_clean, left_on="naam", right_on="wijk_name", how="left"
        )

        if verbose:
            print(f"\n📊 Final dataset: {len(gdf_final)} rows")
            green_matched = gdf_final["wijk_name"].notna().sum()
            print(
                f"   🌿 {green_matched}/{len(gdf_merged)} regions matched with green data ({green_matched / len(gdf_merged):.1%})"
            )

        # Step 6: Final cleanup and validation
        gdf_final = self._finalize_geodataframe(gdf_final, verbose)

        if verbose:
            print("\n🎉 Merging completed successfully!")
            print("=" * 60)
            self._print_final_stats(gdf_final)

        return gdf_final

    def _finalize_geodataframe(self, gdf_merged, verbose=True):
        """
        Final cleanup and validation of the merged GeoDataFrame
        """
        if verbose:
            print("\n🔧 Finalizing GeoDataFrame...")

        # Ensure it's a proper GeoDataFrame
        if not isinstance(gdf_merged, gpd.GeoDataFrame):
            if "geometry_x" in gdf_merged.columns:
                gdf_merged = gpd.GeoDataFrame(gdf_merged, geometry="geometry_x")
            elif "geometry" in gdf_merged.columns:
                gdf_merged = gpd.GeoDataFrame(gdf_merged, geometry="geometry")
            else:
                raise ValueError("No geometry column found in the dataframe")

        # Set the active geometry column explicitly
        if "geometry_x" in gdf_merged.columns and "geometry_y" in gdf_merged.columns:
            # Use the original geometry (from gdf)
            gdf_merged = gdf_merged.set_geometry("geometry_x")
            if verbose:
                print("   🗺️ Set geometry to 'geometry_x' (original GDF geometry)")
        elif "geometry" in gdf_merged.columns:
            gdf_merged = gdf_merged.set_geometry("geometry")
            if verbose:
                print("   🗺️ Set geometry to 'geometry'")

        # Handle numeric columns that might need conversion (but don't drop rows)
        numeric_columns = ["m_hh_ver"]  # Add other numeric columns as needed

        for col in numeric_columns:
            if col in gdf_merged.columns:
                # Convert to numeric, keeping NaN for invalid values
                original_na_count = gdf_merged[col].isna().sum()
                gdf_merged[col] = pd.to_numeric(gdf_merged[col], errors="coerce")
                new_na_count = gdf_merged[col].isna().sum()

                if verbose:
                    if new_na_count > original_na_count:
                        newly_invalid = new_na_count - original_na_count
                        print(
                            f"   ℹ️ Column '{col}': {newly_invalid} values converted to NaN (invalid data)"
                        )

                    if new_na_count > 0:
                        valid_count = len(gdf_merged) - new_na_count
                        print(
                            f"   ℹ️ Column '{col}': {valid_count}/{len(gdf_merged)} regions have valid data ({valid_count / len(gdf_merged):.1%})"
                        )
                    else:
                        print(f"   ✅ Column '{col}': All regions have valid data")

        # Only drop rows if they have no geometry (critical for mapping)
        original_count = len(gdf_merged)
        gdf_merged = gdf_merged.dropna(subset=[gdf_merged.geometry.name])

        if verbose and len(gdf_merged) < original_count:
            dropped = original_count - len(gdf_merged)
            print(f"   ⚠️ Dropped {dropped} rows with invalid geometry (critical)")

        return gdf_merged

    def _print_final_stats(self, gdf_final):
        """
        Print summary statistics of the final merged dataset
        """
        print(f"\n📈 Final Dataset Summary:")
        print(f"   • Total regions: {len(gdf_final)}")
        print(f"   • Columns: {len(gdf_final.columns)}")

        # Check data completeness
        if "regio_mapped" in gdf_final.columns:
            cbs_complete = gdf_final["regio_mapped"].notna().sum()
            print(
                f"   • Regions with CBS data: {cbs_complete} ({cbs_complete / len(gdf_final):.1%})"
            )

        if "wijk_name" in gdf_final.columns:
            green_complete = gdf_final["wijk_name"].notna().sum()
            print(
                f"   • Regions with green data: {green_complete} ({green_complete / len(gdf_final):.1%})"
            )

        # Check for key numeric columns
        if "m_hh_ver" in gdf_final.columns:
            valid_hh = gdf_final["m_hh_ver"].notna().sum()
            print(
                f"   • Regions with valid household data: {valid_hh} ({valid_hh / len(gdf_final):.1%})"
            )

            if valid_hh < len(gdf_final):
                missing_hh = len(gdf_final) - valid_hh
                print(
                    f"   • Regions with missing household data: {missing_hh} (preserved for mapping)"
                )

        print(f"   ✅ All {len(gdf_final)} geographic regions preserved")

    def get_merge_report(self):
        """
        Get detailed report of the merging process
        """
        return {
            "merge_stats": self.merge_stats,
            "similarity_threshold": self.similarity_threshold,
            "manual_replacements": self.create_manual_replacements(),
        }


# Example usage:
def merge_neighborhood_data(
    gdf, df_cbs, df_green, similarity_threshold=0.8, verbose=True
):
    """
    Convenience function to merge neighborhood datasets

    Parameters:
    -----------
    gdf : GeoDataFrame
        Geographic data with 'naam' column
    df_cbs : DataFrame
        CBS data with 'gm_naam', 'recs', and 'regio' columns
    df_green : DataFrame
        Green space data with 'wijk_name' column
    similarity_threshold : float
        Threshold for fuzzy name matching (0.0 to 1.0)
    verbose : bool
        Whether to print progress information

    Returns:
    --------
    GeoDataFrame
        Merged dataset with all three data sources
    """
    merger = RobustNeighborhoodMerger(similarity_threshold=similarity_threshold)
    return merger.merge_datasets(gdf, df_cbs, df_green, verbose=verbose)


# Usage example:
# gdf_merged = merge_neighborhood_data(gdf, df_cbs, df_green, verbose=True)

# %%
# Simple usage
gdf_merged = merge_neighborhood_data(gdf, df_cbs, df_green)

# Or with custom parameters
merger = RobustNeighborhoodMerger(similarity_threshold=0.5)
gdf_merged = merger.merge_datasets(gdf, df_cbs, df_green, verbose=True)

# Get detailed report
report = merger.get_merge_report()

# %% calculate percentage 65+ age
gdf_merged["perc_65"] = gdf_merged["a_65_oo"] / gdf_merged["a_inw"] * 100

# %%
import pandas as pd

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

# Filter for specific year (using the year variable)
df_knmi = df_knmi[df_knmi["date"].dt.year == year]

# Heat wave analysis
hot_days = df_knmi[df_knmi["temp_max"] > 30.0]  # Days above 30°C
tropical_nights = df_knmi[df_knmi["temp_min"] > 20.0]  # Nights above 20°C

print(f"=== Heat Analysis for {year} ===")
print(f"Days with max temperature > 30°C: {len(hot_days)}")
print(f"Nights with min temperature > 20°C: {len(tropical_nights)}")

if len(hot_days) > 0:
    print(f"\nHot days (>30°C):")
    for _, row in hot_days.iterrows():
        print(f"  {row['date'].strftime('%Y-%m-%d')}: {row['temp_max']:.1f}°C")

if len(tropical_nights) > 0:
    print(f"\nTropical nights (>20°C):")
    for _, row in tropical_nights.iterrows():
        print(f"  {row['date'].strftime('%Y-%m-%d')}: {row['temp_min']:.1f}°C")


# Consecutive hot days analysis
def find_consecutive_periods(df, condition_column, threshold, condition_name):
    """Find consecutive periods where condition is met"""
    condition_met = df[condition_column] > threshold

    # Find groups of consecutive days
    groups = (condition_met != condition_met.shift()).cumsum()
    consecutive_periods = df[condition_met].groupby(groups[condition_met])

    periods = []
    for group_id, group in consecutive_periods:
        if len(group) >= 1:  # At least 1 day
            periods.append(
                {
                    "start_date": group["date"].min(),
                    "end_date": group["date"].max(),
                    "duration": len(group),
                    "max_temp": group[condition_column].max(),
                }
            )

    if periods:
        print(f"\nConsecutive {condition_name} periods:")
        for i, period in enumerate(periods, 1):
            if period["duration"] == 1:
                print(
                    f"  {i}. {period['start_date'].strftime('%Y-%m-%d')}: 1 day ({period['max_temp']:.1f}°C)"
                )
            else:
                print(
                    f"  {i}. {period['start_date'].strftime('%Y-%m-%d')} to {period['end_date'].strftime('%Y-%m-%d')}: {period['duration']} days (max: {period['max_temp']:.1f}°C)"
                )

        longest = max(periods, key=lambda x: x["duration"])
        print(
            f"  Longest period: {longest['duration']} days ({longest['start_date'].strftime('%Y-%m-%d')} to {longest['end_date'].strftime('%Y-%m-%d')})"
        )


# Analyze consecutive hot days and tropical nights
find_consecutive_periods(df_knmi, "temp_max", 30.0, "hot days (>30°C)")
find_consecutive_periods(df_knmi, "temp_min", 20.0, "tropical nights (>20°C)")

print(df_knmi.head())

# %%
# Ensure the column "m_hh_ver" is numeric
gdf_merged["m_hh_ver"] = pd.to_numeric(gdf_merged["m_hh_ver"], errors="coerce")

# Drop rows with NaN values in "m_hh_ver" (if any)
gdf_merged = gdf_merged.dropna(subset=["m_hh_ver"])

# --- Plot average income per inhabitant ---
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Define the minimum and maximum values for the color scale
vmin = gdf_merged["m_hh_ver"].min()
vmax = gdf_merged["m_hh_ver"].max()

# Plot the data
gdf_merged.plot(column="m_hh_ver", cmap="OrRd", legend=False, ax=ax)

# Add a colorbar with a logical scale
sm = plt.cm.ScalarMappable(cmap="OrRd", norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm._A = []  # Required for ScalarMappable
cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
cbar.set_label("Gemiddeld inkomen per inwoner (€)", fontsize=12)

# Add title and remove axes
ax.set_title("Gemiddeld inkomen per inwoner (2023)", fontsize=14)
ax.axis("off")

plt.show()


# %%
# First, calculate the percentage of 65+ inhabitants
gdf_merged["perc_65"] = (gdf_merged["a_65_oo"] / gdf_merged["a_inw"]) * 100

# --- Plot percentage of 65+ inhabitants ---
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Define the minimum and maximum values for the color scale
vmin = gdf_merged["perc_65"].min()
vmax = gdf_merged["perc_65"].max()

# Plot the data with proper styling
gdf_merged.plot(
    column="perc_65", cmap="OrRd", legend=False, ax=ax, edgecolor="black", linewidth=0.5
)

# Add a colorbar with a logical scale
sm = plt.cm.ScalarMappable(cmap="OrRd", norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm._A = []  # Required for ScalarMappable
cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
cbar.set_label("Percentage 65+ inwoners (%)", fontsize=12)  # Updated label

# Add title and remove axes
ax.set_title("Percentage 65+ inwoners per wijk (2023)", fontsize=14)  # Updated title
ax.axis("off")

plt.show()

# %%
# Define coefficients (you can adjust these based on your model)
alpha = 0  # Intercept
beta = 1  # Coefficient for hot days (H_n)
gamma1 = 1  # Coefficient for %65+ (%65_n^+)
gamma2 = 1  # Coefficient for greenness (Greenness_n)
gamma3 = 1  # Coefficient for interaction H_n × %65_n^+
gamma4 = 1  # Coefficient for interaction H_n × Greenness_n

# Convert hot_days to a numeric count
hot_days_count = len(hot_days)  # Number of hot days

# Calculate the regression model
# log(μ_n) = α + β*H_n + γ1*%65_n^+ + γ2*Greenness_n + γ3*(H_n × %65_n^+) + γ4*(H_n × Greenness_n) + log(Pop_n)
gdf_merged["log_mu"] = (
    alpha
    + beta * hot_days_count
    + gamma1 * gdf_merged["perc_65"]
    + gamma2 * gdf_merged["green_percentage"]
    + gamma3 * (hot_days_count * gdf_merged["perc_65"])
    + gamma4 * (hot_days_count * gdf_merged["green_percentage"])
    + np.log(gdf_merged["a_inw"])
)

# Convert back to expected number of deaths (μ_n)
gdf_merged["expected_deaths"] = np.exp(gdf_merged["log_mu"])

# Optional: Calculate risk relative to population
gdf_merged["deaths_per_1000"] = (
    gdf_merged["expected_deaths"] / gdf_merged["a_inw"]
) * 1000

print(f"Number of hot days used in calculation: {hot_days_count}")

# %%
# --- Plot percentage of 65+ inhabitants ---
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Define the minimum and maximum values for the color scale
vmin = gdf_merged["deaths_per_1000"].min()
vmax = gdf_merged["deaths_per_1000"].max()

# Plot the data with proper styling
gdf_merged.plot(
    column="deaths_per_1000",
    cmap="OrRd",
    legend=False,
    ax=ax,
    edgecolor="black",
    linewidth=0.5,
)

# Add a colorbar with a logical scale
sm = plt.cm.ScalarMappable(cmap="OrRd", norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm._A = []  # Required for ScalarMappable
cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
cbar.set_label(
    "Aantal risico volle ouderen per 1000 inwoners", fontsize=12
)  # Updated label

# Add title and remove axes
ax.set_title(
    "Aantal risico volle ouderen per 1000 inwoners", fontsize=14
)  # Updated title
ax.axis("off")

plt.show()

# %%
