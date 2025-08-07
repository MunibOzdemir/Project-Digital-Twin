from pathlib import Path
import geopandas as gpd
import os




def convert_geojson_to_wgs84(filename, input_path=None, output_path=None):
    """
    Convert a GeoJSON file to WGS84 coordinate reference system (EPSG:4326).
    """
    current_dir = Path(__file__).resolve().parent
    data_dir = current_dir.parent / 'data'
    
    # Load your geojson (with unknown CRS)
    gdf = gpd.read_file(data_dir / filename)

    if gdf.crs is None:
        raise ValueError("No CRS found in the file.")

    # Convert to WGS84 (EPSG:4326)
    gdf_wgs84 = gdf.to_crs(epsg=4326)

    if output_path is None:
        base, ext = os.path.splitext(data_dir / filename)
        output_path = f"{base}_wgs84{ext}"

    # Save the new geojson file
    gdf_wgs84.to_file(str(output_path), driver="GeoJSON")

    print("Conversion complete, new file saved as:", output_path)

def get_geojson_path(filename):
    current_dir = Path(__file__).resolve().parent
    data_file_path = current_dir.parent / 'data' / filename
    
    if not data_file_path.exists():
        raise FileNotFoundError(f"GeoJSON file '{filename}' not found at: {data_file_path}")
   
    print(f"GeoJSON file path: {data_file_path}")
    return str(data_file_path)

def get_tif_path(filename=None, ndvi=True):
    """
    Get the path to a .tif file in the 'data' directory.
    - If filename is provided, return that specific file path.
    - If ndvi is True, return the NDVI tif file path.
    - If exclude_ndvi is True, return a .tif file without 'ndvi' in its name.
    """
    current_dir = Path(__file__).resolve().parent
    data_dir = current_dir.parent / 'data'

    # If a specific filename is provided, return its path
    if filename:
        return str(data_dir / filename)

    # If 'exclude_ndvi' is True, return a .tif file without 'ndvi' in its name
    if not ndvi:
        for file in data_dir.iterdir():
            if file.suffix.lower() == '.tif' and 'ndvi' not in file.name.lower():
                return str(file)
        raise FileNotFoundError("No .tif file without 'ndvi' in the name was found.")
    
    # If ndvi=True, return a .tif file that contains 'ndvi' in its name
    for file in data_dir.iterdir():
        if file.suffix.lower() == '.tif' and 'ndvi' in file.name.lower():
            return str(file)
    
    # If no 'ndvi' .tif file is found, raise an error
    raise FileNotFoundError("No .tif file with 'ndvi' in the name was found.")

