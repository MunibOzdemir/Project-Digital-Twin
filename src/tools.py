from pathlib import Path

def get_geojson_path(filename):
    # Define the path to the JSON file
    current_dir = Path(__file__).resolve().parent
    data_file_path = current_dir.parent / 'data' / filename

    return data_file_path

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
        return data_dir / filename
    
    # If 'exclude_ndvi' is True, return a .tif file without 'ndvi' in its name
    if not ndvi:
        for file in data_dir.iterdir():
            if file.suffix.lower() == '.tif' and 'ndvi' not in file.name.lower():
                return file
        raise FileNotFoundError("No .tif file without 'ndvi' in the name was found.")
    
    # Default behavior: if ndvi=True, return 'ndvi.tif'
    return data_dir / 'ndvi.tif'