from pathlib import Path

def get_geojson_path(filename):
    # Define the path to the JSON file
    current_dir = Path(__file__).resolve().parent
    data_file_path = current_dir.parent / 'data' / filename

    return data_file_path

def get_tif_path(filename=None, ndvi=True):
    """
    Get the path to the most recent .tif file in the 'data' directory.
    If filename is provided, return that specific file path.
    If ndvi is True, return the NDVI tif file path.
    """
    current_dir = Path(__file__).resolve().parent
    data_dir = current_dir.parent / 'data'
    if filename:
        return data_dir / filename
    if ndvi:
        return data_dir / 'ndvi.tif'
    return data_dir / 'latest.tif'