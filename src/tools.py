from pathlib import Path

def get_geojson_path(filename):
    # Define the path to the JSON file
    current_dir = Path(__file__).resolve().parent
    data_file_path = current_dir.parent / 'data' / filename

    return data_file_path