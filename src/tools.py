from pathlib import Path
import geopandas as gpd
import os
from datetime import datetime




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

def get_tif_path(filename=None, ndvi=True, most_recent=True, year=None, season=None, show_files=False):
    """
    Advanced version with additional filtering options.
    
    Parameters:
    - filename: If provided, return that specific file path
    - ndvi: If True, look for files with 'ndvi' in name; if False, exclude 'ndvi' files
    - most_recent: If True, return the most recently modified file; if False, return first found
    - year: Filter by year in filename (e.g., 2019, 2022)
    - season: Filter by season in filename (e.g., 'spring', 'summer', 'autumn', 'winter')
    - show_options: If True, print all available files before selecting
    
    Returns:
    - str: Path to the .tif file
    """
    current_dir = Path(__file__).resolve().parent
    data_dir = current_dir.parent / 'data'
    
    # If a specific filename is provided, return its path
    if filename:
        file_path = data_dir / filename
        if file_path.exists():
            return str(file_path)
        else:
            raise FileNotFoundError(f"Specified file '{filename}' not found in data directory.")
    
    # Find all .tif files
    all_tif_files = [f for f in data_dir.iterdir() if f.suffix.lower() == '.tif']

    if show_files:
        print(f"All .tif files in data directory:")
        for i, file in enumerate(all_tif_files, 1):
            mod_time = datetime.fromtimestamp(file.stat().st_mtime)
            size = file.stat().st_size
            print(f"  {i}. {file.name} (modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}, size: {size:,} bytes)")
        print()
    
    # Apply filters
    filtered_files = []
    
    for file in all_tif_files:
        filename_lower = file.name.lower()
        
        # NDVI filter
        if ndvi and 'ndvi' not in filename_lower:
            continue
        if not ndvi and 'ndvi' in filename_lower:
            continue
        
        # Year filter
        if year and str(year) not in file.name:
            continue
        
        # Season filter
        if season and season.lower() not in filename_lower:
            continue
        
        filtered_files.append(file)
    
    # Check if any files were found
    if not filtered_files:
        filters = []
        if ndvi:
            filters.append("with 'ndvi'")
        else:
            filters.append("without 'ndvi'")
        if year:
            filters.append(f"from year {year}")
        if season:
            filters.append(f"from {season} season")
        
        filter_desc = " and ".join(filters) if filters else "matching criteria"
        raise FileNotFoundError(f"No .tif file {filter_desc} was found.")
    
    # Return based on most_recent parameter
    if most_recent:
        # Sort by modification time (most recent first)
        filtered_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        selected_file = filtered_files[0]
        print(f"Selected most recent .tif file: {selected_file.name}")
    else:
        # Return first found file (or you could sort alphabetically)
        filtered_files.sort(key=lambda x: x.name)  # Alphabetical sort
        selected_file = filtered_files[0]
        print(f"Selected first .tif file (alphabetically): {selected_file.name}")
    
    mod_time = datetime.fromtimestamp(selected_file.stat().st_mtime)
    print(f"File details: modified {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    return str(selected_file)

