import os
import sys
from pathlib import Path

def setup_gdal_environment():
    """Set up GDAL environment variables for the project."""
    # Get project root (parent of scripts folder)
    project_root = Path(__file__).parent.parent.absolute()
    gdal_dir = project_root / 'gdal'
    
    if not gdal_dir.exists():
        print(f"ERROR: GDAL directory not found at {gdal_dir}")
        print("Please run the installation script first.")
        return False
    
    # Set environment variables
    os.environ['GDAL_DATA'] = str(gdal_dir / 'data')
    os.environ['PROJ_LIB'] = str(gdal_dir / 'projlib')
    
    # Add to PATH
    gdal_bin = str(gdal_dir / 'bin')
    if gdal_bin not in os.environ.get('PATH', ''):
        os.environ['PATH'] = gdal_bin + os.pathsep + os.environ.get('PATH', '')
    
    print("✅ GDAL environment configured successfully")
    return True

if __name__ == "__main__":
    setup_gdal_environment()