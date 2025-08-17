import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from pathlib import Path

# Import the functions from your tools.py
try:
    from tools import get_geojson_path, get_tif_path

    print("Successfully imported functions from tools.py")
except ImportError:
    print("Could not import from tools.py, using local function")

    # Fallback local function if import fails
    def get_geojson_path(filename):
        """
        Get the full path to a GeoJSON file in the data directory
        Following the same pattern as your tools.py
        """
        current_dir = Path(__file__).resolve().parent
        data_file_path = current_dir.parent / "data" / filename

        if not data_file_path.exists():
            raise FileNotFoundError(
                f"GeoJSON file '{filename}' not found at: {data_file_path}"
            )

        print(f"GeoJSON file path: {data_file_path}")
        return str(data_file_path)


def plot_geojson_comparison(
    geojson_filenames,
    column_names,
    titles,
    colormap="RdYlGn_r",
    figsize=(20, 8),
    colorbar_label="Risk per 1000 inhabitants",
):
    """
    Plot multiple GeoJSON files side by side with data-driven coloring

    Parameters:
    -----------
    geojson_filenames : list
        List of GeoJSON filenames (just the filename, not full path)
    column_names : list
        List of column names to use for coloring each plot
    titles : list
        List of titles for each subplot
    colormap : str
        Matplotlib colormap name (default: 'RdYlGn_r')
    figsize : tuple
        Figure size (width, height)
    colorbar_label : str
        Label for the colorbar
    """

    # Load all GeoJSON files using get_geojson_path
    gdfs = []
    for filename in geojson_filenames:
        file_path = get_geojson_path(filename)
        gdf = gpd.read_file(file_path)
        gdfs.append(gdf)

    # Create subplots
    n_plots = len(geojson_filenames)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)

    # Handle single plot case
    if n_plots == 1:
        axes = [axes]

    # Find global min/max for consistent color scale
    all_values = []
    for i, gdf in enumerate(gdfs):
        values = gdf[column_names[i]].dropna()
        all_values.extend(values)

    vmin = min(all_values)
    vmax = max(all_values)

    # Plot each GeoJSON
    for i, (gdf, column, title, ax) in enumerate(zip(gdfs, column_names, titles, axes)):
        gdf.plot(
            column=column,
            cmap=colormap,
            legend=False,
            ax=ax,
            edgecolor="black",
            linewidth=0.5,
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_title(title, fontsize=14)
        ax.axis("off")

    # Add shared colorbar
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    cbar = fig.colorbar(
        sm, ax=axes, orientation="vertical", fraction=0.03, pad=0.1, shrink=0.8
    )
    cbar.set_label(colorbar_label, fontsize=12)

    plt.tight_layout()
    plt.show()


def plot_single_geojson(
    geojson_filename,
    column_name,
    title,
    colormap="RdYlGn_r",
    figsize=(10, 8),
    colorbar_label="Value",
):
    """
    Plot a single GeoJSON file with data-driven coloring

    Parameters:
    -----------
    geojson_filename : str
        GeoJSON filename (just the filename, not full path)
    column_name : str
        Column name to use for coloring
    title : str
        Plot title
    colormap : str
        Matplotlib colormap name
    figsize : tuple
        Figure size (width, height)
    colorbar_label : str
        Label for the colorbar
    """

    # Get the full path using get_geojson_path
    file_path = get_geojson_path(geojson_filename)

    # Load GeoJSON
    gdf = gpd.read_file(file_path)

    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Get min/max for color scale
    values = gdf[column_name].dropna()
    vmin = values.min()
    vmax = values.max()

    # Plot
    gdf.plot(
        column=column_name,
        cmap=colormap,
        legend=False,
        ax=ax,
        edgecolor="black",
        linewidth=0.5,
        vmin=vmin,
        vmax=vmax,
    )

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
    cbar.set_label(colorbar_label, fontsize=12)

    # Style
    ax.set_title(title, fontsize=14)
    ax.axis("off")

    plt.show()


# Test the import and functions
print("Functions defined successfully!")
print("You can now use:")
print("- plot_single_geojson(filename, column_name, title)")
print("- plot_geojson_comparison(filenames_list, columns_list, titles_list)")

# Example usage (uncomment to run):

# Single plots
plot_single_geojson(
    "risk_original_scenario.geojson",
    "risk_per_1000_inhabitants",
    "Origineel scenario\\nRisico overleiden ouderen per 1000 inwoners",
    colorbar_label="Risico aantal doden per 1000 inwoners",
)

plot_single_geojson(
    "risk_plus10_green_scenario.geojson",
    "risk_per_1000_inhabitants",
    "10% meer groen\\nRisico overleiden ouderen per 1000 inwoners",
    colorbar_label="Risico aantal doden per 1000 inwoners",
)

# Comparison plot
plot_geojson_comparison(
    geojson_filenames=[
        "risk_original_scenario.geojson",
        "risk_plus10_green_scenario.geojson",
    ],
    column_names=["risk_per_1000_inhabitants", "risk_per_1000_inhabitants"],
    titles=[
        "Origineel scenario\\nRisico overleiden ouderen per 1000 inwoners",
        "10% meer groen\\nRisico overleiden ouderen per 1000 inwoners",
    ],
    colorbar_label="Risico aantal doden per 1000 inwoners",
)

# Risk reduction plot
plot_single_geojson(
    "risk_comparison_scenarios.geojson",
    "risk_reduction",
    "Risico reductie door 10% meer groen\\nVerschil in doden per 1000 inwoners",
    colormap="RdYlGn",
    colorbar_label="Reductie in risico per 1000 inwoners",
)
