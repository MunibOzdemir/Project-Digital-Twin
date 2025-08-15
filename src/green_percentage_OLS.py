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
warnings.filterwarnings('ignore')

# Path to your GeoJSON files
GEOJSON_PATH =  get_geojson_path('alkmaar.geojson')

# Read GeoJSON and extract correct geometry
with open(GEOJSON_PATH) as f:
    gj = json.load(f)

geom = gj["features"][0]["geometry"] if "features" in gj else gj.get("geometry", gj)


def get_bounds_from_geometry(geometry):
    """Extract bounding box from GeoJSON geometry."""
    if geometry['type'] == 'Polygon':
        coords = geometry['coordinates'][0]
    elif geometry['type'] == 'MultiPolygon':
        coords = []
        for poly in geometry['coordinates']:
            coords.extend(poly[0])
    else:
        raise ValueError(f"Unsupported geometry type: {geometry['type']}")
    
    lons = [coord[0] for coord in coords]
    lats = [coord[1] for coord in coords]
    
    return (min(lons), min(lats), max(lons), max(lats))


# %% [markdown]
# # Using PDOK data to measure green

# %%
def convert_gml_to_geojson_and_clip_robust(gml_path, alkmaar_geojson_path, output_path=None):
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
            print("⚠️  GML file has no CRS defined. Attempting to detect or assign CRS...")
            
            # Check the coordinate values to guess the CRS
            sample_bounds = gml_gdf.total_bounds
            print(f"   Sample bounds: {sample_bounds}")
            
            # If coordinates look like they're in the Netherlands and in meters, likely RD New (EPSG:28992)
            if (sample_bounds[0] > 10000 and sample_bounds[0] < 300000 and 
                sample_bounds[1] > 300000 and sample_bounds[1] < 700000):
                print("   Coordinates suggest Dutch RD New projection (EPSG:28992)")
                gml_gdf.set_crs('EPSG:28992', inplace=True)
            # If coordinates look like lat/lon
            elif (sample_bounds[0] > -180 and sample_bounds[0] < 180 and
                  sample_bounds[1] > -90 and sample_bounds[1] < 90):
                print("   Coordinates suggest WGS84 (EPSG:4326)")
                gml_gdf.set_crs('EPSG:4326', inplace=True)
            else:
                print("   ❌ Cannot determine CRS automatically. Assuming EPSG:28992 (Dutch standard)")
                gml_gdf.set_crs('EPSG:28992', inplace=True)
        
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
        print(f"✅ Clipping complete: {len(clipped_gdf)} features remain after clipping")
        
        # Save to GeoJSON if output path provided
        if output_path:
            print(f"💾 Saving to: {output_path}")
            clipped_gdf.to_file(output_path, driver='GeoJSON')
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
        temp_geojson = str(gml_path).replace('.gml', '_temp.geojson')
        
        # If no CRS, try to assign one based on data inspection
        if gml_gdf.crs is None:
            sample_bounds = gml_gdf.total_bounds
            print(f"   Inspecting coordinates: {sample_bounds}")
            
            # Dutch data is often in RD New (EPSG:28992)
            if (sample_bounds[0] > 10000 and sample_bounds[1] > 300000):
                print("   Assigning EPSG:28992 (Dutch RD New)")
                gml_gdf.set_crs('EPSG:28992', inplace=True)
            else:
                print("   Assigning EPSG:4326 (WGS84)")
                gml_gdf.set_crs('EPSG:4326', inplace=True)
        
        # Save as temporary GeoJSON
        gml_gdf.to_file(temp_geojson, driver='GeoJSON')
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
            clipped_gdf.to_file(output_path, driver='GeoJSON')
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
    if gml_filename.startswith('/'):
        gml_path = Path(gml_filename)
        data_dir = gml_path.parent
    else:
        current_dir = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
        data_dir = current_dir.parent / 'data'
        gml_path = data_dir / gml_filename
    
    alkmaar_path = get_geojson_path('alkmaar.geojson')
    
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
GML_GREEN = get_geojson_path('top10nl_terrein.gml')

# Convert your specific GML file with robust handling
result = convert_and_clip_with_auto_paths_robust(
    GML_GREEN, 
    "top10nl_terrein_alkmaar_clipped.geojson"
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
    green_keywords = ['grasland', 'bos: loofbos', 'dodenakker', 'bos: griend',
                     'boomgaard', 'akkerland', 'fruitkwekerij', 'boomkwekerij']
    
    green_areas = []
    for col in ['typeLandgebruik', 'fysiekVoorkomen']:
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
    green_gdf = green_gdf.drop_duplicates(subset=['gml_id'])
    
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
        "Vroonermeer"
        ]

    #drop wijken from dataframes
    wijken_gdf = wijken_gdf[~wijken_gdf['naam'].isin(drop_names)].copy()
    
    # Ensure same CRS
    if wijken_gdf.crs != green_gdf.crs:
        wijken_gdf = wijken_gdf.to_crs(green_gdf.crs)
    
    # Find name column
    name_col = None
    for col in ['naam', 'wijknaam', 'buurtnaam', 'gebiedsnaam']:
        if col in wijken_gdf.columns:
            name_col = col
            break
    
    # Calculate green percentage for each wijk
    results = []
    for idx, wijk_row in wijken_gdf.iterrows():
        wijk_name = str(wijk_row[name_col]) if name_col and pd.notna(wijk_row[name_col]) else f'Wijk_{idx}'
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
                    if hasattr(intersection, 'area'):
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
        
        results.append({
            'wijk_name': wijk_name,
            'green_percentage': green_percentage,
            'total_area_ha': wijk_area_ha,
            'green_area_ha': green_area_ha,
            'green_features_count': green_features,
            'geometry': wijk_geometry
        })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Create only the two visualizations you want
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 1. Green areas with wijk boundaries
    wijken_gdf.plot(ax=ax1, facecolor='none', edgecolor='black', linewidth=1)
    green_gdf.plot(ax=ax1, color='green', edgecolor='darkgreen', linewidth=0.2, alpha=0.7)
    ax1.set_title('Green Areas with Wijk Boundaries', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # 2. Green percentage per wijk
    wijken_with_results = wijken_gdf.copy()
    wijken_with_results['green_percentage'] = results_df['green_percentage'].values
    
    if wijken_with_results['green_percentage'].max() > 0:
        wijken_with_results.plot(column='green_percentage', cmap='RdYlGn', 
                               legend=True, ax=ax2, edgecolor='black', linewidth=0.5)
        ax2.set_title('Green Percentage per Wijk/Buurt', fontsize=14, fontweight='bold')
    else:
        wijken_with_results.plot(ax=ax2, facecolor='lightgray', edgecolor='black', linewidth=0.5)
        ax2.set_title('Wijken/Buurten (No Green Data)', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return results_df

# Load TOP10NL data
geojson_green = get_geojson_path('top10nl_terrein_alkmaar_clipped.geojson')

gdf = gpd.read_file(geojson_green)


# Execute the clean analysis
wijken_path = get_geojson_path('alkmaar_wijken_buurten.geojson')
# Run the analysis and get results
top10nl_green_results_fixed = calculate_green_percentage_per_wijk_clean(wijken_path, gdf)

# %%
top10nl_green_results_fixed

# %%
#pip install statsmodels
import statsmodels.api as sm

layer3_data = pd.read_csv(get_geojson_path('df_green_merge(6).csv'))
data_needed = layer3_data[['green_percentage', 'a_65_oo', 'g_hh_sti', 'p_hh_li', 'p_hh_lkk', 'm_hh_ver', 'a_lan_ha', 'p_ste']]

#change all commas in data_needed into dots
data_needed = data_needed.replace(',', '.', regex=True)

#every cell with only a '.' is an NA value
data_needed = data_needed.replace('^\\.$', np.nan, regex=True)

#change to numeric
data_needed = data_needed.apply(pd.to_numeric)


# Simple KNN Imputation for X
from sklearn.impute import KNNImputer

def simple_knn_imputation(X, n_neighbors=5, verbose=True):
    """
    Simple KNN imputation for missing values in X.
    
    Args:
        X: DataFrame with missing values
        n_neighbors: Number of neighbors to use for imputation
        verbose: Whether to print information
    
    Returns:
        DataFrame with missing values filled using KNN
    """
    
    if verbose:
        print("🔍 KNN IMPUTATION")
        print("=" * 30)
        print(f"Original shape: {X.shape}")
        print(f"Missing values before: {X.isnull().sum().sum()}")
        
        # Show missing values per column
        missing_by_col = X.isnull().sum()
        if missing_by_col.sum() > 0:
            print("\nMissing values per column:")
            for col in missing_by_col[missing_by_col > 0].index:
                pct = (missing_by_col[col] / len(X)) * 100
                print(f"  {col}: {missing_by_col[col]} ({pct:.1f}%)")
    
    # Create KNN imputer
    knn_imputer = KNNImputer(n_neighbors=n_neighbors)
    
    # Fit and transform the data
    X_filled_array = knn_imputer.fit_transform(X)
    
    # Convert back to DataFrame with original column names and index
    X_filled = pd.DataFrame(X_filled_array, columns=X.columns, index=X.index)
    
    if verbose:
        print(f"\n✅ KNN Imputation complete!")
        print(f"Missing values after: {X_filled.isnull().sum().sum()}")
        print(f"Used {n_neighbors} nearest neighbors for imputation")
        
        # Show some statistics about the changes
        print(f"\nData summary:")
        print(f"  Columns: {len(X.columns)}")
        print(f"  Rows: {len(X)}")
        print(f"  Total values imputed: {X.isnull().sum().sum()}")
    
    return X_filled

# Apply KNN imputation to your X dataframe
data_filled = simple_knn_imputation(data_needed, n_neighbors=5, verbose=True)

y = data_filled['p_ste']
X = data_filled.drop(columns=['p_ste'])
# Basic OLS Regression with Intercept
def run_ols_regression_analysis(X, y):
    """
    Run OLS regression with intercept and analyze results.
    
    Args:
        X: DataFrame with independent variables
        y: Series/DataFrame with dependent variable
    
    Returns:
        Fitted model and summary analysis
    """
    
    print("🔍 OLS REGRESSION ANALYSIS")
    print("=" * 50)
    
    # Step 1: Add constant (intercept) to X
    X_with_const = sm.add_constant(X)
    print(f"Variables in model: {list(X_with_const.columns)}")
    
    # Step 2: Fit the OLS model
    model = sm.OLS(y, X_with_const).fit()
    
    # Step 3: Display full regression results
    print("\n📊 FULL REGRESSION RESULTS:")
    print("=" * 50)
    print(model.summary())
    
    # Step 4: Extract and display significant variables
    print("\n🎯 SIGNIFICANCE ANALYSIS:")
    print("=" * 50)
    
    # Get coefficients, p-values, and confidence intervals
    results_df = pd.DataFrame({
        'Variable': model.params.index,
        'Beta_Coefficient': model.params.values,
        'Standard_Error': model.bse.values,
        'P_Value': model.pvalues.values,
        'Lower_CI': model.conf_int()[0].values,
        'Upper_CI': model.conf_int()[1].values
    })
    
    # Mark significant variables
    results_df['Significant_5%'] = results_df['P_Value'] < 0.05
    results_df['Significant_1%'] = results_df['P_Value'] < 0.01
    results_df['Significance_Stars'] = results_df['P_Value'].apply(
        lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    )
    
    print("Significance levels: *** p<0.001, ** p<0.01, * p<0.05")
    print("\nAll Variables (sorted by p-value):")
    print(results_df.sort_values('P_Value').round(4))
    
    # Step 5: Show only significant variables
    significant_vars = results_df[results_df['Significant_5%']].sort_values('P_Value')
    
    if len(significant_vars) > 0:
        print(f"\n✅ SIGNIFICANT VARIABLES (p < 0.05):")
        print("-" * 50)
        for _, row in significant_vars.iterrows():
            direction = "positive" if row['Beta_Coefficient'] > 0 else "negative"
            print(f"{row['Variable']}: β = {row['Beta_Coefficient']:.4f} {row['Significance_Stars']}")
            print(f"   P-value: {row['P_Value']:.4f}, {direction} relationship")
            print(f"   95% CI: [{row['Lower_CI']:.4f}, {row['Upper_CI']:.4f}]")
            print()
    else:
        print("❌ No variables are statistically significant at α = 0.05")
    
    # Step 6: Model performance metrics
    print("📈 MODEL PERFORMANCE:")
    print("-" * 30)
    print(f"R-squared: {model.rsquared:.4f}")
    print(f"Adjusted R-squared: {model.rsquared_adj:.4f}")
    print(f"F-statistic: {model.fvalue:.4f}")
    print(f"F-statistic p-value: {model.f_pvalue:.4f}")
    print(f"AIC: {model.aic:.2f}")
    print(f"BIC: {model.bic:.2f}")
    print(f"Number of observations: {model.nobs}")
    
    return model, results_df

# Run the regression analysis
model, results_df = run_ols_regression_analysis(X, y)

# Additional analysis: Create visualization of significant coefficients
def plot_significant_coefficients(results_df):
    """
    Plot significant coefficients with confidence intervals.
    """
    significant_vars = results_df[results_df['Significant_5%'] & (results_df['Variable'] != 'const')]
    
    if len(significant_vars) == 0:
        print("No significant variables to plot")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Sort by coefficient magnitude for better visualization
    significant_vars = significant_vars.reindex(
        significant_vars['Beta_Coefficient'].abs().sort_values(ascending=True).index
    )
    
    y_pos = range(len(significant_vars))
    
    # Plot coefficients with error bars
    plt.errorbar(
        significant_vars['Beta_Coefficient'], 
        y_pos,
        xerr=[
            significant_vars['Beta_Coefficient'] - significant_vars['Lower_CI'],
            significant_vars['Upper_CI'] - significant_vars['Beta_Coefficient']
        ],
        fmt='o', 
        capsize=5,
        capthick=2,
        markersize=8
    )
    
    # Add vertical line at zero
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    
    # Customize plot
    plt.yticks(y_pos, significant_vars['Variable'])
    plt.xlabel('Beta Coefficient')
    plt.title('Significant Variables with 95% Confidence Intervals')
    plt.grid(True, alpha=0.3)
    
    # Add significance stars
    for i, (_, row) in enumerate(significant_vars.iterrows()):
        plt.text(row['Beta_Coefficient'], i, f"  {row['Significance_Stars']}", 
                va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

# Visualize significant coefficients
plot_significant_coefficients(results_df)

# Interpretation helper function
def interpret_results(model, results_df, dependent_var_name="p_ste"):
    """
    Provide plain English interpretation of results.
    """
    print(f"\n📝 INTERPRETATION FOR {dependent_var_name.upper()}:")
    print("=" * 50)
    
    significant_vars = results_df[results_df['Significant_5%'] & (results_df['Variable'] != 'const')]
    
    print(f"Model explains {model.rsquared:.1%} of the variance in {dependent_var_name}")
    
    if len(significant_vars) > 0:
        print(f"\nSignificant predictors:")
        for _, row in significant_vars.iterrows():
            var_name = row['Variable']
            beta = row['Beta_Coefficient']
            p_val = row['P_Value']
            
            if beta > 0:
                direction = "increases"
            else:
                direction = "decreases"
            
            print(f"• {var_name}: For every 1-unit increase in {var_name}, "
                  f"{dependent_var_name} {direction} by {abs(beta):.4f} units "
                  f"(p = {p_val:.4f})")
    else:
        print("No variables significantly predict the outcome at α = 0.05")

# Get interpretation
interpret_results(model, results_df, "p_ste")

# %%
# PRIORITY AREA IDENTIFICATION USING OLS PREDICTIONS
def identify_priority_areas_for_intervention(model, X, y, data_filled):
    """
    Use the OLS model to predict death rates and identify priority areas
    for green space interventions based on elderly population and current green levels.
    
    Args:
        model: Fitted OLS model
        X: Independent variables
        y: Actual death rates
        data_filled: Complete dataset with all variables
    
    Returns:
        DataFrame with priority rankings and intervention recommendations
    """
    
    print("🎯 PRIORITY AREA IDENTIFICATION")
    print("=" * 60)
    
    # Step 1: Make predictions using the OLS model
    X_with_const = sm.add_constant(X)
    y_predicted = model.predict(X_with_const)
    
    # Step 2: Create comprehensive analysis DataFrame
    priority_df = data_filled.copy()
    priority_df['predicted_deaths'] = y_predicted
    priority_df['actual_deaths'] = y
    priority_df['prediction_error'] = y - y_predicted
    
    # Step 3: Calculate intervention potential scores
    # Higher scores = more potential benefit from green interventions
    
    # Normalize variables for scoring (0-1 scale)
    priority_df['elderly_score'] = (priority_df['a_65_oo'] - priority_df['a_65_oo'].min()) / (priority_df['a_65_oo'].max() - priority_df['a_65_oo'].min())
    priority_df['green_deficit_score'] = 1 - ((priority_df['green_percentage'] - priority_df['green_percentage'].min()) / (priority_df['green_percentage'].max() - priority_df['green_percentage'].min()))
    priority_df['death_rate_score'] = (priority_df['predicted_deaths'] - priority_df['predicted_deaths'].min()) / (priority_df['predicted_deaths'].max() - priority_df['predicted_deaths'].min())
    
    # Combined priority score (weighted)
    priority_df['intervention_priority_score'] = (
        0.4 * priority_df['elderly_score'] +        # 40% weight on elderly population
        0.4 * priority_df['green_deficit_score'] +  # 40% weight on green deficit
        0.2 * priority_df['death_rate_score']       # 20% weight on predicted death rate
    )
    
    # Step 4: Calculate potential intervention impact
    # Simulate increasing green space by 10% and see predicted death reduction
    X_intervention = X.copy()
    X_intervention['green_percentage'] = X_intervention['green_percentage'] + 10  # Add 10% green space
    X_intervention_const = sm.add_constant(X_intervention)
    y_predicted_after_intervention = model.predict(X_intervention_const)
    
    priority_df['potential_death_reduction'] = y_predicted - y_predicted_after_intervention
    priority_df['intervention_efficiency'] = priority_df['potential_death_reduction'] / 10  # Deaths reduced per 1% green increase
    
    # Step 5: Rank areas by priority
    priority_df = priority_df.sort_values('intervention_priority_score', ascending=False)
    priority_df['priority_rank'] = range(1, len(priority_df) + 1)
    
    # Step 6: Categorize priority levels
    n_areas = len(priority_df)
    priority_df['priority_category'] = pd.cut(
        priority_df['priority_rank'], 
        bins=[0, n_areas*0.2, n_areas*0.5, n_areas],
        labels=['HIGH PRIORITY', 'MEDIUM PRIORITY', 'LOW PRIORITY']
    )
    
    return priority_df

# Step 7: Generate intervention recommendations
def generate_intervention_recommendations(priority_df, top_n=10):
    """
    Generate specific recommendations for top priority areas.
    """
    print(f"\n🏥 TOP {top_n} PRIORITY AREAS FOR GREEN INTERVENTIONS")
    print("=" * 80)
    print("Areas ranked by combination of: High elderly population + Low green space + High predicted death rate")
    print()
    
    top_areas = priority_df.head(top_n)
    
    for i, (idx, row) in enumerate(top_areas.iterrows(), 1):
        print(f"🎯 RANK #{i} - AREA INDEX {idx}")
        print(f"   Elderly Population (65+): {row['a_65_oo']:.1f}%")
        print(f"   Current Green Space: {row['green_percentage']:.1f}%")
        print(f"   Predicted Death Rate: {row['predicted_deaths']:.2f}")
        print(f"   Priority Score: {row['intervention_priority_score']:.3f}")
        print(f"   Potential Death Reduction (10% more green): {row['potential_death_reduction']:.3f}")
        print(f"   Intervention Efficiency: {row['intervention_efficiency']:.4f} deaths reduced per 1% green")
        
        # Generate specific recommendations
        if row['green_percentage'] < 20:
            green_rec = "URGENT: Create new parks, green corridors, or urban forests"
        elif row['green_percentage'] < 40:
            green_rec = "MODERATE: Enhance existing green spaces, add pocket parks"
        else:
            green_rec = "MAINTENANCE: Improve quality of existing green infrastructure"
        
        if row['a_65_oo'] > 25:
            elderly_rec = "Focus on elderly-friendly green spaces (benches, accessible paths, health stations)"
        else:
            elderly_rec = "Multi-generational green spaces with diverse amenities"
        
        print(f"   🌱 Green Space Recommendation: {green_rec}")
        print(f"   👥 Target Population Focus: {elderly_rec}")
        print("-" * 80)
    
    return top_areas

# Run the priority identification
priority_results = identify_priority_areas_for_intervention(model, X, y, data_filled)
top_priority_areas = generate_intervention_recommendations(priority_results, top_n=10)

# %%
# Create simplified map showing potential death reduction by wijk
def create_priority_intervention_map_simple(layer3_data, priority_results):
    """
    Create a simplified map showing wijken colored by potential death reduction from green interventions.
    Only shows the two main maps without scatter plot and summary table.
    
    Args:
        layer3_data: Original dataframe with geometry_y column
        priority_results: DataFrame with priority analysis results
    
    Returns:
        Map visualization
    """
    
    print("🗺️ CREATING SIMPLIFIED PRIORITY INTERVENTION MAP")
    print("=" * 50)
    
    # Create a copy of the original data for mapping
    map_data = layer3_data.copy()
    
    # Add priority results to the map data
    map_data['potential_death_reduction'] = priority_results['potential_death_reduction'].values
    map_data['intervention_priority_score'] = priority_results['intervention_priority_score'].values
    map_data['priority_rank'] = priority_results['priority_rank'].values
    
    # Create the simplified map with just 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Check if geometry_y column exists and has valid geometries
    if 'geometry_y' in map_data.columns:
        # Convert WKT strings to actual geometry objects
        from shapely import wkt
        
        print("Converting WKT strings to geometry objects...")
        try:
            # Convert WKT strings to Shapely geometries
            map_data['geometry_shapely'] = map_data['geometry_y'].apply(lambda x: wkt.loads(x) if pd.notna(x) and isinstance(x, str) else None)
            
            # Convert to GeoDataFrame using the converted geometries
            gdf_map = gpd.GeoDataFrame(map_data, geometry='geometry_shapely')
            
            # Remove any invalid geometries
            gdf_map = gdf_map[gdf_map.geometry.notna()]
            
            if len(gdf_map) > 0:
                print(f"✅ Successfully converted {len(gdf_map)} geometries")
                
                # 1. Potential Death Reduction Map
                gdf_map.plot(column='potential_death_reduction', 
                            cmap='Reds', 
                            legend=True, 
                            ax=ax1, 
                            edgecolor='black', 
                            linewidth=0.8)
                ax1.set_title('Potential Death Reduction per Wijk\n(10% Green Space Increase)', 
                             fontsize=14, fontweight='bold')
                ax1.axis('off')
                
                # 2. Priority Score Map
                gdf_map.plot(column='intervention_priority_score', 
                            cmap='YlOrRd', 
                            legend=True, 
                            ax=ax2, 
                            edgecolor='black', 
                            linewidth=0.8)
                ax2.set_title('Intervention Priority Score per Wijk\n(Higher = More Priority)', 
                             fontsize=14, fontweight='bold')
                ax2.axis('off')
                
            else:
                ax1.text(0.5, 0.5, 'No valid geometries found', 
                        transform=ax1.transAxes, ha='center', va='center')
                ax1.set_title('Potential Death Reduction Map (No Data)')
                
                ax2.text(0.5, 0.5, 'No valid geometries found', 
                        transform=ax2.transAxes, ha='center', va='center')
                ax2.set_title('Priority Score Map (No Data)')
                
        except Exception as e:
            print(f"❌ Error converting geometries: {e}")
            ax1.text(0.5, 0.5, f'Error processing geometries:\n{str(e)}', 
                    transform=ax1.transAxes, ha='center', va='center')
            ax1.set_title('Potential Death Reduction Map (Error)')
            
            ax2.text(0.5, 0.5, f'Error processing geometries:\n{str(e)}', 
                    transform=ax2.transAxes, ha='center', va='center')
            ax2.set_title('Priority Score Map (Error)')
            gdf_map = None
    else:
        ax1.text(0.5, 0.5, 'geometry_y column not found', 
                transform=ax1.transAxes, ha='center', va='center')
        ax1.set_title('Potential Death Reduction Map (No Geometry)')
        
        ax2.text(0.5, 0.5, 'geometry_y column not found', 
                transform=ax2.transAxes, ha='center', va='center')
        ax2.set_title('Priority Score Map (No Geometry)')
        gdf_map = None
    
    plt.tight_layout()
    plt.show()
    
    return gdf_map

# Create the simplified priority intervention map
print("Creating simplified priority intervention map...")

# Check if geometry_y exists in layer3_data
if 'geometry_y' in layer3_data.columns:
    print(f"✅ Found geometry_y column with {layer3_data['geometry_y'].notna().sum()} valid geometries")
    priority_map = create_priority_intervention_map_simple(layer3_data, priority_results)
    
    # Export the map data for GIS use (if successful)
    if priority_map is not None and len(priority_map) > 0:
        output_path = "/Users/jelle/Library/CloudStorage/OneDrive-Personal/GitHub/Project-Digital-Twin/outputs/priority_intervention_map.geojson"
        
        # Create a clean export DataFrame with only the necessary columns
        export_columns = ['potential_death_reduction', 'intervention_priority_score', 
                         'priority_rank', 'geometry']
        
        # Make sure all columns exist before exporting
        available_columns = [col for col in export_columns if col in priority_map.columns]
        
        if 'geometry' in available_columns:
            try:
                priority_map[available_columns].to_file(output_path, driver='GeoJSON')
                print(f"✅ Priority intervention map exported to: {output_path}")
            except Exception as e:
                print(f"❌ Export failed: {e}")
        else:
            print("❌ Cannot export: geometry column missing from processed data")
    else:
        print("❌ No valid map data to export")
else:
    print("❌ geometry_y column not found in layer3_data")
    print("Available columns:", list(layer3_data.columns))

# %%
# # Simple VIF Test
# from statsmodels.stats.outliers_influence import variance_inflation_factor

# def simple_vif_test(X):
#     """Simple VIF calculation for multicollinearity detection."""
#     X_with_const = sm.add_constant(X)
    
#     vif_data = []
#     for i in range(X_with_const.shape[1]):
#         vif = variance_inflation_factor(X_with_const.values, i)
#         vif_data.append([X_with_const.columns[i], vif])
    
#     vif_df = pd.DataFrame(vif_data, columns=['Variable', 'VIF'])
#     return vif_df.sort_values('VIF', ascending=False)

# # Test VIF on your data
# print("🔍 VIF TEST (Variance Inflation Factor)")
# print("=" * 40)
# print("Rule of thumb: VIF > 5 = problematic, VIF > 10 = severe multicollinearity")
# print()

# vif_results = simple_vif_test(X)
# print(vif_results)

# # Identify problematic variables
# high_vif = vif_results[vif_results['VIF'] > 10]
# if len(high_vif) > 0:
#     print(f"\n❌ HIGH VIF VARIABLES (>10):")
#     for _, row in high_vif.iterrows():
#         print(f"   {row['Variable']}: {row['VIF']:.1f}")
# else:
#     print("\n✅ No severe multicollinearity detected")


