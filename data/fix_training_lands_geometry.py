"""Fix invalid geometries in training-lands.geojson."""

import geopandas as gpd
from shapely.geometry import mapping
from pathlib import Path
import json

data = Path(__file__).parent

# Read the GeoJSON file
gdf = gpd.read_file(data / 'training-lands.geojson')

print(f"Original features: {len(gdf)}")
print(f"Valid geometries: {gdf.geometry.is_valid.sum()}")
print(f"Invalid geometries: {(~gdf.geometry.is_valid).sum()}")

# Check for invalid geometries
invalid_mask = ~gdf.geometry.is_valid
if invalid_mask.any():
    print(f"\nFixing {invalid_mask.sum()} invalid geometries...")
    # Fix invalid geometries using buffer(0) trick
    gdf.loc[invalid_mask, 'geometry'] = gdf.loc[invalid_mask, 'geometry'].buffer(0)
    
    # Check again
    still_invalid = ~gdf.geometry.is_valid
    if still_invalid.any():
        print(f"Warning: {still_invalid.sum()} geometries still invalid after buffer(0)")
        # Try make_valid if available (shapely 2.0+)
        try:
            gdf.loc[still_invalid, 'geometry'] = gdf.loc[still_invalid, 'geometry'].make_valid()
        except AttributeError:
            print("make_valid() not available, trying alternative fix...")
            # Alternative: use unary_union and then reconstruct
            for idx in gdf[still_invalid].index:
                geom = gdf.loc[idx, 'geometry']
                if geom.is_valid:
                    continue
                # Try to fix by simplifying
                try:
                    fixed = geom.buffer(0.0001).buffer(-0.0001)
                    if fixed.is_valid:
                        gdf.loc[idx, 'geometry'] = fixed
                except:
                    print(f"Could not fix geometry at index {idx}")

# Union all geometries into a single MultiPolygon
print("\nUnioning all geometries...")
union_geom = gdf.geometry.union_all()

# Force 2D coordinates (remove z-coordinates)
from shapely.ops import transform
import pyproj

# Create a transformer to remove z-coordinates
def remove_z(geom):
    """Remove z-coordinates from geometry."""
    return transform(lambda x, y, z=None: (x, y), geom)

union_geom_2d = remove_z(union_geom)

# Create a new GeoDataFrame with the unioned geometry
fixed_gdf = gpd.GeoDataFrame(
    [{'geometry': union_geom_2d}],
    crs='EPSG:4326'  # Use standard EPSG code instead of CRS84
)

print(f"Final geometry valid: {fixed_gdf.geometry.is_valid.iloc[0]}")
print(f"Final geometry type: {type(union_geom_2d).__name__}")

# Save as GeoJSON without CRS field (Earth Engine prefers this)
output_path = data / 'training-lands.geojson'

# Create a clean GeoJSON structure
geojson_dict = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {},
            "geometry": mapping(union_geom_2d)
        }
    ]
}

# Write to file
with open(output_path, 'w') as f:
    json.dump(geojson_dict, f)

print(f"\nFixed geometry saved to: {output_path}")

# Verify it can be loaded
with open(output_path, 'r') as f:
    geojson_data = json.load(f)
    
print(f"GeoJSON type: {geojson_data['type']}")
print(f"Features: {len(geojson_data['features'])}")
print(f"Geometry type: {geojson_data['features'][0]['geometry']['type']}")
print(f"Has CRS field: {'crs' in geojson_data}")
print(f"First coordinate (should be 2D): {geojson_data['features'][0]['geometry']['coordinates'][0][0][0]}")

