"""Test that the fixed training-lands.geojson works with Earth Engine."""

import json
import geemap
import ee
from pathlib import Path

ee.Initialize(project='ee-abrupt-thaw')

data_dir = Path(__file__).parent

with open(data_dir / 'training-lands.geojson', 'r') as f:
    data = json.load(f)

print(f"GeoJSON type: {data['type']}")
print(f"Features: {len(data['features'])}")
print(f"Geometry type: {data['features'][0]['geometry']['type']}")

# Convert to Earth Engine
geom_ee = geemap.geojson_to_ee(data)
print(f"\nEarth Engine type: {type(geom_ee)}")

# If it's a FeatureCollection, get geometry from first feature
if isinstance(geom_ee, ee.FeatureCollection):
    # Get the first feature and extract its geometry
    first_feature = ee.Feature(geom_ee.first())
    geom_ee = first_feature.geometry()
    print("Extracted geometry from first feature in FeatureCollection")

# Check validity
is_valid = geom_ee.isValid().getInfo()
print(f"Is valid: {is_valid}")

if is_valid:
    area = geom_ee.area(maxError=1000).getInfo()
    print(f"Area: {area:,.0f} m² ({area/1e6:.2f} km²)")
    print("\n✓ Geometry is valid and ready to use!")
else:
    print("\n✗ Geometry is still invalid!")

