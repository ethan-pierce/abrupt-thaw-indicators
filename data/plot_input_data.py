"""Plot the Alaska 2k curvature dataset as a map."""

import ee
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from settings import EE_PROJECT
import gee_features

ee.Initialize(project=EE_PROJECT)

# Re-derived inline from 3DEP (no custom asset; see gee_features.mean_curvature).
curve2k = gee_features.mean_curvature(2000).select('MeanCurvature').reproject('EPSG:4326', scale=10000)
alaska = ee.Geometry.Rectangle([-180, 51, -130, 72])

image = curve2k.sampleRectangle(region=alaska, defaultValue=-9999)
data = np.array(image.get('MeanCurvature').getInfo(), dtype=float)
data[data == -9999] = np.nan
lon = np.array(image.get('lon').getInfo(), dtype=float)
lat = np.array(image.get('lat').getInfo(), dtype=float)

fig, ax = plt.subplots(figsize=(14, 10))
im = ax.imshow(np.flipud(data), extent=[-180, -130, 51, 72], cmap='RdYlBu_r', aspect='auto', origin='lower')
plt.colorbar(im, ax=ax, label='Mean Curvature (2 km)')
ax.set_xlabel('Longitude (°E)')
ax.set_ylabel('Latitude (°N)')
ax.set_title('Alaska 2 km Curvature Dataset')
ax.grid(True, alpha=0.3)

output_path = Path(__file__).parent.parent / 'output' / 'alaska_curvature_2k_map.png'
output_path.parent.mkdir(exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()

