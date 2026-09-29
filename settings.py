from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
MODELS = ROOT / 'models'
OUTPUT = ROOT / 'output'

# Non-feature columns of features_clean.csv.
METADATA_COLUMNS = ('Class', 'Latitude', 'Longitude', 'Thaw Type', 'Thaw Database Row')

# Compute project passed to ee.Initialize(project=...).
EE_PROJECT = 'abrupt-thaw-indicators'