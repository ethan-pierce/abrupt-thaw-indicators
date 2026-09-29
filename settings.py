from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
MODELS = ROOT / 'models'
OUTPUT = ROOT / 'output'

# Compute project passed to ee.Initialize(project=...).
EE_PROJECT = 'abrupt-thaw-indicators'