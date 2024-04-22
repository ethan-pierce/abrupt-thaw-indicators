"""Compile points from the Thaw Database and control point lists into one table."""

import os
import numpy as np
import pandas as pd
from settings import DATA

thawdb = pd.read_csv(os.path.join(DATA, 'ThawDatabase06222023.csv'))