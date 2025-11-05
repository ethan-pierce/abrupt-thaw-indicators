"""Investigate the SHAP values of the best XGBoost model."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
import shap

data = Path(__file__).parent.parent / 'data'
feats = pd.read_csv(data / 'features_clean.csv')
rng = np.random.default_rng(100)

matrix = feats.drop('Class', axis = 1)
target = feats['Class']
X_train, X_test, y_train, y_test = train_test_split(
    matrix, target, 
    test_size = 0.2, 
    random_state = rng.integers(0, 100), 
    shuffle = True, 
    stratify = target
)
dtrain = xgb.DMatrix(X_train, label = y_train, missing = np.nan)
dtest = xgb.DMatrix(X_test, label = y_test, missing = np.nan)

model = xgb.XGBClassifier()
model.load_model('models/model.json')

# Calculate SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)

# Find indices for abrupt thaw (class 1) and gradual thaw (class 0)
abrupt_indices = np.where(y_test == 1)[0]
gradual_indices = np.where(y_test == 0)[0]

base_values_rounded = explainer.expected_value.round(2)
shap_values_rounded = shap_values.values.round(2)
X_test_rounded = X_test.round(2)


idx = abrupt_indices[100]

im = shap.plots.force(
    base_values_rounded,
    shap_values_rounded[idx],
    X_test_rounded.iloc[idx],
    contribution_threshold = 0.1,
    matplotlib = True,
    show = False,
    figsize = (12, 4)
)
plt.tight_layout()
plt.savefig('output/shap_force_plot_abrupt.png', dpi = 300)
plt.show()

idx = gradual_indices[50]
im = shap.plots.force(
    base_values_rounded,
    shap_values_rounded[idx],
    X_test_rounded.iloc[idx],
    contribution_threshold = 0.15,
    matplotlib = True,
    show = False,
    figsize = (12, 4)
)
plt.tight_layout()
plt.savefig('output/shap_force_plot_gradual.png', dpi = 300)
plt.show()

shap.summary_plot(shap_values, max_display = 10, show = False)
plt.tight_layout()
plt.savefig('output/shap_summary_plot.png', dpi = 300)
plt.show()

shap.dependence_plot(
    'Nitrogen (0-30 cm)', 
    explainer.shap_values(X_test), 
    X_test, 
    interaction_index = 'Nitrogen (30-200 cm)',
    show = False
)
plt.tight_layout()
plt.savefig('output/shap_dependence_plot_nitrogen.png', dpi = 300)
plt.show()

