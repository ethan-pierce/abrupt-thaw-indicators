"""Investigate the SHAP values of the best XGBoost model."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
import shap

data = Path(__file__).parent.parent / 'data'
feats = pd.read_csv(data / 'features.csv')
rng = np.random.default_rng(42)

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

prediction = model.predict(X_test, output_margin = True)
explainer = shap.TreeExplainer(model, feature_names = X_train.columns)
explanation = explainer(dtest)
shap_values = explanation.values

# shap.plots.beeswarm(explanation, show = False, max_display = 10)
# plt.tight_layout()
# plt.show()

shap.plots.violin(
    shap_values, 
    features = X_test, 
    feature_names = X_train.columns, 
    plot_type = 'layered_violin', 
    show = False,
    max_display = 10
)
plt.tight_layout()
plt.show()

