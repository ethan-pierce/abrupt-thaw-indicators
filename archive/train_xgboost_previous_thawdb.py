"""Use cross-validation to find the best parameters for an XGBoost model."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, brier_score_loss, f1_score, roc_curve, precision_recall_curve, PrecisionRecallDisplay, confusion_matrix, ConfusionMatrixDisplay

data = Path(__file__).parent.parent / 'data'
feats = pd.read_csv(data / 'features_clean.csv')
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

print(y_train.value_counts())

dtrain = xgb.DMatrix(X_train, label = y_train, missing = np.nan)
dtest = xgb.DMatrix(X_test, label = y_test, missing = np.nan)

# Best fit: n_estimators = 100, max_depth = 9, learning_rate = 0.1, subsample = 0.75, min_child_weight = 5 
# Score = 0.8794
# param_grid = {
#     'n_estimators': [100, 500, 1000],
#     'max_depth': [3, 6, 9],
#     'learning_rate': [0.01, 0.1, 0.3],
#     'subsample': [0.5, 0.75, 1.0],
#     'min_child_weight': [1, 3, 5]
# }

# Best fit: n_estimators = 100, max_depth = 9, learning_rate = 0.1, subsample = 0.75, min_child_weight = 5
# Score = 0.8794
# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'max_depth': [9, 12, 15],
#     'learning_rate': [0.05, 0.1, 0.15],
#     'subsample': [0.6, 0.75, 0.9],
#     'min_child_weight': [5, 10, 20]
# }

# Best fit: n_estimators = 125, max_depth = 9, learning_rate = 0.1, subsample = 0.8, min_child_weight = 5
# Score = 0.8818
# param_grid = {
#     'n_estimators': [75, 100, 125],
#     'max_depth': [8, 9, 10],
#     'learning_rate': [0.075, 0.1, 0.125],
#     'subsample': [0.7, 0.75, 0.8],
#     'min_child_weight': [3, 5, 7]
# }

# Best fit: n_estimators = 125, max_depth = 9, learning_rate = 0.1, subsample = 0.8, min_child_weight = 5
# Score = 0.8818
# param_grid = {
#     'n_estimators': [100, 125, 150],
#     'max_depth': [9],
#     'learning_rate': [0.09, 0.1, 0.11],
#     'subsample': [0.75, 0.8, 0.85],
#     'min_child_weight': [5]
# }

param_grid = {
    'n_estimators': [125],
    'max_depth': [9],
    'learning_rate': [0.1],
    'subsample': [0.8],
    'min_child_weight': [5]
}

xgb = xgb.XGBClassifier(
    random_state = rng.integers(0, 100),
    objective = 'binary:logistic',
    eval_metric = 'aucpr',
    scale_pos_weight = 30,
    base_score = 0.94
)

grid = GridSearchCV(
    estimator = xgb,
    param_grid = param_grid,
    cv = 5,
    scoring = 'f1',
    verbose = 3,
    n_jobs = 8,
    return_train_score = True # Disable for many jobs
)

fit = grid.fit(X_train, y_train)

print(f"Best parameters: {grid.best_params_}")
print(f"Best score: {grid.best_score_:.4f}")



# Get feature importance from best model
best_model = grid.best_estimator_
feature_importance = best_model.feature_importances_
feature_names = X_train.columns

PR = PrecisionRecallDisplay.from_estimator(
    best_model, X_test, y_test, name = 'XGBoost',
    plot_chance_level = False,
)
plt.show()

# Plot confusion matrix
y_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
# Convert to percentages
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
disp = ConfusionMatrixDisplay(confusion_matrix = cm_percent, display_labels=['Abrupt Thaw', 'Gradual Thaw'])
fig, ax = plt.subplots(figsize = (8, 6))
disp.plot(ax = ax, cmap = 'Blues', values_format = '.1f')
plt.title('Confusion Matrix - XGBoost Model (Percentages)')
plt.show()

# Create DataFrame for plotting
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
})

# Sort by importance
importance_df = importance_df.sort_values('importance', ascending=False)

# Plot top 20 features
plt.figure(figsize=(12, 8))
top_features = importance_df.head(20)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Feature Importance')
plt.title('Top 20 Feature Importances (XGBoost)')
plt.gca().invert_yaxis()  # Show most important at top
plt.tight_layout()
plt.show()

best_model.save_model('models/model.json')