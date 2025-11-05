"""Use cross-validation to find the best parameters for an XGBoost model."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, f1_score, roc_curve, precision_recall_curve, 
    PrecisionRecallDisplay, confusion_matrix, ConfusionMatrixDisplay, accuracy_score,
    precision_score, recall_score, classification_report, average_precision_score,
    log_loss, matthews_corrcoef, cohen_kappa_score, hamming_loss, jaccard_score
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import cross_val_score, cross_validate

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

param_grid = {
    'n_estimators': [80],
    'max_depth': [10],
    'learning_rate': [0.06],
    'subsample': [0.7],
    'min_child_weight': [2]
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
print(f"Best CV score: {grid.best_score_:.4f}")
print(f"CV train score: {grid.cv_results_['mean_train_score'][grid.best_index_]:.4f}")
print(f"CV test score: {grid.cv_results_['mean_test_score'][grid.best_index_]:.4f}")
print("\n" + "="*80)
print("COMPREHENSIVE MODEL METRICS")
print("="*80)

# Get the best model
best_model = grid.best_estimator_

# Get predictions
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)
y_train_proba = best_model.predict_proba(X_train)[:, 1]
y_test_proba = best_model.predict_proba(X_test)[:, 1]

# ============================================================================
# PERFORMANCE METRICS
# ============================================================================
print("\n" + "-"*80)
print("PERFORMANCE METRICS")
print("-"*80)

# Basic classification metrics
print("\n--- Classification Metrics (Test Set) ---")
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, zero_division=0)
test_recall = recall_score(y_test, y_test_pred, zero_division=0)
test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
test_f1_macro = f1_score(y_test, y_test_pred, average='macro', zero_division=0)
test_f1_weighted = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)

print(f"Accuracy:           {test_accuracy:.4f}")
print(f"Precision:          {test_precision:.4f}")
print(f"Recall (Sensitivity): {test_recall:.4f}")
print(f"F1-score:           {test_f1:.4f}")
print(f"F1-score (macro):   {test_f1_macro:.4f}")
print(f"F1-score (weighted): {test_f1_weighted:.4f}")

# Calculate specificity from confusion matrix
cm_test = confusion_matrix(y_test, y_test_pred)
if cm_test.shape == (2, 2):
    tn, fp, fn, tp = cm_test.ravel()
    test_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    print(f"Specificity:        {test_specificity:.4f}")
    print(f"True Positives:     {tp}")
    print(f"True Negatives:     {tn}")
    print(f"False Positives:    {fp}")
    print(f"False Negatives:    {fn}")

# Additional classification metrics
test_mcc = matthews_corrcoef(y_test, y_test_pred)
test_kappa = cohen_kappa_score(y_test, y_test_pred)
test_hamming = hamming_loss(y_test, y_test_pred)
test_jaccard = jaccard_score(y_test, y_test_pred, zero_division=0)

print(f"\nMatthews Corr Coef: {test_mcc:.4f}")
print(f"Cohen's Kappa:      {test_kappa:.4f}")
print(f"Hamming Loss:       {test_hamming:.4f}")
print(f"Jaccard Score:      {test_jaccard:.4f}")

# Probability-based metrics
print("\n--- Probability-Based Metrics (Test Set) ---")
test_auc_roc = roc_auc_score(y_test, y_test_proba)
test_auc_pr = average_precision_score(y_test, y_test_proba)
test_brier = brier_score_loss(y_test, y_test_proba)
test_log_loss = log_loss(y_test, y_test_proba)

print(f"AUC-ROC:            {test_auc_roc:.4f}")
print(f"AUC-PR (AP):        {test_auc_pr:.4f}")
print(f"Brier Score:        {test_brier:.4f}")
print(f"Log Loss:           {test_log_loss:.4f}")

# Classification report
print("\n--- Detailed Classification Report (Test Set) ---")
print(classification_report(y_test, y_test_pred, target_names=['Gradual Thaw', 'Abrupt Thaw']))

# ============================================================================
# ROBUSTNESS METRICS (Cross-Validation)
# ============================================================================
print("\n" + "-"*80)
print("ROBUSTNESS METRICS (Cross-Validation Stability)")
print("-"*80)

# Extract CV scores from grid search
cv_test_scores = []
cv_train_scores = []
for i in range(5):
    test_key = f'split{i}_test_score'
    train_key = f'split{i}_train_score'
    if test_key in grid.cv_results_:
        cv_test_scores.extend(grid.cv_results_[test_key].tolist())
    if train_key in grid.cv_results_:
        cv_train_scores.extend(grid.cv_results_[train_key].tolist())

if cv_test_scores:
    print(f"\n--- CV F1 Scores (5-fold) ---")
    print(f"Mean CV F1 (test):     {np.mean(cv_test_scores):.4f}")
    print(f"Std CV F1 (test):      {np.std(cv_test_scores):.4f}")
    print(f"Min CV F1 (test):      {np.min(cv_test_scores):.4f}")
    print(f"Max CV F1 (test):      {np.max(cv_test_scores):.4f}")
    print(f"Range CV F1 (test):    {np.max(cv_test_scores) - np.min(cv_test_scores):.4f}")
    print(f"CV F1 Scores:          {[f'{s:.4f}' for s in cv_test_scores]}")

if cv_train_scores:
    print(f"\nMean CV F1 (train):    {np.mean(cv_train_scores):.4f}")
    print(f"Std CV F1 (train):     {np.std(cv_train_scores):.4f}")
    if cv_test_scores:
        print(f"Overfitting gap:       {np.mean(cv_train_scores) - np.mean(cv_test_scores):.4f}")

# Additional cross-validation with multiple metrics
print("\n--- Cross-Validation with Multiple Metrics ---")
cv_metrics = cross_validate(
    best_model, X_train, y_train, 
    cv=5, 
    scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'average_precision'],
    return_train_score=True
)

for metric in ['test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_roc_auc', 'test_average_precision']:
    scores = cv_metrics[metric]
    print(f"{metric.replace('test_', '').replace('_', ' ').title():25s}: "
          f"mean={np.mean(scores):.4f}, std={np.std(scores):.4f}, "
          f"range=[{np.min(scores):.4f}, {np.max(scores):.4f}]")

# ============================================================================
# GENERALIZABILITY METRICS (Train vs Test Comparison)
# ============================================================================
print("\n" + "-"*80)
print("GENERALIZABILITY METRICS (Train vs Test Comparison)")
print("-"*80)

# Train metrics
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred, zero_division=0)
train_recall = recall_score(y_train, y_train_pred, zero_division=0)
train_f1 = f1_score(y_train, y_train_pred, zero_division=0)
train_auc_roc = roc_auc_score(y_train, y_train_proba)
train_auc_pr = average_precision_score(y_train, y_train_proba)
train_brier = brier_score_loss(y_train, y_train_proba)

print("\n--- Train vs Test Performance Gap ---")
print(f"{'Metric':<25s} {'Train':<12s} {'Test':<12s} {'Gap':<12s}")
print("-" * 65)
print(f"{'Accuracy':<25s} {train_accuracy:<12.4f} {test_accuracy:<12.4f} {train_accuracy - test_accuracy:<12.4f}")
print(f"{'Precision':<25s} {train_precision:<12.4f} {test_precision:<12.4f} {train_precision - test_precision:<12.4f}")
print(f"{'Recall':<25s} {train_recall:<12.4f} {test_recall:<12.4f} {train_recall - test_recall:<12.4f}")
print(f"{'F1-score':<25s} {train_f1:<12.4f} {test_f1:<12.4f} {train_f1 - test_f1:<12.4f}")
print(f"{'AUC-ROC':<25s} {train_auc_roc:<12.4f} {test_auc_roc:<12.4f} {train_auc_roc - test_auc_roc:<12.4f}")
print(f"{'AUC-PR':<25s} {train_auc_pr:<12.4f} {test_auc_pr:<12.4f} {train_auc_pr - test_auc_pr:<12.4f}")
print(f"{'Brier Score':<25s} {train_brier:<12.4f} {test_brier:<12.4f} {train_brier - test_brier:<12.4f}")

# Overfitting indicator
overfitting_warning = ""
if (train_accuracy - test_accuracy) > 0.1:
    overfitting_warning = "WARNING: Large accuracy gap suggests possible overfitting"
elif (train_f1 - test_f1) > 0.1:
    overfitting_warning = "WARNING: Large F1 gap suggests possible overfitting"
else:
    overfitting_warning = "Model appears to generalize well"

print(f"\n{overfitting_warning}")

# ============================================================================
# PREDICTION CONFIDENCE ANALYSIS
# ============================================================================
print("\n" + "-"*80)
print("PREDICTION CONFIDENCE ANALYSIS")
print("-"*80)

print(f"\n--- Probability Distribution Statistics ---")
print(f"Test predictions - Mean: {np.mean(y_test_proba):.4f}, Std: {np.std(y_test_proba):.4f}")
print(f"Test predictions - Min: {np.min(y_test_proba):.4f}, Max: {np.max(y_test_proba):.4f}")
print(f"Test predictions - Median: {np.median(y_test_proba):.4f}")
print(f"Test predictions - Q25: {np.percentile(y_test_proba, 25):.4f}, Q75: {np.percentile(y_test_proba, 75):.4f}")

# Confidence by class
print(f"\n--- Confidence by True Class ---")
for class_label, class_name in [(0, 'Gradual Thaw'), (1, 'Abrupt Thaw')]:
    mask = y_test == class_label
    if mask.sum() > 0:
        class_proba = y_test_proba[mask]
        print(f"{class_name}: Mean={np.mean(class_proba):.4f}, Std={np.std(class_proba):.4f}, "
              f"Min={np.min(class_proba):.4f}, Max={np.max(class_proba):.4f}")

# ============================================================================
# CALIBRATION ANALYSIS
# ============================================================================
print("\n" + "-"*80)
print("CALIBRATION ANALYSIS")
print("-"*80)

# Calculate calibration curve
fraction_of_positives, mean_predicted_value = calibration_curve(
    y_test, y_test_proba, n_bins=10, strategy='uniform'
)

print(f"\nCalibration curve (10 bins):")
for i, (frac, mean) in enumerate(zip(fraction_of_positives, mean_predicted_value)):
    print(f"  Bin {i+1}: Predicted={mean:.3f}, Actual={frac:.3f}, Gap={abs(mean-frac):.3f}")

mean_calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
print(f"\nMean Calibration Error: {mean_calibration_error:.4f}")

# ============================================================================
# DATA SUMMARY
# ============================================================================
print("\n" + "-"*80)
print("DATA SUMMARY")
print("-"*80)

print(f"\nTotal samples:        {len(feats)}")
print(f"Training samples:    {len(X_train)} ({len(X_train)/len(feats)*100:.1f}%)")
print(f"Test samples:        {len(X_test)} ({len(X_test)/len(feats)*100:.1f}%)")
print(f"Features:            {len(X_train.columns)}")
print(f"\nClass distribution (train):")
print(y_train.value_counts().sort_index())
print(f"\nClass distribution (test):")
print(y_test.value_counts().sort_index())

print("\n" + "="*80)
print("END OF METRICS REPORT")
print("="*80 + "\n")

# Get feature importance from best model
feature_importance = best_model.feature_importances_
feature_names = X_train.columns

PR = PrecisionRecallDisplay.from_estimator(
    best_model, X_test, y_test, name = 'XGBoost',
    plot_chance_level = False,
)
plt.savefig('output/precision_recall.png', dpi = 300)
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
plt.savefig('output/confusion_matrix.png', dpi = 300)
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
plt.savefig('output/feature_importance.png', dpi = 300)
plt.show()

best_model.save_model('models/model.json')