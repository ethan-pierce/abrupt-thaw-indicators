"""
Use cross-validation to find the best parameters for an XGBoost model.
Fixed for XGBoost API compatibility and probability calibration.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, f1_score, confusion_matrix, 
    accuracy_score, average_precision_score, log_loss, matthews_corrcoef,
    roc_curve, precision_recall_curve, precision_score, recall_score,
    RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

# --- DATA LOADING ---
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from settings import DATA, MODELS, OUTPUT

data = DATA
output_dir = OUTPUT
feats = pd.read_csv(data / 'features_clean.csv')

matrix = feats.drop('Class', axis=1)
target = feats['Class']

# First split: separate test set (completely unseen)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    matrix, target, 
    test_size=0.3, 
    random_state=42, 
    shuffle=True, 
    stratify=target
)

# Second split: separate validation set for early stopping AND calibration fitting
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.2,
    random_state=42,
    shuffle=True, 
    stratify=y_train_full
)

print(f"Training set: {len(X_train)} | Val set: {len(X_val)} | Test set: {len(X_test)}")

# --- PHASE 1: HYPERPARAMETER TUNING ---
# We use a conservative grid to stop oscillations and over-confidence
param_grid = {
    'max_depth': [4, 5],
    'min_child_weight': [20, 50],
    'learning_rate': [0.05],
    'reg_lambda': [50, 100], 
    'gamma': [1, 5],
    'subsample': [0.8],
    'colsample_bytree': [0.8]
}

# Define the model with early stopping in the constructor
# This fixes the TypeError you encountered
xgb_base = xgb.XGBClassifier(
    n_estimators=1000,           # Set high, let early stopping trim it
    early_stopping_rounds=50,    # Moved from .fit() to here
    random_state=42,
    objective='binary:logistic',
    eval_metric='logloss',       # Matches our goal of probability accuracy
    tree_method='hist'           # Faster for grid searches
)

# When using GridSearchCV with early stopping, you pass the eval_set to .fit()
fit_params = {
    "eval_set": [(X_val, y_val)],
    "verbose": False
}

print("\n--- Running GridSearchCV (Scoring: Neg Log Loss) ---")
grid = GridSearchCV(
    estimator=xgb_base,
    param_grid=param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='neg_log_loss',
    n_jobs=-1,
    verbose=1
)

# Pass fit_params here. This avoids the TypeError and handles early stopping.
grid.fit(X_train, y_train, **fit_params)
best_stable_xgb = grid.best_estimator_

print(f"Best Params: {grid.best_params_}")

# --- PHASE 2: POST-HOC CALIBRATION ---
# This fixes the "U-shape" (over-confidence) by applying Sigmoid scaling
print("\n--- Applying Sigmoid Calibration Scaling ---")
calibrated_model = CalibratedClassifierCV(
    best_stable_xgb,
    method='sigmoid',
    cv='prefit' 
)
calibrated_model.fit(X_val, y_val)

# --- EVALUATION ---
y_test_proba = calibrated_model.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_proba > 0.5).astype(int)

print("\n" + "="*40)
print("FINAL CALIBRATED MODEL METRICS (TEST SET)")
print("="*40)
print(f"AUC-ROC:    {roc_auc_score(y_test, y_test_proba):.4f}")
print(f"Log Loss:   {log_loss(y_test, y_test_proba):.4f}")
print(f"Brier Score:{brier_score_loss(y_test, y_test_proba):.4f}")
print(f"F1-Score:   {f1_score(y_test, y_test_pred):.4f}")
print(f"Precision:  {precision_score(y_test, y_test_pred, zero_division=0):.4f}")
print(f"Recall:     {recall_score(y_test, y_test_pred, zero_division=0):.4f}")
print(f"Accuracy:   {accuracy_score(y_test, y_test_pred):.4f}")
print(f"AUC-PR:     {average_precision_score(y_test, y_test_proba):.4f}")

# ============================================================================
# ROC CURVE
# ============================================================================
print("\n--- Generating ROC Curve ---")
roc_display = RocCurveDisplay.from_estimator(
    calibrated_model, X_test, y_test, name='Calibrated XGBoost'
)
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.5)')
plt.title('ROC Curve - Calibrated XGBoost Model', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / 'roc_curve_calibrated.png', dpi=300)
plt.show()

# ============================================================================
# PRECISION-RECALL CURVE
# ============================================================================
print("--- Generating Precision-Recall Curve ---")
pr_display = PrecisionRecallDisplay.from_estimator(
    calibrated_model, X_test, y_test, name='Calibrated XGBoost',
    plot_chance_level=True
)
plt.title('Precision-Recall Curve - Calibrated XGBoost Model', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / 'precision_recall_calibrated.png', dpi=300)
plt.show()

# ============================================================================
# ENHANCED CALIBRATION CURVES
# ============================================================================
print("--- Generating Enhanced Calibration Curves ---")
# Calculate calibration curves with different binning strategies
prob_true_uniform, prob_pred_uniform = calibration_curve(
    y_test, y_test_proba, n_bins=10, strategy='uniform'
)
prob_true_quantile, prob_pred_quantile = calibration_curve(
    y_test, y_test_proba, n_bins=10, strategy='quantile'
)

# Calculate calibration errors
mean_calibration_error_uniform = np.mean(np.abs(prob_true_uniform - prob_pred_uniform))
mean_calibration_error_quantile = np.mean(np.abs(prob_true_quantile - prob_pred_quantile))

# Expected Calibration Error (ECE)
def expected_calibration_error(y_true, y_pred, n_bins=10):
    """Calculate Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_pred > bin_lower) & (y_pred <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_pred[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece

ece = expected_calibration_error(y_test, y_test_proba)
print(f"Calibration errors:")
print(f"  Uniform binning:  {mean_calibration_error_uniform:.4f}")
print(f"  Quantile binning: {mean_calibration_error_quantile:.4f}")
print(f"  Expected Calibration Error (ECE): {ece:.4f}")

# Plot enhanced calibration curves
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Uniform binning
ax1 = axes[0]
ax1.plot(prob_pred_uniform, prob_true_uniform, 's-', 
         label='Calibrated XGBoost', linewidth=2, markersize=8, color='blue')
ax1.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated', linewidth=2)
ax1.set_xlabel('Mean Predicted Probability', fontsize=12)
ax1.set_ylabel('Fraction of Positives', fontsize=12)
ax1.set_title(f'Calibration Curve (Uniform Binning)\nMCE = {mean_calibration_error_uniform:.4f}, ECE = {ece:.4f}', 
              fontsize=12, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xlim([0, 1])
ax1.set_ylim([0, 1])

# Plot 2: Quantile binning
ax2 = axes[1]
ax2.plot(prob_pred_quantile, prob_true_quantile, 'o-', 
         label='Calibrated XGBoost', linewidth=2, markersize=8, color='green')
ax2.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated', linewidth=2)
ax2.set_xlabel('Mean Predicted Probability', fontsize=12)
ax2.set_ylabel('Fraction of Positives', fontsize=12)
ax2.set_title(f'Calibration Curve (Quantile Binning)\nMCE = {mean_calibration_error_quantile:.4f}', 
              fontsize=12, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_xlim([0, 1])
ax2.set_ylim([0, 1])

plt.tight_layout()
plt.savefig(output_dir / 'calibration_curve_enhanced_calibrated.png', dpi=300)
plt.show()

# Simple calibration curve for backward compatibility
plt.figure(figsize=(10, 8))
plt.plot(prob_pred_quantile, prob_true_quantile, 's-', 
         label='Calibrated XGBoost', linewidth=2, markersize=8)
plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated', linewidth=2)
plt.xlabel('Mean Predicted Probability', fontsize=12)
plt.ylabel('Fraction of Positives', fontsize=12)
plt.title(f'Calibration Curve (Calibrated Model)\nMCE = {mean_calibration_error_quantile:.4f}, ECE = {ece:.4f}', 
          fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / 'calibration_curve_calibrated.png', dpi=300)
plt.show()

# ============================================================================
# THRESHOLD ANALYSIS
# ============================================================================
print("--- Generating Threshold Analysis ---")
thresholds = np.arange(0.01, 1.0, 0.01)
metrics = {
    'precision': [],
    'recall': [],
    'f1': [],
    'specificity': [],
    'accuracy': [],
    'false_positive_rate': [],
    'false_negative_rate': []
}

for threshold in thresholds:
    y_pred_thresh = (y_test_proba >= threshold).astype(int)
    
    # Calculate all metrics
    prec = precision_score(y_test, y_pred_thresh, zero_division=0)
    rec = recall_score(y_test, y_pred_thresh, zero_division=0)
    f1 = f1_score(y_test, y_pred_thresh, zero_division=0)
    acc = accuracy_score(y_test, y_pred_thresh)
    
    cm = confusion_matrix(y_test, y_pred_thresh)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    else:
        spec = 0.0
        fpr = 0.0
        fnr = 0.0
    
    metrics['precision'].append(prec)
    metrics['recall'].append(rec)
    metrics['f1'].append(f1)
    metrics['specificity'].append(spec)
    metrics['accuracy'].append(acc)
    metrics['false_positive_rate'].append(fpr)
    metrics['false_negative_rate'].append(fnr)

# Find optimal thresholds
best_f1_idx = np.argmax(metrics['f1'])
best_f1_threshold = thresholds[best_f1_idx]
best_f1_score = metrics['f1'][best_f1_idx]

balanced_idx = np.argmin([abs(p - r) for p, r in zip(metrics['precision'], metrics['recall'])])
balanced_threshold = thresholds[balanced_idx]

best_spec_idx = np.argmax(metrics['specificity'])
best_spec_threshold = thresholds[best_spec_idx]

print(f"\nOptimal thresholds:")
print(f"  Best F1-score:        {best_f1_threshold:.3f} (F1 = {best_f1_score:.3f})")
print(f"  Balanced P/R:          {balanced_threshold:.3f} (Precision = {metrics['precision'][balanced_idx]:.3f}, Recall = {metrics['recall'][balanced_idx]:.3f})")
print(f"  Max Specificity:      {best_spec_threshold:.3f} (Specificity = {metrics['specificity'][best_spec_idx]:.3f})")
print(f"  Default (0.5):        Precision = {metrics['precision'][np.argmin(np.abs(thresholds - 0.5))]:.3f}, Recall = {metrics['recall'][np.argmin(np.abs(thresholds - 0.5))]:.3f}")

# Plot comprehensive threshold analysis
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Precision, Recall, F1
ax1 = axes[0, 0]
ax1.plot(thresholds, metrics['precision'], label='Precision', linewidth=2)
ax1.plot(thresholds, metrics['recall'], label='Recall', linewidth=2)
ax1.plot(thresholds, metrics['f1'], label='F1-Score', linewidth=2)
ax1.axvline(x=0.5, color='r', linestyle='--', alpha=0.5, label='Default (0.5)')
ax1.axvline(x=best_f1_threshold, color='g', linestyle='--', alpha=0.7, label=f'Best F1 ({best_f1_threshold:.2f})')
ax1.axvline(x=balanced_threshold, color='orange', linestyle='--', alpha=0.7, label=f'Balanced ({balanced_threshold:.2f})')
ax1.set_xlabel('Decision Threshold', fontsize=11)
ax1.set_ylabel('Score', fontsize=11)
ax1.set_title('Precision, Recall, and F1-Score vs Threshold', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Plot 2: Specificity and False Positive Rate
ax2 = axes[0, 1]
ax2.plot(thresholds, metrics['specificity'], label='Specificity (True Negative Rate)', linewidth=2, color='green')
ax2.plot(thresholds, metrics['false_positive_rate'], label='False Positive Rate', linewidth=2, color='red')
ax2.axvline(x=0.5, color='r', linestyle='--', alpha=0.5, label='Default (0.5)')
ax2.axvline(x=best_spec_threshold, color='g', linestyle='--', alpha=0.7, label=f'Max Specificity ({best_spec_threshold:.2f})')
ax2.set_xlabel('Decision Threshold', fontsize=11)
ax2.set_ylabel('Rate', fontsize=11)
ax2.set_title('Specificity and False Positive Rate vs Threshold', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Plot 3: Accuracy
ax3 = axes[1, 0]
ax3.plot(thresholds, metrics['accuracy'], label='Accuracy', linewidth=2, color='blue')
ax3.axvline(x=0.5, color='r', linestyle='--', alpha=0.5, label='Default (0.5)')
ax3.axvline(x=best_f1_threshold, color='g', linestyle='--', alpha=0.7, label=f'Best F1 ({best_f1_threshold:.2f})')
ax3.set_xlabel('Decision Threshold', fontsize=11)
ax3.set_ylabel('Accuracy', fontsize=11)
ax3.set_title('Accuracy vs Threshold', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# Plot 4: False Negative Rate
ax4 = axes[1, 1]
ax4.plot(thresholds, metrics['false_negative_rate'], label='False Negative Rate', linewidth=2, color='purple')
ax4.axvline(x=0.5, color='r', linestyle='--', alpha=0.5, label='Default (0.5)')
ax4.axvline(x=best_f1_threshold, color='g', linestyle='--', alpha=0.7, label=f'Best F1 ({best_f1_threshold:.2f})')
ax4.set_xlabel('Decision Threshold', fontsize=11)
ax4.set_ylabel('False Negative Rate', fontsize=11)
ax4.set_title('False Negative Rate vs Threshold', fontsize=12, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'threshold_analysis_calibrated.png', dpi=300)
plt.show()

# Simple F1 vs threshold plot
plt.figure(figsize=(10, 6))
plt.plot(thresholds, metrics['f1'], linewidth=2)
plt.xlabel('Decision Threshold', fontsize=12)
plt.ylabel('F1-Score', fontsize=12)
plt.title('F1-Score vs Decision Threshold - Calibrated Model', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.axvline(x=0.5, color='r', linestyle='--', label='Default Threshold (0.5)')
plt.axvline(x=best_f1_threshold, color='g', linestyle='--', label=f'Best Threshold ({best_f1_threshold:.2f}, F1={best_f1_score:.3f})')
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / 'f1_vs_threshold_calibrated.png', dpi=300)
plt.show()

# ============================================================================
# CONFUSION MATRIX
# ============================================================================
print("--- Generating Confusion Matrix ---")
cm = confusion_matrix(y_test, y_test_pred)
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
disp = ConfusionMatrixDisplay(confusion_matrix=cm_percent, display_labels=['Abrupt Thaw', 'Gradual Thaw'])
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap='Blues', values_format='.1f')
plt.title('Confusion Matrix - Calibrated XGBoost Model (Percentages)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / 'confusion_matrix_calibrated.png', dpi=300)
plt.show()

# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================
print("--- Generating Feature Importance Plot ---")
# Get feature importance from the base model (before calibration)
feature_importance = best_stable_xgb.feature_importances_
feature_names = X_train.columns

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

# Plot top 20 features
plt.figure(figsize=(12, 8))
top_features = importance_df.head(20)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Feature Importance', fontsize=12)
plt.title('Top 20 Feature Importances - Calibrated XGBoost Model', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(output_dir / 'feature_importance_calibrated.png', dpi=300)
plt.show()

# ============================================================================
# SAVE MODEL
# ============================================================================
print("\n--- Saving Calibrated Model ---")
models_dir = MODELS
model_path = models_dir / 'model_calibrated.pkl'
joblib.dump(calibrated_model, model_path)
print(f"Saved calibrated model to: {model_path}")

# Also save the base XGBoost model for reference
base_model_path = models_dir / 'model_calibrated_base.json'
best_stable_xgb.save_model(str(base_model_path))
print(f"Saved base XGBoost model to: {base_model_path}")

print("\n" + "="*40)
print("ALL PLOTS GENERATED AND MODEL SAVED")
print("="*40)