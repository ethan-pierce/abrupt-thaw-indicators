"""Use cross-validation to find the best parameters for an XGBoost model."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, f1_score, roc_curve, precision_recall_curve, 
    PrecisionRecallDisplay, RocCurveDisplay, confusion_matrix, ConfusionMatrixDisplay, accuracy_score,
    precision_score, recall_score, classification_report, average_precision_score,
    log_loss, matthews_corrcoef, cohen_kappa_score, hamming_loss, jaccard_score
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from settings import DATA, MODELS, OUTPUT

data = DATA
feats = pd.read_csv(data / 'features_clean.csv')
rng = np.random.default_rng(42)

matrix = feats.drop('Class', axis = 1)
target = feats['Class']

# First split: separate test set
X_train_full, X_test, y_train_full, y_test = train_test_split(
    matrix, target, 
    test_size = 0.3, 
    random_state = rng.integers(0, 100), 
    shuffle = True, 
    stratify = target
)

# Use full training set (no separate validation set needed without early stopping)
X_train = X_train_full
y_train = y_train_full

print(f"Training set size: {len(X_train)} samples")
print(f"Test set size: {len(X_test)} samples")
print(f"\nTraining class distribution:")
print(y_train.value_counts())
print(f"Class imbalance ratio: {y_train.value_counts()[0] / y_train.value_counts()[1]:.2f}:1 (Abrupt:Gradual)")


param_grid = {
    # Structural Stability
    'max_depth': [4, 5],
    'min_child_weight': [20],       # Prevents small-bin oscillation
    'learning_rate': [0.05],        # "Slow and steady"
    
    # Regularization (The "Smoothing" Trio)
    'reg_lambda': [10, 50],         # L2 is best for probability smoothing
    'alpha': [0],                   # Keep L1 off to avoid "dead" features for now
    'gamma': [1],                   # Minimum gain for a split
    
    # Randomization (Noise Reduction)
    'subsample': [0.8],
    'colsample_bytree': [0.8],
    
    # Number of trees
    'n_estimators': [100, 200, 300]
}

# Create base XGBoost classifier for GridSearchCV
xgb_classifier = xgb.XGBClassifier(
    random_state = rng.integers(0, 100),
    objective = 'binary:logistic',
    eval_metric = 'aucpr',
    scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
)

stratified_kfold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = rng.integers(0, 100))

print(f"\n--- Hyperparameter Grid Search ---")
print(f"Total combinations: {np.prod([len(v) for v in param_grid.values()])}")
print(f"With {stratified_kfold.n_splits}-fold CV: {np.prod([len(v) for v in param_grid.values()]) * stratified_kfold.n_splits} model fits")

grid = GridSearchCV(
    estimator = xgb_classifier,
    param_grid = param_grid,
    cv = stratified_kfold,
    scoring = 'neg_brier_score',
    verbose = 3,
    n_jobs = 8,
    return_train_score = True # Disable for many jobs
)

fit = grid.fit(X_train, y_train)

print(f"Best parameters: {grid.best_params_}")
print(f"Best CV score: {grid.best_score_:.4f}")
print(f"CV train score: {grid.cv_results_['mean_train_score'][grid.best_index_]:.4f}")
print(f"CV test score: {grid.cv_results_['mean_test_score'][grid.best_index_]:.4f}")

# Get the best model from GridSearchCV
best_model = grid.best_estimator_

print("\n" + "="*80)
print("COMPREHENSIVE MODEL METRICS")
print("="*80)

# Get predictions
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)
y_train_proba_abrupt = best_model.predict_proba(X_train)[:, 0]  # Probabilities for Abrupt (class 0)
y_test_proba_abrupt = best_model.predict_proba(X_test)[:, 0]   # Probabilities for Abrupt (class 0)
y_train_proba_positive = best_model.predict_proba(X_train)[:, 1]  # Probabilities for Gradual (class 1, positive class)
y_test_proba_positive = best_model.predict_proba(X_test)[:, 1]   # Probabilities for Gradual (class 1, positive class)

# Diagnostic: Check prediction distribution
print("\n--- Prediction Distribution Check ---")
print(f"Test set predictions - Class 0: {(y_test_pred == 0).sum()}, Class 1: {(y_test_pred == 1).sum()}")
print(f"Test set true labels - Class 0: {(y_test == 0).sum()}, Class 1: {(y_test == 1).sum()}")
print(f"Test set probability range (Abrupt): [{y_test_proba_abrupt.min():.4f}, {y_test_proba_abrupt.max():.4f}]")
print(f"Test set probability mean (Abrupt): {y_test_proba_abrupt.mean():.4f}")
print(f"Test set probability median (Abrupt): {np.median(y_test_proba_abrupt):.4f}")

# Check if model is just predicting majority class
pred_class_1_pct = (y_test_pred == 1).sum() / len(y_test_pred) * 100
true_class_1_pct = (y_test == 1).sum() / len(y_test) * 100
print(f"\n--- Model Behavior Check ---")
print(f"True class 1 percentage: {true_class_1_pct:.2f}%")
print(f"Predicted class 1 percentage: {pred_class_1_pct:.2f}%")
if abs(pred_class_1_pct - true_class_1_pct) < 1.0:
    print("⚠️  WARNING: Model predictions match class distribution - may be just predicting majority class!")
    print("   Check probability distribution - if all probabilities are >0.5, model is not learning.")

# Check probability distribution for minority class (using positive class probabilities)
minority_proba = y_test_proba_positive[y_test == 1]  # Probabilities for Gradual when true label is Gradual
majority_proba = y_test_proba_positive[y_test == 0]   # Probabilities for Gradual when true label is Abrupt
print(f"\nProbability distribution for MINORITY class (Gradual, class 1) - using positive class probabilities:")
print(f"  Mean: {np.mean(minority_proba):.4f}, Median: {np.median(minority_proba):.4f}")
print(f"  Range: [{np.min(minority_proba):.4f}, {np.max(minority_proba):.4f}]")
print(f"  Samples with prob > 0.5: {(minority_proba > 0.5).sum()} / {len(minority_proba)} ({(minority_proba > 0.5).sum()/len(minority_proba)*100:.1f}%)")
print(f"\nProbability distribution for MAJORITY class (Abrupt, class 0) - using positive class probabilities:")
print(f"  Mean: {np.mean(majority_proba):.4f}, Median: {np.median(majority_proba):.4f}")
print(f"  Range: [{np.min(majority_proba):.4f}, {np.max(majority_proba):.4f}]")
print(f"  Samples with prob < 0.5: {(majority_proba < 0.5).sum()} / {len(majority_proba)} ({(majority_proba < 0.5).sum()/len(majority_proba)*100:.1f}%)")

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

# Probability-based metrics (using positive class probabilities)
print("\n--- Probability-Based Metrics (Test Set) ---")
test_auc_roc = roc_auc_score(y_test, y_test_proba_positive)
test_auc_pr = average_precision_score(y_test, y_test_proba_positive)
test_brier = brier_score_loss(y_test, y_test_proba_positive)
test_log_loss = log_loss(y_test, y_test_proba_positive)

print(f"AUC-ROC:            {test_auc_roc:.4f}")
print(f"AUC-PR (AP):        {test_auc_pr:.4f}")
print(f"Brier Score:        {test_brier:.4f}")
print(f"Log Loss:           {test_log_loss:.4f}")

# Classification report
print("\n--- Detailed Classification Report (Test Set) ---")
print(classification_report(y_test, y_test_pred, target_names=['Abrupt Thaw', 'Gradual Thaw']))

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
train_auc_roc = roc_auc_score(y_train, y_train_proba_positive)
train_auc_pr = average_precision_score(y_train, y_train_proba_positive)
train_brier = brier_score_loss(y_train, y_train_proba_positive)

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

print(f"\n--- Probability Distribution Statistics (Positive Class) ---")
print(f"Test predictions - Mean: {np.mean(y_test_proba_positive):.4f}, Std: {np.std(y_test_proba_positive):.4f}")
print(f"Test predictions - Min: {np.min(y_test_proba_positive):.4f}, Max: {np.max(y_test_proba_positive):.4f}")
print(f"Test predictions - Median: {np.median(y_test_proba_positive):.4f}")
print(f"Test predictions - Q25: {np.percentile(y_test_proba_positive, 25):.4f}, Q75: {np.percentile(y_test_proba_positive, 75):.4f}")

# Confidence by class (using positive class probabilities)
print(f"\n--- Confidence by True Class (Positive Class Probabilities) ---")
for class_label, class_name in [(0, 'Abrupt Thaw'), (1, 'Gradual Thaw')]:
    mask = y_test == class_label
    if mask.sum() > 0:
        class_proba = y_test_proba_positive[mask]
        print(f"{class_name}: Mean={np.mean(class_proba):.4f}, Std={np.std(class_proba):.4f}, "
              f"Min={np.min(class_proba):.4f}, Max={np.max(class_proba):.4f}")

# ============================================================================
# CALIBRATION ANALYSIS
# ============================================================================
print("\n" + "-"*80)
print("CALIBRATION ANALYSIS")
print("-"*80)

# Calculate calibration curve (using positive class probabilities)
fraction_of_positives, mean_predicted_value = calibration_curve(
    y_test, y_test_proba_positive, n_bins=10, strategy='uniform'
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

# Comprehensive feature importance check
print(f"\n--- Feature Importance Analysis ---")
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

max_importance = feature_importance.max()
total_importance = feature_importance.sum()

print(f"Total features: {len(feature_names)}")
print(f"Max importance: {max_importance:.6f}")
print(f"Total importance: {total_importance:.6f}")

# Check for features that are too important
# Criteria: >30% of max importance, or >15% of total importance, or in top 3 with >10% of total
problematic_features = []
threshold_max_ratio = 0.30  # 30% of max importance
threshold_total_ratio = 0.15  # 15% of total importance
top_n_threshold = 3  # Check top N features
top_n_total_ratio = 0.10  # 10% of total importance for top N

print(f"\n--- Checking for Overly Important Features ---")
print(f"Thresholds:")
print(f"  - Max importance ratio: >{threshold_max_ratio:.0%}")
print(f"  - Total importance ratio: >{threshold_total_ratio:.0%}")
print(f"  - Top {top_n_threshold} features with >{top_n_total_ratio:.0%} of total importance")

for idx, row in importance_df.iterrows():
    feature = row['feature']
    importance = row['importance']
    max_ratio = importance / max_importance if max_importance > 0 else 0
    total_ratio = importance / total_importance if total_importance > 0 else 0
    rank = importance_df.index.get_loc(idx) + 1
    
    is_problematic = False
    reasons = []
    
    if max_ratio > threshold_max_ratio:
        is_problematic = True
        reasons.append(f"{max_ratio:.1%} of max importance")
    
    if total_ratio > threshold_total_ratio:
        is_problematic = True
        reasons.append(f"{total_ratio:.1%} of total importance")
    
    if rank <= top_n_threshold and total_ratio > top_n_total_ratio:
        is_problematic = True
        reasons.append(f"Top {rank} feature with {total_ratio:.1%} of total")
    
    if is_problematic:
        problematic_features.append({
            'feature': feature,
            'importance': importance,
            'rank': rank,
            'max_ratio': max_ratio,
            'total_ratio': total_ratio,
            'reasons': reasons
        })
        print(f"\n⚠️  PROBLEMATIC: {feature} (Rank #{rank})")
        print(f"   Importance: {importance:.6f}")
        print(f"   {max_ratio:.1%} of max importance, {total_ratio:.1%} of total importance")
        print(f"   Reasons: {', '.join(reasons)}")
        print(f"   ⚠️  Consider removing this feature if it's too dominant")

if not problematic_features:
    print("\n✅ No overly important features detected - feature importance is well distributed")
else:
    print(f"\n📊 Summary: Found {len(problematic_features)} potentially problematic feature(s)")
    print("   Consider reviewing these features and potentially removing them to improve model generalization")

# Show top 10 features for reference
print(f"\n--- Top 10 Most Important Features ---")
top_10 = importance_df.head(10)
for rank, (idx, row) in enumerate(top_10.iterrows(), 1):
    feature = row['feature']
    importance = row['importance']
    total_ratio = importance / total_importance if total_importance > 0 else 0
    print(f"  {rank:2d}. {feature:40s} {importance:.6f} ({total_ratio:.1%} of total)")

# Verify class encoding before plotting
print("\n--- Class Encoding Verification ---")
print(f"Class 0 count in y_test: {(y_test == 0).sum()} (should be Abrupt)")
print(f"Class 1 count in y_test: {(y_test == 1).sum()} (should be Gradual)")
print(f"Using predict_proba[:, 0] for Abrupt probabilities, predict_proba[:, 1] for Gradual (positive class) probabilities")

# Compute precision-recall manually to verify (using positive class probabilities)
precision, recall, thresholds_pr = precision_recall_curve(y_test, y_test_proba_positive)
print(f"Manual PR-AUC: {average_precision_score(y_test, y_test_proba_positive):.4f}")
print(f"Precision at recall=0.5: {precision[recall >= 0.5][0] if len(precision[recall >= 0.5]) > 0 else 'N/A'}")

# Plot Precision-Recall curve
PR = PrecisionRecallDisplay.from_estimator(
    best_model, X_test, y_test, name = 'XGBoost',
    plot_chance_level = True,  # Show baseline for comparison
)
plt.title('Precision-Recall Curve')
plt.savefig(OUTPUT / 'precision_recall.png', dpi = 300)
plt.show()

# Plot ROC curve (less sensitive to class imbalance)
roc_display = RocCurveDisplay.from_estimator(
    best_model, X_test, y_test, name = 'XGBoost'
)
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.5)')
plt.title('ROC Curve (Receiver Operating Characteristic)')
plt.legend()
plt.savefig(OUTPUT / 'roc_curve.png', dpi = 300)
plt.show()

# ============================================================================
# ENHANCED CALIBRATION ANALYSIS
# ============================================================================
print("\n" + "-"*80)
print("ENHANCED CALIBRATION ANALYSIS")
print("-"*80)

# Calculate calibration curves with different binning strategies (using positive class probabilities)
fraction_of_positives_uniform, mean_predicted_value_uniform = calibration_curve(
    y_test, y_test_proba_positive, n_bins=10, strategy='uniform'
)

fraction_of_positives_quantile, mean_predicted_value_quantile = calibration_curve(
    y_test, y_test_proba_positive, n_bins=10, strategy='quantile'
)

# Calculate calibration errors
mean_calibration_error_uniform = np.mean(np.abs(fraction_of_positives_uniform - mean_predicted_value_uniform))
mean_calibration_error_quantile = np.mean(np.abs(fraction_of_positives_quantile - mean_predicted_value_quantile))

print(f"\nCalibration errors:")
print(f"  Uniform binning:  {mean_calibration_error_uniform:.4f}")
print(f"  Quantile binning: {mean_calibration_error_quantile:.4f}")

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

ece = expected_calibration_error(y_test, y_test_proba_positive)
print(f"  Expected Calibration Error (ECE): {ece:.4f}")

# Plot enhanced calibration curves
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Uniform binning
ax1 = axes[0]
ax1.plot(mean_predicted_value_uniform, fraction_of_positives_uniform, 's-', 
         label='XGBoost', linewidth=2, markersize=8, color='blue')
ax1.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated', linewidth=2)
ax1.set_xlabel('Mean Predicted Probability', fontsize=12)
ax1.set_ylabel('Fraction of Positives', fontsize=12)
ax1.set_title(f'Calibration Curve (Uniform Binning)\nMCE = {mean_calibration_error_uniform:.4f}, ECE = {ece:.4f}', 
              fontsize=12, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xlim([0, 1])
ax1.set_ylim([0, 1])

# Add bin counts as text
for i, (mean_pred, frac_pos) in enumerate(zip(mean_predicted_value_uniform, fraction_of_positives_uniform)):
    bin_size = len(y_test) / 10  # Approximate bin size
    ax1.text(mean_pred, frac_pos + 0.05, f'n={int(bin_size)}', 
             fontsize=8, ha='center', alpha=0.7)

# Plot 2: Quantile binning
ax2 = axes[1]
ax2.plot(mean_predicted_value_quantile, fraction_of_positives_quantile, 'o-', 
         label='XGBoost', linewidth=2, markersize=8, color='green')
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
plt.savefig(OUTPUT / 'calibration_curve_enhanced.png', dpi=300)
plt.show()

# Also save the simple calibration curve for backward compatibility
plt.figure(figsize=(10, 8))
plt.plot(mean_predicted_value_uniform, fraction_of_positives_uniform, 's-', 
         label='XGBoost', linewidth=2, markersize=8)
plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated', linewidth=2)
plt.xlabel('Mean Predicted Probability', fontsize=12)
plt.ylabel('Fraction of Positives', fontsize=12)
plt.title(f'Calibration Curve (Probability Calibration)\nMCE = {mean_calibration_error_uniform:.4f}, ECE = {ece:.4f}', 
          fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT / 'calibration_curve.png', dpi=300)
plt.show()

# Print detailed calibration statistics
print(f"\nDetailed calibration (uniform bins):")
for i, (frac, mean) in enumerate(zip(fraction_of_positives_uniform, mean_predicted_value_uniform)):
    bin_size = len(y_test) / 10
    print(f"  Bin {i+1}: Predicted={mean:.3f}, Actual={frac:.3f}, Gap={abs(mean-frac):.3f}, Samples≈{int(bin_size)}")

# ============================================================================
# COMPREHENSIVE THRESHOLD ANALYSIS
# ============================================================================
print("\n" + "-"*80)
print("THRESHOLD ANALYSIS")
print("-"*80)

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
    y_pred_thresh = (y_test_proba_positive >= threshold).astype(int)  # Use positive class probabilities
    
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

# Find optimal thresholds for different objectives
best_f1_idx = np.argmax(metrics['f1'])
best_f1_threshold = thresholds[best_f1_idx]
best_f1_score = metrics['f1'][best_f1_idx]

# Find threshold that balances precision and recall (closest to equal)
balanced_idx = np.argmin([abs(p - r) for p, r in zip(metrics['precision'], metrics['recall'])])
balanced_threshold = thresholds[balanced_idx]

# Find threshold that maximizes specificity (minimize false positives for positive class)
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

# Plot 4: False Negative Rate (important for detecting minority class)
ax4 = axes[1, 1]
ax4.plot(thresholds, metrics['false_negative_rate'], label='False Negative Rate', linewidth=2, color='purple')
ax4.axvline(x=0.5, color='r', linestyle='--', alpha=0.5, label='Default (0.5)')
ax4.axvline(x=best_f1_threshold, color='g', linestyle='--', alpha=0.7, label=f'Best F1 ({best_f1_threshold:.2f})')
ax4.set_xlabel('Decision Threshold', fontsize=11)
ax4.set_ylabel('False Negative Rate', fontsize=11)
ax4.set_title('False Negative Rate vs Threshold\n(Lower is better for detecting Gradual Thaw)', fontsize=12, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT / 'threshold_analysis.png', dpi=300)
plt.show()

# Also save the simple F1 vs threshold plot for backward compatibility
plt.figure(figsize=(10, 6))
plt.plot(thresholds, metrics['f1'], linewidth=2)
plt.xlabel('Decision Threshold', fontsize=12)
plt.ylabel('F1-Score', fontsize=12)
plt.title('F1-Score vs Decision Threshold', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.axvline(x=0.5, color='r', linestyle='--', label='Default Threshold (0.5)')
plt.axvline(x=best_f1_threshold, color='g', linestyle='--', label=f'Best Threshold ({best_f1_threshold:.2f}, F1={best_f1_score:.3f})')
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT / 'f1_vs_threshold.png', dpi=300)
plt.show()

# Plot confusion matrix
y_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
# Convert to percentages
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
disp = ConfusionMatrixDisplay(confusion_matrix = cm_percent, display_labels=['Abrupt Thaw', 'Gradual Thaw'])  # [class 0, class 1]
fig, ax = plt.subplots(figsize = (8, 6))
disp.plot(ax = ax, cmap = 'Blues', values_format = '.1f')
plt.title('Confusion Matrix - XGBoost Model (Percentages)')
plt.savefig(OUTPUT / 'confusion_matrix.png', dpi = 300)
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
plt.savefig(OUTPUT / 'feature_importance.png', dpi = 300)
plt.show()

# Save the XGBoost model
best_model.save_model(str(MODELS / 'model.json'))
print(f"Saved XGBoost model to: models/model.json")