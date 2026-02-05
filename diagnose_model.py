"""Diagnostic script to check why the model appears perfect."""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_score, recall_score, f1_score, classification_report
)

# Load data
data = Path(__file__).parent / 'data'
feats = pd.read_csv(data / 'features_clean.csv')
rng = np.random.default_rng(42)

matrix = feats.drop('Class', axis=1)
target = feats['Class']

X_train, X_test, y_train, y_test = train_test_split(
    matrix, target,
    test_size=0.3,
    random_state=42,
    shuffle=True,
    stratify=target
)

print("="*80)
print("DIAGNOSTIC ANALYSIS")
print("="*80)

# 1. Check class distribution
print("\n1. CLASS DISTRIBUTION")
print("-" * 80)
print(f"Training set: Class 0 (Gradual): {(y_train == 0).sum()}, Class 1 (Abrupt): {(y_train == 1).sum()}")
print(f"Test set: Class 0 (Gradual): {(y_test == 0).sum()}, Class 1 (Abrupt): {(y_test == 1).sum()}")
naive_accuracy = (y_test == 1).mean()  # Always predict majority class
print(f"\nNaive majority-class predictor accuracy: {naive_accuracy:.4f} ({naive_accuracy*100:.2f}%)")

# 2. Load the trained model
print("\n2. LOADING TRAINED MODEL")
print("-" * 80)
model_path = Path(__file__).parent / 'models' / 'model.json'
if model_path.exists():
    model = xgb.XGBClassifier()
    model.load_model(str(model_path))
    print(f"Model loaded from: {model_path}")
else:
    print(f"ERROR: Model not found at {model_path}")
    print("Please train the model first.")
    exit(1)

# 3. Get predictions
y_test_pred = model.predict(X_test)
y_test_proba = model.predict_proba(X_test)[:, 1]

# 4. Check if model is just predicting majority class
print("\n3. PREDICTION DISTRIBUTION")
print("-" * 80)
print(f"Predictions - Class 0: {(y_test_pred == 0).sum()}, Class 1: {(y_test_pred == 1).sum()}")
print(f"True labels - Class 0: {(y_test == 0).sum()}, Class 1: {(y_test == 1).sum()}")
pred_class_1_pct = (y_test_pred == 1).sum() / len(y_test_pred) * 100
true_class_1_pct = (y_test == 1).sum() / len(y_test) * 100
print(f"\nTrue class 1 percentage: {true_class_1_pct:.2f}%")
print(f"Predicted class 1 percentage: {pred_class_1_pct:.2f}%")
if abs(pred_class_1_pct - true_class_1_pct) < 2.0:
    print("⚠️  WARNING: Predictions closely match class distribution!")
    print("   This suggests the model may be just predicting the majority class.")

# 5. Probability distribution analysis
print("\n4. PROBABILITY DISTRIBUTION")
print("-" * 80)
print(f"Overall probability range: [{y_test_proba.min():.4f}, {y_test_proba.max():.4f}]")
print(f"Overall probability mean: {y_test_proba.mean():.4f}, median: {np.median(y_test_proba):.4f}")
print(f"Overall probability std: {y_test_proba.std():.4f}")

# By class
minority_proba = y_test_proba[y_test == 0]
majority_proba = y_test_proba[y_test == 1]

print(f"\nMINORITY class (Gradual, class 0) probabilities:")
print(f"  Mean: {np.mean(minority_proba):.4f}, Median: {np.median(minority_proba):.4f}")
print(f"  Range: [{np.min(minority_proba):.4f}, {np.max(minority_proba):.4f}]")
print(f"  Std: {np.std(minority_proba):.4f}")
print(f"  Samples with prob < 0.5: {(minority_proba < 0.5).sum()} / {len(minority_proba)} ({(minority_proba < 0.5).sum()/len(minority_proba)*100:.1f}%)")
print(f"  Samples with prob < 0.3: {(minority_proba < 0.3).sum()} / {len(minority_proba)} ({(minority_proba < 0.3).sum()/len(minority_proba)*100:.1f}%)")

print(f"\nMAJORITY class (Abrupt, class 1) probabilities:")
print(f"  Mean: {np.mean(majority_proba):.4f}, Median: {np.median(majority_proba):.4f}")
print(f"  Range: [{np.min(majority_proba):.4f}, {np.max(majority_proba):.4f}]")
print(f"  Std: {np.std(majority_proba):.4f}")
print(f"  Samples with prob > 0.5: {(majority_proba > 0.5).sum()} / {len(majority_proba)} ({(majority_proba > 0.5).sum()/len(majority_proba)*100:.1f}%)")
print(f"  Samples with prob > 0.7: {(majority_proba > 0.7).sum()} / {len(majority_proba)} ({(majority_proba > 0.7).sum()/len(majority_proba)*100:.1f}%)")

# Check if probabilities are well-separated
separation = np.mean(majority_proba) - np.mean(minority_proba)
print(f"\nProbability separation (majority mean - minority mean): {separation:.4f}")
if separation < 0.1:
    print("⚠️  WARNING: Very small probability separation - model may not be learning!")

# 6. Confusion matrix
print("\n5. CONFUSION MATRIX")
print("-" * 80)
cm = confusion_matrix(y_test, y_test_pred)
tn, fp, fn, tp = cm.ravel()
print(f"                Predicted")
print(f"                0      1")
print(f"Actual  0    {tn:5d}  {fp:5d}")
print(f"        1    {fn:5d}  {tp:5d}")
print(f"\nTrue Negatives (TN): {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Positives (TP): {tp}")

# 7. Key metrics
print("\n6. KEY METRICS")
print("-" * 80)
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = precision_score(y_test, y_test_pred, zero_division=0)
recall = recall_score(y_test, y_test_pred, zero_division=0)
f1 = f1_score(y_test, y_test_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_test_proba)
pr_auc = average_precision_score(y_test, y_test_proba)

print(f"Accuracy:           {accuracy:.4f}")
print(f"Precision:          {precision:.4f}")
print(f"Recall:             {recall:.4f}")
print(f"F1-score:           {f1:.4f}")
print(f"ROC-AUC:            {roc_auc:.4f}")
print(f"PR-AUC:             {pr_auc:.4f}")

# Compare to naive baseline
print(f"\nNaive baseline (always predict class 1):")
naive_precision = (y_test == 1).sum() / len(y_test)
naive_recall = 1.0
naive_f1 = 2 * naive_precision / (1 + naive_precision)
print(f"  Accuracy:  {naive_accuracy:.4f}")
print(f"  Precision: {naive_precision:.4f}")
print(f"  Recall:    {naive_recall:.4f}")
print(f"  F1-score:  {naive_f1:.4f}")

# 8. Check for data leakage
print("\n7. DATA LEAKAGE CHECK")
print("-" * 80)
# Check for features that might be perfectly correlated with target
high_corr_features = []
for col in X_test.columns:
    if X_test[col].dtype in [np.float64, np.int64]:
        corr = abs(X_test[col].corr(y_test))
        if corr > 0.8:  # Very high correlation
            high_corr_features.append((col, corr))

if high_corr_features:
    print("⚠️  WARNING: Found features with very high correlation (>0.8) with target:")
    for col, corr in sorted(high_corr_features, key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {col}: {corr:.4f}")
else:
    print("No features with correlation >0.8 found.")

# Check for features that might directly encode the class
print("\nChecking for features that might directly encode the class...")
for col in X_test.columns:
    if X_test[col].dtype in [np.float64, np.int64]:
        # Check if feature values perfectly separate classes
        class_0_values = set(X_test[y_test == 0][col].dropna().unique())
        class_1_values = set(X_test[y_test == 1][col].dropna().unique())
        if len(class_0_values) > 0 and len(class_1_values) > 0:
            overlap = len(class_0_values & class_1_values)
            total_unique = len(class_0_values | class_1_values)
            if overlap / total_unique < 0.1 and total_unique < 20:
                print(f"⚠️  SUSPICIOUS: {col} - very little overlap between classes")
                print(f"   Class 0 unique values: {sorted(list(class_0_values))[:10]}")
                print(f"   Class 1 unique values: {sorted(list(class_1_values))[:10]}")

# 9. Feature importance check
print("\n8. TOP FEATURE IMPORTANCES")
print("-" * 80)
feature_importance = model.feature_importances_
feature_names = X_test.columns
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("Top 15 most important features:")
for i, row in importance_df.head(15).iterrows():
    print(f"  {row['feature']:40s} {row['importance']:.4f}")

# Check if top features make sense
top_features = importance_df.head(5)['feature'].tolist()
print(f"\nTop 5 features: {', '.join(top_features)}")

print("\n" + "="*80)
print("DIAGNOSTIC COMPLETE")
print("="*80)

