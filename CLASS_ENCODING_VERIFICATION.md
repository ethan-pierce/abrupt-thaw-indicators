# Class Encoding Verification Report

## Standard Encoding (CONFIRMED CORRECT)
- **Class 0 = Gradual Thaw** (negative class, ~6% of data)
- **Class 1 = Abrupt Thaw** (positive class, ~94% of data)

## File-by-File Verification

### ✅ data/build_feature_table.py
- **Line 18**: `thawdb['Class'] = np.where(thawdb['ThawType'] == 'Abrupt', 1, 0)`
- **Status**: ✅ CORRECT - Abrupt = 1, Gradual = 0

### ✅ data/clean_feature_table.py
- **Line 8**: `feats['Class'] = np.where(feats['ThawType'] == 'Abrupt', 1, 0)`
- **Status**: ✅ CORRECT - Abrupt = 1, Gradual = 0
- **Comment**: "Abrupt = 1 (positive class), Gradual = 0"

### ✅ models/train_xgboost.py
- **Line 79-80**: `predict_proba(X_train)[:, 1]` and `predict_proba(X_test)[:, 1]`
  - **Status**: ✅ CORRECT - Using index 1 for abrupt (positive class)
- **Line 149**: `target_names=['Gradual Thaw', 'Abrupt Thaw']`
  - **Status**: ✅ CORRECT - Index 0 = Gradual, Index 1 = Abrupt
- **Line 252**: `for class_label, class_name in [(0, 'Gradual Thaw'), (1, 'Abrupt Thaw')]`
  - **Status**: ✅ CORRECT - 0 = Gradual, 1 = Abrupt
- **Line 306**: Comment says "class 1 = Abrupt"
  - **Status**: ✅ CORRECT
- **Line 377**: `display_labels=['Gradual Thaw', 'Abrupt Thaw']` for confusion matrix
  - **Status**: ✅ CORRECT - [class 0, class 1] order
- **Line 267-268**: `calibration_curve(y_test, y_test_proba, ...)`
  - **Status**: ✅ CORRECT - y_test_proba is for class 1 (abrupt)

### ✅ models/predict.py
- **Line 97**: `probabilities = model.predict_proba(feature_array)[:, 1]`
  - **Status**: ✅ CORRECT - Using index 1 for abrupt
- **Line 114**: `Abrupt thaw predictions: {(predictions == 1).sum()}`
  - **Status**: ✅ CORRECT - Class 1 = Abrupt
- **Line 115**: `Gradual thaw predictions: {(predictions == 0).sum()}`
  - **Status**: ✅ CORRECT - Class 0 = Gradual
- **Line 132**: `'probability_description': 'Probability of abrupt thaw (class 1)'`
  - **Status**: ✅ CORRECT
- **Line 133**: `'prediction_description': 'Binary prediction: 0=Gradual Thaw, 1=Abrupt Thaw'`
  - **Status**: ✅ CORRECT
- **Line 237**: `cbar2.set_ticklabels(['Gradual', 'Abrupt'])` with comment `# 0=Gradual, 1=Abrupt`
  - **Status**: ✅ CORRECT

### ✅ models/shap_values.py
- **Line 34-35**: Comments say "abrupt thaw (class 1) and gradual thaw (class 0)"
- **Line 36**: `abrupt_indices = np.where(y_test == 1)[0]`
  - **Status**: ✅ CORRECT - Class 1 = Abrupt
- **Line 37**: `gradual_indices = np.where(y_test == 0)[0]`
  - **Status**: ✅ CORRECT - Class 0 = Gradual

## Summary

**ALL FILES ARE CONSISTENT** ✅

The encoding is standardized across all files:
- **Class 0 = Gradual Thaw** (minority class, ~6%)
- **Class 1 = Abrupt Thaw** (majority class, ~94%, positive class)

All probability extractions use `[:, 1]` for abrupt thaw probabilities.
All labels and comments correctly identify class 1 as abrupt and class 0 as gradual.

## Notes

1. The model uses `base_score = 0.94` which matches the ~94% abrupt class distribution
2. `scale_pos_weight = 0.064` is approximately 6/94, the ratio of negative to positive samples
3. All evaluation metrics (ROC, PR, calibration) correctly use class 1 probabilities
4. Confusion matrices use the correct label order: [Gradual (0), Abrupt (1)]

