"""Investigate the SHAP values of the best XGBoost model for abrupt thaw features."""

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
    test_size = 0.3, 
    random_state = rng.integers(0, 100), 
    shuffle = True, 
    stratify = target
)
dtrain = xgb.DMatrix(X_train, label = y_train, missing = np.nan)
dtest = xgb.DMatrix(X_test, label = y_test, missing = np.nan)

model = xgb.XGBClassifier()
model.load_model('models/model.json')

# Calculate SHAP values
# For binary classification, we want SHAP values for class 0 (Abrupt - majority class)
# Compute SHAP values in probability space (requires interventional feature perturbation with background data)
# Use a large background dataset for accuracy (more samples = more accurate but slower)
background_data = X_train.sample(min(5000, len(X_train)), random_state=42)
explainer = shap.TreeExplainer(model, background_data, model_output='probability', feature_perturbation='interventional')
shap_values = explainer(X_test)


# For binary classification with XGBClassifier, TreeExplainer returns 2D array (samples, features)
# representing the positive class (Class 1 = Gradual)
# Transform to Class 0 (Abrupt) SHAP values:
# Since probabilities sum to 1: P(Class 0) = 1 - P(Class 1)
# Therefore: SHAP(Class 0) = -SHAP(Class 1)
base_value_class_1 = explainer.expected_value
shap_values_class_1 = shap_values

# Transform to Class 0 (Abrupt) SHAP values
# Base value for Class 0 is 1 - base_value_class_1
# SHAP values for Class 0 are negative of Class 1 SHAP values
if np.isscalar(base_value_class_1):
    base_value_class_0 = 1 - base_value_class_1
else:
    base_value_class_0 = 1 - base_value_class_1[0] if len(base_value_class_1) > 0 else 1 - base_value_class_1

# Create new Explanation object for Class 0 with negated SHAP values
shap_values_abrupt_class = shap.Explanation(
    values=-shap_values_class_1.values,
    base_values=np.full(len(X_test), base_value_class_0),
    data=shap_values_class_1.data,
    feature_names=shap_values_class_1.feature_names
)

# Find indices for abrupt thaw (class 0)
# Note: Encoding is: Abrupt=0 (majority class), Gradual=1 (minority class)
abrupt_indices = np.where(y_test == 0)[0]

base_values_rounded = base_value_class_0.round(2)
shap_values_rounded = shap_values_abrupt_class.values.round(2)
X_test_rounded = X_test.round(2)


# idx = abrupt_indices[100]

# im = shap.plots.force(
#     base_values_rounded,
#     shap_values_rounded[idx],
#     X_test_rounded.iloc[idx],
#     contribution_threshold = 0.1,
#     matplotlib = True,
#     show = False,
#     figsize = (12, 4)
# )
# plt.tight_layout()
# plt.savefig('output/shap_force_plot_abrupt.png', dpi = 300)
# plt.show()


# Use SHAP values for Abrupt class (class 0) for main plots
# This shows what features drive predictions toward Abrupt thaw
shap_values_for_plot = shap_values_abrupt_class.values if hasattr(shap_values_abrupt_class, 'values') else shap_values_abrupt_class

# # ============================================================================
# # COMPUTE SHAP INTERACTION VALUES
# # ============================================================================
# print("\n" + "="*80)
# print("COMPUTING SHAP INTERACTION VALUES")
# print("="*80)
# print("This may take a while for large datasets...")

# # Compute SHAP interaction values (for class 0 - Abrupt)
# # Interaction values show how features interact with each other
# shap_interaction_values = explainer.shap_interaction_values(X_test)

# # Handle binary classification: extract interactions for class 0 (Abrupt)
# if len(shap_interaction_values.shape) == 4:
#     # Shape: (samples, features, features, classes)
#     shap_interaction_values_abrupt = shap_interaction_values[:, :, :, 0]
# else:
#     # Shape: (samples, features, features)
#     shap_interaction_values_abrupt = shap_interaction_values

# print(f"Interaction values shape: {shap_interaction_values_abrupt.shape}")
# print("(samples, features, features) - each [i,j] entry shows interaction between feature i and j")

# # Compute mean absolute interaction strength for each feature pair
# # Average across all samples to get overall interaction strength
# mean_interaction_strength = np.abs(shap_interaction_values_abrupt).mean(axis=0)
# # Shape: (features, features) - symmetric matrix

# # Get feature names
# feature_names = X_test.columns.tolist()
# n_features = len(feature_names)

# # Create a list of all feature pairs with their interaction strengths
# interaction_pairs = []
# for i in range(n_features):
#     for j in range(i+1, n_features):  # Only upper triangle to avoid duplicates
#         interaction_strength = mean_interaction_strength[i, j]
#         interaction_pairs.append({
#             'feature_i': feature_names[i],
#             'feature_j': feature_names[j],
#             'interaction_strength': interaction_strength,
#             'index_i': i,
#             'index_j': j
#         })

# # Sort by interaction strength (descending)
# interaction_pairs_sorted = sorted(interaction_pairs, key=lambda x: x['interaction_strength'], reverse=True)

# # Display top interactions
# print("\n" + "-"*80)
# print("TOP 20 STRONGEST FEATURE INTERACTIONS")
# print("-"*80)
# print(f"{'Rank':<6} {'Feature 1':<35} {'Feature 2':<35} {'Strength':<12}")
# print("-"*80)

# for rank, pair in enumerate(interaction_pairs_sorted[:20], 1):
#     print(f"{rank:<6} {pair['feature_i']:<35} {pair['feature_j']:<35} {pair['interaction_strength']:.6f}")

# # Save top interactions to CSV
# output_dir = Path(__file__).parent.parent / 'output'
# output_dir.mkdir(exist_ok=True)
# interactions_df = pd.DataFrame(interaction_pairs_sorted)
# interactions_df.to_csv(output_dir / 'shap_interaction_strengths.csv', index=False)
# print(f"\nAll interaction strengths saved to: {output_dir / 'shap_interaction_strengths.csv'}")

# # Create a heatmap of top interactions
# print("\nCreating interaction strength heatmap...")
# top_n = 15  # Show top N features by total interaction strength
# # Calculate total interaction strength per feature (sum of all interactions)
# feature_total_interactions = mean_interaction_strength.sum(axis=0) + mean_interaction_strength.sum(axis=1)
# top_feature_indices = np.argsort(feature_total_interactions)[-top_n:][::-1]
# top_feature_names = [feature_names[i] for i in top_feature_indices]

# # Extract submatrix for top features
# interaction_submatrix = mean_interaction_strength[np.ix_(top_feature_indices, top_feature_indices)]

# # Create heatmap
# fig, ax = plt.subplots(figsize=(14, 12))
# im = ax.imshow(interaction_submatrix, cmap='YlOrRd', aspect='auto')
# ax.set_xticks(range(len(top_feature_names)))
# ax.set_yticks(range(len(top_feature_names)))
# ax.set_xticklabels(top_feature_names, rotation=45, ha='right')
# ax.set_yticklabels(top_feature_names)
# ax.set_title(f'SHAP Interaction Strength Matrix (Top {top_n} Features)', fontsize=14, fontweight='bold')
# plt.colorbar(im, ax=ax, label='Mean Absolute Interaction Strength')
# plt.tight_layout()
# plt.savefig(output_dir / 'shap_interaction_heatmap.png', dpi=300)
# print(f"Interaction heatmap saved to: {output_dir / 'shap_interaction_heatmap.png'}")
# plt.show()

# print("\n" + "="*80)
# print("INTERACTION ANALYSIS COMPLETE")
# print("="*80)

# Dependence plot for Slope
var = "Slope"
shap.dependence_plot(var, shap_values_for_plot, X_test, interaction_index=var, show=False)
plt.tight_layout()
plt.savefig('output/shap_dependence_plot_slope.png', dpi=300)
plt.show()

# Dependence plot for Mean Curvature (2 km)
var = "Mean curvature (500 m)"
shap.dependence_plot(var, shap_values_for_plot, X_test, interaction_index=var, show=False)
plt.tight_layout()
plt.savefig('output/shap_dependence_plot_curvature.png', dpi=300)
plt.show()

# Dependence plot for Nitrogen (0-30 cm)
var = "Nitrogen (0-30 cm)"
shap.dependence_plot(var, shap_values_for_plot, X_test, interaction_index="Nitrogen (30-200 cm)", show=False)
plt.tight_layout()
plt.savefig('output/shap_dependence_plot_nitrogen.png', dpi=300)
plt.show()

# Dependence plot for Silt (0-30 cm)
var = "Silt (0-30 cm)"
shap.dependence_plot(var, shap_values_for_plot, X_test, interaction_index="Silt (30-200 cm)", show=False)
plt.tight_layout()
plt.savefig('output/shap_dependence_plot_sil.png', dpi=300)
plt.show()

# Dependence plot for Trend in SWE
var = "Trend in SWE"
shap.dependence_plot(var, shap_values_for_plot, X_test, interaction_index=var, show=False)
plt.tight_layout()
plt.savefig('output/shap_dependence_plot_trend_swe.png', dpi=300)
plt.show()

# Dependence plot for Mean Annual SWE
var = "Mean Annual SWE"
shap.dependence_plot(var, shap_values_for_plot, X_test, interaction_index=var, show=False)
plt.tight_layout()
plt.savefig('output/shap_dependence_plot_mean_annual_swe.png', dpi=300)
plt.show()

# Dependence plot for Annual Precipitation
var = "Annual Precipitation"
shap.dependence_plot(var, shap_values_for_plot, X_test, interaction_index=var, show=False)
plt.tight_layout()
plt.savefig('output/shap_dependence_plot_annual_precip.png', dpi=300)
plt.show()

# Summary plot using SHAP values for Abrupt class (class 0)
shap.summary_plot(shap_values_abrupt_class, max_display = 10, show = False)
plt.tight_layout()
plt.savefig('output/shap_summary_plot.png', dpi = 300)
plt.show()

# Create beeswarm plot for abrupt thaw points only
# Using SHAP values for Abrupt class (class 0) to see what drives Abrupt predictions
shap_values_abrupt_samples = shap_values_abrupt_class[abrupt_indices]

# Beeswarm plot for actual abrupt thaw points (using Abrupt class SHAP values)
shap.plots.beeswarm(shap_values_abrupt_samples, max_display=10, show=False)
plt.tight_layout()
plt.savefig('output/shap_beeswarm_abrupt.png', dpi=300)
plt.show()
