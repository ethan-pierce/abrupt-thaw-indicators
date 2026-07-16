"""Spatial-leakage trace: is the high random-split AUC earned, or an artifact of the split?

The Thaw Database is lake-dominated and road-clustered (SCOPE.md). Coarse GEE
feature extraction makes nearby points near-identical in feature space with the
same label. A random train/test split then scatters members of the same spatial
cluster across both sides, so the test set contains near-copies of training rows
-- classic geospatial leakage that inflates AUC.

Two checks:
  (A) Near-duplicate census: for each test row under the random split, is there a
      train row within a tiny feature distance AND sharing the label? Also measure
      geographic near-neighbours across the split.
  (B) Random KFold vs spatial GroupKFold. Group points into spatial blocks so a
      whole block is held out together. If AUC drops sharply under spatial CV, the
      random-split number was borrowing information across the split.

Run: poetry run python diagnostics/spatial_leakage.py
"""
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from _data import load

SEED = 42
X, y, lat, lon = load(verify=True)
yv = y.to_numpy()
print(f"n={len(y)}  features={X.shape[1]}  positive(Non-abrupt) prevalence={y.mean():.4f}\n")


def xgb_model(spw):
    return xgb.XGBClassifier(
        n_estimators=300, max_depth=5, min_child_weight=20, learning_rate=0.05,
        reg_lambda=50, gamma=1, subsample=0.8, colsample_bytree=0.8,
        objective='binary:logistic', eval_metric='aucpr', tree_method='hist',
        scale_pos_weight=spw, random_state=SEED)


# ============================================================
# (A) NEAR-DUPLICATE CENSUS under the pipeline's random split
# ============================================================
print("=" * 74)
print("(A) NEAR-DUPLICATE CENSUS (random split)")
print("=" * 74)
rng = np.random.default_rng(SEED)
split_seed = int(rng.integers(0, 100))
idx = np.arange(len(y))
itr, ite = train_test_split(idx, test_size=0.3, random_state=split_seed,
                            shuffle=True, stratify=y)

# Feature-space nearest train neighbour for each test row (standardized, imputed).
Xf = SimpleImputer(strategy='median').fit_transform(X)
Xf = StandardScaler().fit_transform(Xf)
nn = NearestNeighbors(n_neighbors=1).fit(Xf[itr])
dist, nbr = nn.kneighbors(Xf[ite])
dist = dist.ravel()
train_lbl_of_nbr = yv[itr][nbr.ravel()]
same_label = (train_lbl_of_nbr == yv[ite])
for tol in [1e-9, 1e-3, 0.01, 0.1]:
    close = dist <= tol
    print(f"  feat-dist <= {tol:<7g}: {close.sum():5d}/{len(ite)} test rows "
          f"({100*close.mean():5.1f}%) have a near-twin in train; "
          f"of those {100*same_label[close].mean() if close.any() else 0:5.1f}% share its label")

# Geographic nearest train neighbour (great-circle-ish in degrees; AK ~ fine for ranking).
coords = np.column_stack([lat, lon])
nn_geo = NearestNeighbors(n_neighbors=1).fit(coords[itr])
gdist, _ = nn_geo.kneighbors(coords[ite])
gdist = gdist.ravel()
for tol_km in [0.1, 0.5, 1.0, 5.0]:
    tol_deg = tol_km / 111.0
    close = gdist <= tol_deg
    print(f"  geo-dist  <= {tol_km:>4}km: {close.sum():5d}/{len(ite)} test rows "
          f"({100*close.mean():5.1f}%) sit within {tol_km}km of a train point")

# ============================================================
# (B) RANDOM KFOLD vs SPATIAL GROUPKFOLD
# ============================================================
print("\n" + "=" * 74)
print("(B) RANDOM CV vs SPATIAL-BLOCK CV (5-fold)")
print("=" * 74)


def cv_scores(splitter, groups=None):
    aucs, aps = [], []
    for tr, te in splitter.split(X, y, groups):
        spw = (yv[tr] == 0).sum() / max((yv[tr] == 1).sum(), 1)
        m = xgb_model(spw).fit(X.iloc[tr], y.iloc[tr])
        p = m.predict_proba(X.iloc[te])[:, 1]
        aucs.append(roc_auc_score(yv[te], p))
        aps.append(average_precision_score(yv[te], p))
    return np.array(aucs), np.array(aps)


# Random stratified KFold (matches the spirit of the pipeline's CV)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
r_auc, r_ap = cv_scores(skf)
print(f"random  StratifiedKFold : AUC-ROC {r_auc.mean():.4f}+-{r_auc.std():.4f}   "
      f"AUC-PR {r_ap.mean():.4f}+-{r_ap.std():.4f}")

# Spatial blocks: bin coordinates into a grid, hold whole cells out together.
for cell_deg in [0.5, 1.0, 2.0]:
    lat_bin = np.floor(lat / cell_deg).astype(int)
    lon_bin = np.floor(lon / cell_deg).astype(int)
    blocks = lat_bin * 10000 + lon_bin
    n_blocks = len(np.unique(blocks))
    gkf = GroupKFold(n_splits=5)
    s_auc, s_ap = cv_scores(gkf, groups=blocks)
    print(f"spatial GroupKFold {cell_deg}deg ({n_blocks:4d} blocks): "
          f"AUC-ROC {s_auc.mean():.4f}+-{s_auc.std():.4f}   "
          f"AUC-PR {s_ap.mean():.4f}+-{s_ap.std():.4f}")

print("\nInterpretation: a large random->spatial drop means the random-split AUC is "
      "\nborrowing signal from near-duplicate neighbours across the split (leakage),"
      "\nnot measuring generalization to new locations.")
