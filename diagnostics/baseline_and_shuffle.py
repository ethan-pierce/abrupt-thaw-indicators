"""Move 1 (floor) + Move 2 (shuffle-label) for the abrupt/non-abrupt classifier.

Establishes the floor the reported random-split AUC must clear, and
attacks the too-good hypothesis with a shuffle-label probe. Uses the SAME random,
class-stratified split the training pipeline uses (train_xgboost.py), so numbers
are comparable to the author's evaluation.

Positive class = 1 (Non-abrupt, minority) to match the pipeline's predict_proba[:,1]
and average_precision_score convention. AUC-PR chance floor = positive prevalence.

Run: poetry run python diagnostics/baseline_and_shuffle.py
"""
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from _data import load

SEED = 42
X, y, lat, lon = load(verify=True)
prevalence = y.mean()  # positive (Non-abrupt) prevalence == AUC-PR chance floor
print(f"n={len(y)}  features={X.shape[1]}  positive(Non-abrupt) prevalence={prevalence:.4f}")
print(f"AUC-PR chance floor = {prevalence:.4f}   AUC-ROC chance = 0.5000\n")

# --- pipeline's own split: default_rng(42).integers(0,100), stratified, 30% test
rng = np.random.default_rng(SEED)
split_seed = int(rng.integers(0, 100))
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.3, random_state=split_seed, shuffle=True, stratify=y)
print(f"random stratified split (seed={split_seed}): train={len(ytr)} test={len(yte)}\n")


def ev(name, p):
    print(f"{name:<34} AUC-ROC={roc_auc_score(yte, p):.4f}  "
          f"AUC-PR={average_precision_score(yte, p):.4f}  "
          f"Brier={brier_score_loss(yte, p):.4f}")


def xgb_model(spw):
    return xgb.XGBClassifier(
        n_estimators=300, max_depth=5, min_child_weight=20, learning_rate=0.05,
        reg_lambda=50, gamma=1, subsample=0.8, colsample_bytree=0.8,
        objective='binary:logistic', eval_metric='aucpr', tree_method='hist',
        scale_pos_weight=spw, random_state=SEED)


print("=" * 74)
print("MOVE 1 - THE FLOOR (trivial + simple baselines, positive class = Non-abrupt)")
print("=" * 74)
# Constant / majority
d = DummyClassifier(strategy='most_frequent').fit(Xtr, ytr)
ev("majority-class (predict Abrupt)", d.predict_proba(Xte)[:, 1])
d = DummyClassifier(strategy='stratified', random_state=SEED).fit(Xtr, ytr)
ev("stratified-random", d.predict_proba(Xte)[:, 1])
d = DummyClassifier(strategy='prior').fit(Xtr, ytr)
ev("prior (constant = prevalence)", d.predict_proba(Xte)[:, 1])
# Logistic regression (real but simple)
lr = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                   LogisticRegression(max_iter=2000, class_weight='balanced'))
lr.fit(Xtr, ytr)
ev("logistic regression (balanced)", lr.predict_proba(Xte)[:, 1])
# Single-feature decision stump (depth 1) - simplest "real" learner
stump = DecisionTreeClassifier(max_depth=1, class_weight='balanced', random_state=SEED).fit(Xtr, ytr)
ev("decision stump (depth=1)", stump.predict_proba(Xte)[:, 1])

print("\n" + "=" * 74)
print("REAL MODEL (representative grid config) on the same split")
print("=" * 74)
spw = (ytr == 0).sum() / (ytr == 1).sum()
m = xgb_model(spw).fit(Xtr, ytr)
p_real = m.predict_proba(Xte)[:, 1]
ev("XGBoost (real labels)", p_real)

print("\n" + "=" * 74)
print("MOVE 2 - SHUFFLE-LABEL PROBE (permute TRAIN labels, refit, score real test)")
print("  real signal -> AUC collapses to chance (0.5 / prevalence).")
print("  survives high -> leakage or a bug feeding the target through a side channel.")
print("=" * 74)
aucs, aps = [], []
for i in range(5):
    r = np.random.default_rng(1000 + i)
    y_shuf = ytr.to_numpy().copy()
    r.shuffle(y_shuf)
    spw_s = (y_shuf == 0).sum() / max((y_shuf == 1).sum(), 1)
    ms = xgb_model(spw_s).fit(Xtr, y_shuf)
    ps = ms.predict_proba(Xte)[:, 1]
    aucs.append(roc_auc_score(yte, ps))
    aps.append(average_precision_score(yte, ps))
print(f"shuffled AUC-ROC: mean={np.mean(aucs):.4f} sd={np.std(aucs):.4f}  (expect ~0.50)")
print(f"shuffled AUC-PR : mean={np.mean(aps):.4f} sd={np.std(aps):.4f}  (expect ~{prevalence:.4f})")
