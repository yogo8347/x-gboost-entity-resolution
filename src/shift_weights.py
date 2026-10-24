"""Covariate-shift correction (adversarial validation) for stage 2.

The validation rows (train split, held-out fold) and the test rows come from different mixes: test has more decoys
and contains France. A classifier is trained to tell validation rows from test rows using the stage-2 inputs (no
labels anywhere). For each validation row, w = P(test|x) / P(val|x) * n_val / n_test is the density ratio: how much
more typical of test that row is. Training stage 2 with these weights makes it optimise for the test mix.
Weights are clipped to [0.1, 10] and normalised to mean 1.

  python src/shift_weights.py <final.json> <val_table> <test_table> <out_weights.parquet>
"""
import json, sys
import numpy as np
import polars as pl
import lightgbm as lgb

cfg_p, va_p, te_p, out = sys.argv[1:5]
feats = json.load(open(cfg_p))["feats"]
va = pl.read_parquet(va_p, columns=["qid", *feats])
te = pl.read_parquet(te_p, columns=feats).sample(n=min(va.height * 2, 4_000_000), seed=1)
X = np.vstack([va.select(feats).to_numpy().astype(np.float32), te.to_numpy().astype(np.float32)])
y = np.concatenate([np.zeros(va.height), np.ones(te.height)])
fold = np.random.RandomState(0).randint(0, 2, len(y))
params = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=500, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, num_threads=8, verbose=-1)
p = np.zeros(len(y))
for k in (0, 1):                                   # out-of-fold so no row scores itself
    m = lgb.train(params, lgb.Dataset(X[fold != k], y[fold != k]), 300)
    p[fold == k] = m.predict(X[fold == k])
from sklearn.metrics import roc_auc_score
print(f"adversarial AUC (0.5 = identical distributions): {roc_auc_score(y, p):.4f}")
pv = np.clip(p[: va.height], 1e-4, 1 - 1e-4)
w = pv / (1 - pv) * (va.height / te.height)
w = np.clip(w, 0.1, 10.0); w = w / w.mean()
imp = sorted(zip(feats, m.feature_importance("gain")), key=lambda x: -x[1])[:8]
print("most shifted inputs:", [f for f, _ in imp])
print(f"weights: mean {w.mean():.2f}  p10 {np.percentile(w, 10):.2f}  median {np.median(w):.2f}  p90 {np.percentile(w, 90):.2f}  max {w.max():.2f}")
pl.DataFrame({"qid": va["qid"], "w": w.astype(np.float32)}).write_parquet(out)
print("wrote", out)
