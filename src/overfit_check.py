"""Thorough overfitting / regularisation audit for the v8 models (leaderboard 0.973).

1. Stage 1: training rows vs early-stop hold-out (same distribution, never fitted)
2. Stage 1: learning curve on the hold-out (error vs number of trees)
3. Stage 1: early-stop hold-out vs the held-out 20% S1 fold (businesses never seen in training)
4. Stage 2: learning curve, fit on half A, error on A (train) and untouched half B
5. Stage 2: tuner A(CV) vs B(untouched) F0.5
6. Threshold stability (F0.5 curve on untouched half B)
7. Per-country F0.5 on untouched half B
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.metrics import log_loss, roc_auc_score
from metric import f05
from stage2 import CAP

P = lambda *a: print(*a, flush=True)
P("=== 1-2. STAGE 1 (pair model): train vs hold-out, learning curve ===")
m1 = lgb.Booster(model_file="data/model.txt"); names = m1.feature_name()
F = pl.read_parquet("data/train_feats.parquet", columns=["qid", "label", *names])
es = (F["qid"].hash(seed=9) % 10 == 0).to_numpy(); y = F["label"].to_numpy()
tr = np.random.RandomState(0).choice(np.where(~es)[0], 600_000, replace=False); ho = np.where(es)[0]
X = F.select(names).to_numpy().astype(np.float32); del F
for k in (200, 500, 1000, m1.num_trees()):
    pt = m1.predict(X[tr], num_iteration=k, num_threads=8); ph = m1.predict(X[ho], num_iteration=k, num_threads=8)
    P(f"  trees {k:5d}: TRAIN logloss {log_loss(y[tr], pt):.5f} AUC {roc_auc_score(y[tr], pt):.6f} | "
      f"HOLD-OUT logloss {log_loss(y[ho], ph):.5f} AUC {roc_auc_score(y[ho], ph):.6f}")
hold_auc = roc_auc_score(y[ho], ph); del X

P("\n=== 3. STAGE 1 on businesses never seen in training (held-out 20% S1 fold, out-of-sample scores) ===")
from pipeline import load_gt, val_fold
S = pl.read_parquet("data/train_scored.parquet", columns=["qid", "rid", "p"])
s1 = pl.read_parquet("data/train_s1.parquet", columns=["entity_id"]).with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
q = pl.concat([pl.read_parquet(f"data/train_s{k}.parquet", columns=["entity_id"]) for k in (2, 3)]).with_row_index("qid").with_columns(pl.col("qid").cast(pl.UInt32))
held = s1.filter(val_fold(pl.col("entity_id"))).select("rid", pl.col("entity_id").alias("cand"))
S = S.join(held, on="rid").join(q, on="qid").join(load_gt().rename({"s1_id": "true"}), on="entity_id", how="left")
S = S.with_columns((pl.col("true") == pl.col("cand")).fill_null(False).cast(pl.Int8).alias("y")).sample(3_000_000, seed=1)
auc_new = roc_auc_score(S["y"].to_numpy(), S["p"].to_numpy())
P(f"  pairs {S.height}: AUC {auc_new:.6f} logloss {log_loss(S['y'].to_numpy(), np.clip(S['p'].to_numpy(), 1e-7, 1 - 1e-7)):.5f}"
  f"   (early-stop hold-out AUC {hold_auc:.6f})")
del S

P("\n=== 4. STAGE 2 (v8 verifier): learning curve, fit on half A ===")
cfg = json.load(open("data/stage2v8_final.json"))
t = pl.read_parquet("data/stage2v8_train.parquet")
half = (t["s1_id"].hash(seed=77) % 2).to_numpy(); A, B = half == 0, half == 1
X = t.select(cfg["feats"]).to_numpy().astype(np.float32); y = t["label"].to_numpy()
p = {k: v for k, v in cfg["lgb"].items() if k != "rounds"}
m2 = lgb.train(dict(objective="binary", verbose=-1, num_threads=8, bagging_freq=1, **p), lgb.Dataset(X[A], y[A]), cfg["lgb"]["rounds"])
sa = np.random.RandomState(1).choice(np.where(A)[0], 400_000, replace=False)
for k in (100, 300, 600, 900, cfg["lgb"]["rounds"]):
    pa = m2.predict(X[sa], num_iteration=k); pb = m2.predict(X[B], num_iteration=k)
    P(f"  trees {k:5d}: A(train) logloss {log_loss(y[sa], pa):.5f} AUC {roc_auc_score(y[sa], pa):.6f} | "
      f"B(untouched) logloss {log_loss(y[B], pb):.5f} AUC {roc_auc_score(y[B], pb):.6f}")
pB = m2.predict(X[B])

P("\n=== 5. STAGE 2 tuner: A(CV) vs B(untouched) F0.5 ===")
for k, v in cfg.get("report", {}).items():
    P(f"  {k:34s} A(CV) {v['A_cv_f05']:.5f}  B {v['B_f05']:.5f}  gap {v['B_f05'] - v['A_cv_f05']:+.5f}")

P("\n=== 6-7. Threshold stability and per-country F0.5 on untouched half B ===")
tb = t.filter(pl.lit(B)).select("s1_id", "entity_id", "src").with_columns(pl.Series("p", pB))
truth_df = pl.read_parquet("data/stage2_truth.parquet"); s1v = pl.read_parquet("data/stage2_s1v.parquet")["entity_id"]
idsB = s1v.filter((s1v.hash(seed=77) % 2) == 1)
ctry = pl.read_parquet("data/train_s1.parquet", columns=["entity_id", "country"])
truth = {}
for a, b in truth_df.select("s1_id", "entity_id").iter_rows():
    truth.setdefault(a, set()).add(b)
def score(thr, ids):
    d = tb.filter(pl.col("p") > thr).with_columns(pl.col("p").rank("ordinal", descending=True).over(["s1_id", "src"]).alias("r"))
    d = d.filter(((pl.col("src") == "S2") & (pl.col("r") <= CAP["S2"])) | ((pl.col("src") == "S3") & (pl.col("r") <= CAP["S3"])))
    pred = {}
    for a, b in d.select("s1_id", "entity_id").iter_rows():
        pred.setdefault(a, set()).add(b)
    return f05(pred, truth, ids)
P("  " + "  ".join(f"thr {th:.2f}: {score(th, idsB.to_list()):.5f}" for th in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9)))
cb = pl.DataFrame({"entity_id": idsB}).join(ctry, on="entity_id")
for c in ("US", "India"):
    P(f"  {c:6s} (thr {cfg['thr']:.3f}): F0.5 {score(cfg['thr'], cb.filter(pl.col('country') == c)['entity_id'].to_list()):.5f}")
P("DONE")
