"""Error analysis for the v5 stage-2 verifier on the held-out fold (out-of-fold, 2 folds keyed by S1)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import polars as pl
import lightgbm as lgb

cfg = json.load(open("data/stage2v5_final.json"))
t = pl.read_parquet("data/stage2v3_train.parquet")
X = t.select(cfg["feats"]).to_numpy().astype(np.float32); y = t["label"].to_numpy()
fold = (t["s1_id"].hash(seed=21) % 2).to_numpy()
p = {k: v for k, v in cfg["lgb"].items() if k != "rounds"}
params = dict(objective="binary", verbose=-1, num_threads=8, bagging_freq=1, **p)
prob = np.zeros(len(y))
for k in (0, 1):
    m = lgb.train(params, lgb.Dataset(X[fold != k], y[fold != k]), cfg["lgb"]["rounds"])
    prob[fold == k] = m.predict(X[fold == k])
t = t.with_columns(pl.Series("prob", prob))
t.select("qid", "rid", "entity_id", "s1_id", "label", "prob").write_parquet("data/v5_oof.parquet")
thr = cfg["thr"]
acc = t.filter(pl.col("prob") > thr); fp = acc.filter(pl.col("label") == 0); fn = t.filter((pl.col("prob") <= thr) & (pl.col("label") == 1))
print(f"threshold {thr}: accepted {acc.height}, false accepts {fp.height}, rejected true {fn.height}")
cols = ["a_q_empty", "dm_legal_conflict", "dm_num_equal", "dm_num_missing", "dm_num_one_digit", "dm_num_different",
        "dm_word_sub", "t1_n_clean_ratio", "t1_a_tset", "t1_s_name_dup", "kc_all", "is_s3"]
cols = [c for c in cols if c in t.columns]
summ = lambda d: {c: round(float(d[c].cast(pl.Float64).mean()), 3) for c in cols}
print("\nmean feature values:")
for name, d in (("TRUE accepted", acc.filter(pl.col("label") == 1)), ("FALSE accepts", fp), ("REJECTED true", fn)):
    print(f"  {name:15s} n={d.height:8d}", summ(d))
print("\nprobability bands of errors:")
for lo, hi in ((0, .1), (.1, .3), (.3, thr), (thr, .9), (.9, .99), (.99, 1.01)):
    b = t.filter((pl.col("prob") >= lo) & (pl.col("prob") < hi))
    print(f"  p in [{lo:.2f},{hi:.2f}): n={b.height:8d}  true share {b['label'].mean() if b.height else 0:.3f}")
