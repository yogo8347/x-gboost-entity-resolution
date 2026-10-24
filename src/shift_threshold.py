"""Threshold selection under the test's higher distractor rate (prior shift).

Validation has ~27% distractor records; test has ~37% (estimated from the score distribution). Distractors are the
source of most false merges, so on test every distractor-caused false merge is ~k times as likely:
    k = odds(distractor | test) / odds(distractor | validation)
We re-score out-of-fold stage-2 probabilities with distractor false positives weighted by k and pick the threshold.

  python src/shift_threshold.py <table.parquet> [k ...]
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import polars as pl
import lightgbm as lgb
from stage2 import CAP

table = sys.argv[1] if len(sys.argv) > 1 else "data/stage2_table.parquet"
ks = [float(x) for x in sys.argv[2:]] or [1.0, 1.55, 2.0]
t = pl.read_parquet(table)
FEATS = [c for c, dt in t.schema.items() if c not in ("qid", "rid", "src", "entity_id", "s1_id", "label") and dt.is_numeric()]
X = t.select(FEATS).to_numpy().astype(np.float32); y = t["label"].to_numpy()
fold = (t["s1_id"].hash(seed=21) % 2).to_numpy()
params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.9,
              bagging_fraction=0.8, bagging_freq=1, num_threads=8, verbose=-1)
prob = np.zeros(len(y))
for k in (0, 1):
    m = lgb.train(params, lgb.Dataset(X[fold != k], y[fold != k]), 600)
    prob[fold == k] = m.predict(X[fold == k])

truth = pl.read_parquet("data/stage2_truth.parquet")
ids = pl.read_parquet("data/stage2_s1v.parquet")["entity_id"]
distractors = set(pl.concat([pl.read_parquet(f"data/train_s{k}.parquet", columns=["entity_id"]) for k in (2, 3)])
                  .join(pl.read_csv("dataset/train/train_ground_truth.tsv", separator="\t", infer_schema=False)
                        .with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
                        .drop_nulls().rename({"matched_entity_ids": "entity_id"}), on="entity_id", how="anti")["entity_id"].to_list())
tr_cnt = truth.group_by("s1_id").len().rename({"len": "n_true"})
base = pl.DataFrame({"s1_id": ids}).join(tr_cnt, on="s1_id", how="left").with_columns(pl.col("n_true").fill_null(0))
t = t.select("qid", "s1_id", "entity_id", "src", "label").with_columns(pl.Series("p", prob),
        pl.col("entity_id").is_in(list(distractors)).alias("is_dis"))


def f05_weighted(thr, kd):
    d = t.filter(pl.col("p") > thr)
    d = d.with_columns(pl.col("p").rank("ordinal", descending=True).over(["s1_id", "src"]).alias("r"))
    d = d.filter(((pl.col("src") == "S2") & (pl.col("r") <= CAP["S2"])) | ((pl.col("src") == "S3") & (pl.col("r") <= CAP["S3"])))
    g = d.group_by("s1_id").agg(pl.col("label").sum().alias("tp"),
                                ((pl.col("label") == 0) & ~pl.col("is_dis")).sum().alias("fp_m"),
                                ((pl.col("label") == 0) & pl.col("is_dis")).sum().alias("fp_d"))
    g = base.join(g, on="s1_id", how="left").fill_null(0)
    g = g.with_columns((pl.col("fp_m") + kd * pl.col("fp_d")).alias("fp"))
    g = g.with_columns(
        pl.when((pl.col("n_true") == 0) & (pl.col("tp") + pl.col("fp_m") + pl.col("fp_d") == 0)).then(1.0)
        .when(pl.col("tp") == 0).then(0.0)
        .otherwise(1.25 * (pl.col("tp") / (pl.col("tp") + pl.col("fp"))) * (pl.col("tp") / pl.col("n_true"))
                   / (0.25 * (pl.col("tp") / (pl.col("tp") + pl.col("fp"))) + pl.col("tp") / pl.col("n_true"))).alias("f"))
    return g["f"].mean()


grid = np.round(np.arange(0.50, 0.975, 0.025), 3)
for kd in ks:
    sc = [f05_weighted(th, kd) for th in grid]
    i = int(np.argmax(sc))
    at07 = sc[list(grid).index(0.7)]
    print(f"k={kd:.2f}: best thr={grid[i]:.3f} F0.5={sc[i]:.5f} | at thr 0.70: {at07:.5f} | gain {sc[i]-at07:+.5f}")
    print("   curve:", " ".join(f"{g:.3f}:{s:.4f}" for g, s in zip(grid, sc)))
