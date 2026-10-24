"""Leaderboard direction probe: the v2 stage-2 model applied to test at several thresholds.
Writes output_thr<THR>/matching_results.tsv (+ candidate_pairs.tsv copy). v2's output/ is untouched.

  python src/probe_thresholds.py 0.55 0.85
"""
import os, shutil, sys
sys.path.insert(0, os.path.dirname(__file__))
import lightgbm as lgb
import polars as pl

from pipeline import load, top2, write_lists, log, DATA
from stage2 import cluster_feats, decide

thrs = [float(x) for x in sys.argv[1:]]
s1, q = load("test")
t = cluster_feats(top2(pl.read_parquet(f"{DATA}/test_scored.parquet")), q, s1)
m = lgb.Booster(model_file=f"{DATA}/model2.txt")
prob = m.predict(t.select(m.feature_name()).to_numpy())
idq = q.select("qid", pl.col("entity_id").alias("e")); ids1 = s1.select("rid", pl.col("entity_id").alias("s1"))
for thr in thrs:
    out = f"output_thr{thr:.2f}"
    os.makedirs(out, exist_ok=True)
    a = decide(t, prob, thr).join(idq, on="qid").join(ids1, on="rid")
    res = write_lists(s1, a, "matched_entity_ids", f"{out}/matching_results.tsv")
    shutil.copy("output/candidate_pairs.tsv", f"{out}/candidate_pairs.tsv")
    log(f"thr={thr}: matched {a.height}, empty rows {(res['matched_entity_ids'] == '').sum()} -> {out}/")
