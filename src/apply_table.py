"""Apply a tuned stage-2 config to any test table and write both TSVs (optionally at several thresholds).

  python src/apply_table.py <final.json> <model_prefix> <test_table> <out_dir> [thr ...]
With extra thresholds, writes <out_dir>_t<NN>/ for each (e.g. output_v7_t85/).
"""
import json, os, shutil, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import polars as pl
import lightgbm as lgb
import xgboost as xgb

from stage2 import decide
from pipeline import write_lists, log

cfg_p, prefix, table, out = sys.argv[1:5]
extra = [float(x) for x in sys.argv[5:]]
cfg = json.load(open(cfg_p))
t = pl.read_parquet(table)
prob = np.zeros(t.height)
lgbm = lgb.Booster(model_file=prefix + "_lgb.txt") if cfg["kind"] in ("lgb", "ensemble") else None
if cfg["kind"] in ("xgb", "ensemble"):
    mx = xgb.Booster(); mx.load_model(prefix + "_xgb.json")
STEP = 1_000_000  # predict in chunks: a 10M x 117 float matrix does not fit next to everything else
for st in range(0, t.height, STEP):
    X = t.slice(st, STEP).select(cfg["feats"]).to_numpy().astype(np.float32)
    if lgbm is not None:
        prob[st:st + STEP] += cfg.get("w_lgb", 1.0) * lgbm.predict(X)
    if cfg["kind"] in ("xgb", "ensemble"):
        prob[st:st + STEP] += (1.0 - cfg.get("w_lgb", 0.0)) * mx.predict(xgb.DMatrix(X))
    del X
t = t.select("qid", "rid", "src")
s1 = pl.read_parquet(os.environ.get("ER_PREP", "data/frpatch") + "/test_s1.parquet", columns=["entity_id"]).with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
q = (pl.concat([pl.read_parquet(os.environ.get("ER_PREP", "data/frpatch") + f"/test_s{k}.parquet", columns=["entity_id"]) for k in (2, 3)])
     .with_row_index("qid").with_columns(pl.col("qid").cast(pl.UInt32)))
idq = q.select("qid", pl.col("entity_id").alias("e")); ids1 = s1.select("rid", pl.col("entity_id").alias("s1"))
cand_src = os.environ.get("CAND_TSV", "output_v5/candidate_pairs.tsv")   # candidate list matching the scored table
for thr, d in [(cfg["thr"], out)] + [(x, f"{out}_t{int(round(x * 100))}") for x in extra]:
    os.makedirs(d, exist_ok=True)
    a = decide(t, prob, thr).join(idq, on="qid").join(ids1, on="rid")
    res = write_lists(s1, a, "matched_entity_ids", f"{d}/matching_results.tsv")
    shutil.copy(cand_src, f"{d}/candidate_pairs.tsv")
    log(f"{d}: thr {thr:.3f} matched {a.height} empty {(res['matched_entity_ids'] == '').sum()}")
