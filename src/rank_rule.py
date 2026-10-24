"""Rank-aware acceptance rule on top of the v10 stage-2 model (A/B protocol).

Within each (S1, source), records are ranked by stage-2 probability. The first is accepted if p > t1, the others only if
p > t2 (t2 >= t1): extra copies of an already-matched business are more often decoys. t1, t2 are chosen on half A's
out-of-fold predictions; half B is scored once.
  python src/rank_rule.py   (env TABLE, FINAL, EXCLUDE as for tune_stage2.py)
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("TABLE", "data/stage2v10_train.parquet")
import numpy as np
import polars as pl
import tune_stage2 as T
from metric import f05
from stage2 import CAP

cfg = json.load(open(os.environ.get("FINAL", "data/stage2v10_final.json")))
assert cfg["feats"] == T.FEATS
P = cfg["lgb"]
A, B = T.half == 0, T.half == 1
oof = np.zeros(len(T.y))
for k in (0, 1):
    tr = A & (T.inner != k); te = A & (T.inner == k)
    oof[te] = T.fit_lgb(P, T.X[tr], T.y[tr], T.W[tr]).predict(T.X[te])
T.log("A out-of-fold done")
pb = np.zeros(len(T.y)); pb[B] = T.fit_lgb(P, T.X[A], T.y[A], T.W[A]).predict(T.X[B])
T.log("B predictions done")


def frame(mask, prob):
    idx = np.where(mask)[0]
    d = pl.DataFrame({"s1": T.S1[idx], "e": T.EID[idx], "src": T.SRC[idx], "p": prob[idx]})
    return d.with_columns(pl.col("p").rank("ordinal", descending=True).over(["s1", "src"]).alias("r"))


def score(d, t1, t2, ids):
    d = d.filter(((pl.col("r") == 1) & (pl.col("p") > t1)) | ((pl.col("r") > 1) & (pl.col("p") > t2)))
    d = d.filter(((pl.col("src") == "S2") & (pl.col("r") <= CAP["S2"])) | ((pl.col("src") == "S3") & (pl.col("r") <= CAP["S3"])))
    pred = {}
    for a, b in zip(d["s1"].to_list(), d["e"].to_list()):
        pred.setdefault(a, set()).add(b)
    return f05(pred, T.TRUTH, ids)


dA, dB = frame(A, oof), frame(B, pb)
grid = np.round(np.arange(0.5, 0.96, 0.05), 3)
res = {(t1, t2): score(dA, t1, t2, T.IDS[0]) for t1 in grid for t2 in grid if t2 >= t1}
same = max((v, k) for k, v in res.items() if k[0] == k[1])
best = max((v, k) for k, v in res.items())
T.log(f"A: single threshold {same[1][0]} -> {same[0]:.5f} | rank rule t1={best[1][0]} t2={best[1][1]} -> {best[0]:.5f}")
b_same = score(dB, same[1][0], same[1][0], T.IDS[1]); b_rule = score(dB, *best[1], T.IDS[1])
T.log(f"B(untouched): single threshold {b_same:.5f} | rank rule {b_rule:.5f} | gain {b_rule - b_same:+.5f}")
json.dump({"single": [same[1][0], b_same], "rule": [list(best[1]), b_rule]}, open("data/rank_rule.json", "w"), default=float)
T.log("RANK RULE DONE")
