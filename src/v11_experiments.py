"""v11 low-risk experiments on top of the v10 stage-2 table, same A/B protocol as tune_stage2.py.

Every choice (threshold, weights) is made on half A (out-of-fold); half B is scored once per experiment.
  E0  v10 baseline LGBM (reference, reproduces the tuner's number)
  E1  seed bagging: the same LGBM with 5 different bagging/feature seeds, probabilities averaged
  E2  per-country thresholds (US / India) chosen on A's out-of-fold predictions
  E3  E1 + E2
  python src/v11_experiments.py  (env TABLE, FINAL for the v10 config; writes data/v11_experiments.json)
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("TABLE", "data/stage2v10_train.parquet")
import numpy as np
import polars as pl
import tune_stage2 as T  # loads the table, X, y, halves, TRUTH, score()

cfg = json.load(open(os.environ.get("FINAL", "data/stage2v10_final.json")))
assert cfg["feats"] == T.FEATS, "feature list differs from the v10 config"
P0 = T.BASE_LGB if cfg["kind"] == "lgb" and cfg["lgb"] == T.BASE_LGB else cfg.get("lgb", T.BASE_LGB)
SEEDS = [1, 2, 3, 4, 5]
A, B = T.half == 0, T.half == 1
ctry = pl.read_parquet("data/train_s1.parquet", columns=["entity_id", "country"])
C = dict(zip(ctry["entity_id"].to_list(), ctry["country"].to_list()))
IDS_C = {h: {c: [i for i in T.IDS[h] if C.get(i) == c] for c in ("US", "India")} for h in (0, 1)}
ROW_C = np.array([C.get(s) for s in T.S1])


def with_seed(p, s):
    return dict(p, seed=s, bagging_seed=s, feature_fraction_seed=s)


def oof_A(p):
    pr = np.zeros(len(T.y))
    for k in (0, 1):
        tr = A & (T.inner != k); te = A & (T.inner == k)
        pr[te] = T.fit_lgb(p, T.X[tr], T.y[tr], T.W[tr]).predict(T.X[te])
    return pr


def fitA_predB(p):
    pb = np.zeros(len(T.y)); pb[B] = T.fit_lgb(p, T.X[A], T.y[A], T.W[A]).predict(T.X[B]); return pb


def score_country(half_mask, prob, thr_by_c, ids_by_c):
    """per-country thresholds: macro F0.5 over the union of countries (weighted by S1 counts)."""
    tot, n = 0.0, 0
    for c, th in thr_by_c.items():
        m = half_mask & (ROW_C == c)
        s = T.score(m, prob, th, ids_by_c[c]); k = len(ids_by_c[c])
        tot += s * k; n += k
    return tot / n


res = {}
oof0 = oof_A(P0); pb0 = fitA_predB(P0)
thr0, a0 = T.best_thr(A, oof0, T.IDS[0])
res["E0 v10 baseline"] = dict(A=a0, B=T.score(B, pb0, thr0, T.IDS[1]), thr=thr0); T.log(res)

oofs, pbs = [oof0], [pb0]
for s in SEEDS[1:]:
    oofs.append(oof_A(with_seed(P0, s))); pbs.append(fitA_predB(with_seed(P0, s))); T.log("seed", s, "done")
oofb, pbb = np.mean(oofs, 0), np.mean(pbs, 0)
thr1, a1 = T.best_thr(A, oofb, T.IDS[0])
res["E1 seed-bag x5"] = dict(A=a1, B=T.score(B, pbb, thr1, T.IDS[1]), thr=thr1); T.log(res["E1 seed-bag x5"])

for name, oo, pb in (("E2 per-country thr", oof0, pb0), ("E3 seed-bag + per-country thr", oofb, pbb)):
    thr_c = {c: T.best_thr(A & (ROW_C == c), oo, IDS_C[0][c])[0] for c in ("US", "India")}
    res[name] = dict(A=score_country(A, oo, thr_c, IDS_C[0]), B=score_country(B, pb, thr_c, IDS_C[1]), thr=thr_c)
    T.log(name, res[name])

T.log("=" * 60)
for k, v in res.items():
    T.log(f"{k:32s} B(untouched) {v['B']:.5f}  A(CV) {v['A']:.5f}  thr {v['thr']}")
json.dump(res, open("data/v11_experiments.json", "w"), indent=1, default=float)
T.log("V11 EXPERIMENTS DONE")
