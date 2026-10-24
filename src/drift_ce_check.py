"""Second-opinion drift check for the cross-encoder feature `ce` (v10).

drift_check.py uses quantile bins taken from validation. `ce` is bimodal (most values within 0.001 of 0 or 1), so its
quantile edges collapse into tiny bins at the ends and PSI explodes (1.35) on tiny shifts. Here: fixed 0.1-wide bins,
US+India only (France is served by v8 in the hybrid submission), and the same after matching the stage-1 p1 mix.
  python src/drift_ce_check.py
"""
import numpy as np
import polars as pl

E = np.linspace(0, 1, 11)


def psi(pa, pb):
    pa, pb = pa + 1e-6, pb + 1e-6
    return float(np.sum((pa - pb) * np.log(pa / pb)))


v = pl.read_parquet("data/stage2v10_train.parquet", columns=["s1_id", "p1", "ce"]).drop_nulls("ce")
t = pl.read_parquet("data/frpatch/stage2v10_test.parquet", columns=["s1_id", "p1", "ce"]).drop_nulls("ce")
t = t.join(pl.read_parquet("data/test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"}), on="s1_id")
tn = t.filter(pl.col("country") != "France")
h = lambda x, w=None: np.histogram(x, E, weights=w)[0] / (len(x) if w is None else w.sum())
print("fixed-bin PSI ce, all test      ", round(psi(h(v["ce"].to_numpy()), h(t["ce"].to_numpy())), 3))
print("fixed-bin PSI ce, US+India      ", round(psi(h(v["ce"].to_numpy()), h(tn["ce"].to_numpy())), 3))
vb = np.clip((v["p1"].to_numpy() * 10).astype(int), 0, 9); tb = np.clip((tn["p1"].to_numpy() * 10).astype(int), 0, 9)
w = (np.bincount(tb, minlength=10) / len(tb)) / (np.bincount(vb, minlength=10) / len(vb))
print("PSI ce | p1 mix matched, US+India", round(psi(h(v["ce"].to_numpy(), w[vb]), h(tn["ce"].to_numpy())), 3))
