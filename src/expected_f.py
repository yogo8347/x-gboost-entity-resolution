"""Per-S1 expected-F0.5 decision layer.

The metric is F0.5 averaged over S1 entities, so the best set of matches for one S1 depends on all of its candidate
records together, not on each record's probability alone. For each S1 we sort its assigned records by probability and
pick the prefix (possibly empty) with the highest expected F0.5:
    TP_k = sum of the top-k probabilities, FP_k = k - TP_k,
    N    = expected number of true matches = (sum of all its probabilities) * (1 + alpha)   [alpha: matches we never see]
    E[F0.5](k) ~ 1.25 TP_k / (1.25 TP_k + 0.25 (N - TP_k) + FP_k),     E[score](empty) = prod(1 - p_i)
alpha is chosen on half A of the held-out S1 entities and reported once on the untouched half B.

  python src/expected_f.py            (validation study on data/v5_oof.parquet)
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import polars as pl

from metric import f05
from stage2 import CAP


def choose(df, alpha, thr_floor=0.0):
    """df: s1_id, entity_id, src, prob. Returns the accepted rows."""
    d = df.filter(pl.col("prob") > thr_floor).sort(["s1_id", "prob"], descending=[False, True])
    d = d.with_columns(pl.col("prob").rank("ordinal", descending=True).over(["s1_id", "src"]).alias("r"))
    d = d.filter(((pl.col("src") == "S2") & (pl.col("r") <= CAP["S2"])) | ((pl.col("src") == "S3") & (pl.col("r") <= CAP["S3"])))
    d = d.with_columns(pl.int_range(1, pl.len() + 1).over("s1_id").alias("k"),
                       pl.col("prob").cum_sum().over("s1_id").alias("tp"),
                       (pl.col("prob").sum().over("s1_id") * (1 + alpha)).alias("n"),
                       (1 - pl.col("prob")).log().sum().over("s1_id").exp().alias("p_empty"))
    d = d.with_columns((1.25 * pl.col("tp") / (1.25 * pl.col("tp") + 0.25 * (pl.col("n") - pl.col("tp")) + (pl.col("k") - pl.col("tp")))).alias("ef"))
    best = d.group_by("s1_id").agg(pl.col("ef").max().alias("ef_max"), pl.col("k").filter(pl.col("ef") == pl.col("ef").max()).first().alias("k_best"),
                                   pl.col("p_empty").first())
    keep = best.filter(pl.col("ef_max") > pl.col("p_empty")).select("s1_id", "k_best")
    return d.join(keep, on="s1_id").filter(pl.col("k") <= pl.col("k_best"))


def score(sel, ids, truth):
    pred = {}
    for a, b in sel.select("s1_id", "entity_id").iter_rows():
        pred.setdefault(a, set()).add(b)
    return f05(pred, truth, ids)


if __name__ == "__main__":
    t = pl.read_parquet("data/v5_oof.parquet").with_columns(pl.col("entity_id").str.slice(0, 2).alias("src"))
    tr = pl.read_parquet("data/stage2_truth.parquet")
    ids_all = pl.read_parquet("data/stage2_s1v.parquet")["entity_id"]
    half = (ids_all.hash(seed=77) % 2).to_numpy()
    IDS = {h: ids_all.filter(half == h).to_list() for h in (0, 1)}
    truth = {}
    for a, b in tr.select("s1_id", "entity_id").iter_rows():
        truth.setdefault(a, set()).add(b)
    thr = 0.675
    base = t.filter(pl.col("prob") > thr)
    base = base.with_columns(pl.col("prob").rank("ordinal", descending=True).over(["s1_id", "src"]).alias("r"))
    base = base.filter(((pl.col("src") == "S2") & (pl.col("r") <= CAP["S2"])) | ((pl.col("src") == "S3") & (pl.col("r") <= CAP["S3"])))
    tA = t.filter(pl.col("s1_id").is_in(IDS[0])); tB = t.filter(pl.col("s1_id").is_in(IDS[1]))
    print(f"fixed threshold {thr}:  A {score(base, IDS[0], truth):.5f}   B {score(base, IDS[1], truth):.5f}")
    best = (-1, None)
    for alpha in (0.0, 0.02, 0.04, 0.06, 0.1, 0.15):
        for floor in (0.0, 0.05, 0.1):
            sA = score(choose(tA, alpha, floor), IDS[0], truth)
            print(f"  alpha={alpha:.2f} floor={floor:.2f}: A {sA:.5f}")
            if sA > best[0]:
                best = (sA, (alpha, floor))
    a, f = best[1]
    print(f"chosen on A: alpha={a} floor={f}  ->  B (untouched) {score(choose(tB, a, f), IDS[1], truth):.5f}")
