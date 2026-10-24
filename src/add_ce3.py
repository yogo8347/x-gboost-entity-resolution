"""v14: add the second transformer (xlm-roberta-base, trained on stage-1 training-fold pairs like ce2) to the best
stage-2 table so far. Same uncertain band; rows outside it keep ce3 = NaN.
  python src/add_ce3.py <table_in> <ce3.parquet> <table_out>
"""
import sys
import polars as pl
tin, ce, tout = sys.argv[1:4]
t = pl.read_parquet(tin).join(pl.read_parquet(ce).select("qid", pl.col("ce3").cast(pl.Float32)), on="qid", how="left")
cols = [c for c in ("ce", "ce2", "ce3") if c in t.columns]
t = t.with_columns(pl.mean_horizontal(cols).alias("ce_ens"), (pl.col("ce3") - pl.col("p1")).alias("ce3_minus_p1"))
t.write_parquet(tout)
print(tout, t.shape, "scored rows", int(t["ce3"].is_not_null().sum()), "mean ce3", t["ce3"].mean(), "ensemble of", cols)
