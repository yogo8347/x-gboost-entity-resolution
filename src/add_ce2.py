"""v12: add the cross-encoder v2 probability (one model trained on stage-1 training-fold pairs, see ce2_prep.py) to a
v10 stage-2 table. Same uncertain band as v10; rows outside it keep ce2 = NaN.
  python src/add_ce2.py <table_in> <ce2.parquet> <table_out>
"""
import sys
import polars as pl
tin, ce, tout = sys.argv[1:4]
t = pl.read_parquet(tin).join(pl.read_parquet(ce).select("qid", pl.col("ce2").cast(pl.Float32)), on="qid", how="left")
t = t.with_columns((pl.col("ce2") - pl.col("p1")).alias("ce2_minus_p1"), ((pl.col("ce") + pl.col("ce2")) / 2).alias("ce_avg"))
t.write_parquet(tout)
print(tout, t.shape, "scored rows", int(t["ce2"].is_not_null().sum()), "mean ce2", t["ce2"].mean())
