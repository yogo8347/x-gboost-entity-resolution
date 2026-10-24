"""v10: add the Kaggle cross-encoder probability (bge-reranker-v2-m3, 2-fold, out-of-fold on validation) to a stage-2
table. Rows outside the uncertain band (stage-1 p1 not in [0.02, 0.995]) were not scored and keep ce = NaN.
  python src/add_ce.py <table_in> <ce.parquet> <table_out>
"""
import sys
import polars as pl
tin, ce, tout = sys.argv[1:4]
t = pl.read_parquet(tin).join(pl.read_parquet(ce).select("qid", pl.col("ce").cast(pl.Float32)), on="qid", how="left")
t = t.with_columns((pl.col("ce") - pl.col("p1")).alias("ce_minus_p1"), pl.col("ce").is_not_null().cast(pl.Float32).alias("ce_scored"))
t.write_parquet(tout)
print(tout, t.shape, "scored rows", int(t["ce_scored"].sum()), "mean ce", t["ce"].mean())
