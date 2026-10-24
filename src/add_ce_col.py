"""Add one more transformer score column (e.g. ce4 = full-epoch bge-reranker on both GPUs) to a stage-2 table, plus the
average of every transformer score present (ce_ens). Rows outside the uncertain band keep NaN.
  python src/add_ce_col.py <table_in> <scores.parquet> <column> <table_out>
"""
import sys
import polars as pl
tin, sc, col, tout = sys.argv[1:5]
t = pl.read_parquet(tin).join(pl.read_parquet(sc).select("qid", pl.col(col).cast(pl.Float32)), on="qid", how="left")
cols = [c for c in ("ce", "ce2", "ce3", "ce4") if c in t.columns]
t = t.drop("ce_ens", strict=False).with_columns(pl.mean_horizontal(cols).alias("ce_ens"), (pl.col(col) - pl.col("p1")).alias(f"{col}_minus_p1"))
t.write_parquet(tout)
print(tout, t.shape, "scored rows", int(t[col].is_not_null().sum()), f"mean {col}", t[col].mean(), "ensemble of", cols)
