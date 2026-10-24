"""v13: add the LLM judge probability (Qwen2.5-7B-Instruct-AWQ, zero-shot, see cloud_package3/xgboost_llm.py) to a
stage-2 table. Only hard pairs (cross-encoder score in [0.1, 0.9], same rule on validation and test) were judged; the
rest keep llm = NaN. The LLM never saw any label, so validation and test scores are produced identically.
  python src/add_llm.py <table_in> <llm.parquet> <table_out>
"""
import sys
import polars as pl
tin, llm, tout = sys.argv[1:4]
t = pl.read_parquet(tin).join(pl.read_parquet(llm).select("qid", pl.col("llm").cast(pl.Float32)), on="qid", how="left")
t = t.with_columns((pl.col("llm") - pl.col("ce")).alias("llm_minus_ce"))
t.write_parquet(tout)
print(tout, t.shape, "judged rows", int(t["llm"].is_not_null().sum()), "mean llm", t["llm"].mean())
