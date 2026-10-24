"""Word markers (v6): extra words a record carries that its S1 lacks.

Derived on HALF A of the held-out S1 fold only (src: analysis in STATUS.md, lists saved in data/word_markers_halfA.json);
the effect is reported on the untouched half B.
  * decoys are built by appending a business word to a real name: these ten words appear as an EXTRA word
    in 1,070 decoy pairs and only 4 true pairs on half A
  * alias markers (dba / formerly / aka / fka / nee / known as / doing business) appear only on true copies
Adds columns to an existing stage-2 table (no re-blocking needed).

  python src/word_markers.py <table_in> <table_out> <prep_dir> <split>
"""
import sys
import numpy as np
import polars as pl

DECOY_EXTRA = {"pvt", "holdings", "group", "ventures", "public", "exports", "industries", "enterprises", "overseas", "infratech"}
ALIAS = {"dba", "formerly", "aka", "fka", "nee", "known", "doing", "as"}


def word_markers(q_name, s_name):
    n = len(q_name)
    dec = np.zeros(n, np.float32); ali = np.zeros(n, np.float32); extra = np.zeros(n, np.float32)
    for i in range(n):
        a = set((q_name[i] or "").split()); b = set((s_name[i] or "").split())
        ex = a - b
        dec[i] = len(ex & DECOY_EXTRA); ali[i] = float(bool(ex & ALIAS)); extra[i] = len(ex)
    return {"wm_decoy_extra": dec, "wm_alias": ali, "wm_n_extra": extra}


if __name__ == "__main__":
    tin, tout, prep, split = sys.argv[1:5]
    t = pl.read_parquet(tin)
    q = (pl.concat([pl.read_parquet(f"{prep}/{split}_s{k}.parquet", columns=["name"]) for k in (2, 3)])
         .with_row_index("qid").with_columns(pl.col("qid").cast(pl.UInt32)))
    s1 = pl.read_parquet(f"{prep}/{split}_s1.parquet", columns=["name"]).with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
    p = t.select("qid", "rid").join(q, on="qid", how="left").join(s1.rename({"name": "s_name"}), on="rid", how="left")
    m = word_markers(p["name"].fill_null("").to_list(), p["s_name"].fill_null("").to_list())
    t = t.with_columns([pl.Series(k, v) for k, v in m.items()])
    t.write_parquet(tout)
    print(tout, t.shape, {k: round(float(v.mean()), 4) for k, v in m.items()})
