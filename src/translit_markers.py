"""Translated-comparison markers (v8): record words are translated with the learned dictionary
(data/translit_dict.json, built by translit_dict.py from TRAINING-fold true pairs only), cleaned like clean_markers,
and compared with the S1 again.

  python src/translit_markers.py <table_in> <table_out> <prep_dir> <split>
"""
import json, sys
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

sys.path.insert(0, "src")
from clean_markers import clean_name, clean_addr

D = json.load(open("data/translit_dict.json"))
DN, DA = D["core"], D["addr"]


def tr(s, d):
    return " ".join(d.get(t, t) for t in (s or "").split())


def _cp(scorer, a, b):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)


if __name__ == "__main__":
    tin, tout, prep, split = sys.argv[1:5]
    t = pl.read_parquet(tin)
    q = (pl.concat([pl.read_parquet(f"{prep}/{split}_s{k}.parquet", columns=["core", "addr"]) for k in (2, 3)])
         .with_row_index("qid").with_columns(pl.col("qid").cast(pl.UInt32)))
    s1 = pl.read_parquet(f"{prep}/{split}_s1.parquet", columns=["core", "addr"]).with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
    p = t.select("qid", "rid").join(q, on="qid", how="left").join(s1.rename({"core": "s_core", "addr": "s_addr"}), on="rid", how="left")
    g = lambda c: p[c].fill_null("").to_list()
    qn = [clean_name(tr(x, DN)) for x in g("core")]; sn = [clean_name(tr(x, DN)) for x in g("s_core")]
    qa = [clean_addr(tr(x, DA)) for x in g("addr")]; sa = [clean_addr(tr(x, DA)) for x in g("s_addr")]
    f = {"tr_n_ratio": _cp(fuzz.ratio, qn, sn), "tr_n_tset": _cp(fuzz.token_set_ratio, qn, sn),
         "tr_n_eq": np.array([a == b and a != "" for a, b in zip(qn, sn)], np.float32),
         "tr_a_tset": _cp(fuzz.token_set_ratio, qa, sa), "tr_a_ratio": _cp(fuzz.token_sort_ratio, qa, sa),
         "tr_a_eq": np.array([set(a.split()) == set(b.split()) and a != "" for a, b in zip(qa, sa)], np.float32)}
    t = t.with_columns([pl.Series(k, v) for k, v in f.items()])
    t.write_parquet(tout)
    print(tout, t.shape, {k: round(float(v.mean()), 3) for k, v in f.items()})
