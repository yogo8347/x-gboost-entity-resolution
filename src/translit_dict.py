"""Learn a word-translation dictionary (garbled / transliterated / abbreviated word -> canonical word) from data.

Source: TRUE matched pairs whose S1 is in the TRAINING fold only (never the held-out fold), so the dictionary can be
used for held-out evaluation without leakage. For each pair, the words that differ between record and S1 are aligned
when exactly one word differs on each side (e.g. "dijirrl" <-> "digital"). A mapping is kept when it is frequent
(>= MIN_N occurrences) and consistent (the same target in >= MIN_SHARE of that word's alignments).
Built separately for names (core) and addresses. Writes data/translit_dict.json and a coverage report.

  python src/translit_dict.py
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import polars as pl
from pipeline import load_gt, val_fold

MIN_N, MIN_SHARE = 3, 0.8

gt = load_gt()
s1 = pl.read_parquet("data/train_s1.parquet", columns=["entity_id", "core", "addr"])
s1 = s1.filter(~val_fold(pl.col("entity_id")))                        # TRAINING fold only
q = pl.concat([pl.read_parquet(f"data/train_s{k}.parquet", columns=["entity_id", "core", "addr"]) for k in (2, 3)])
pairs = q.join(gt, on="entity_id").join(s1.rename({"entity_id": "s1_id", "core": "s_core", "addr": "s_addr"}), on="s1_id")
print("true training-fold pairs:", pairs.height)
out = {}
for col in ("core", "addr"):
    d = pairs.select(pl.col(col).str.split(" ").alias("a"), pl.col("s_" + col).str.split(" ").alias("b"))
    d = d.with_columns(pl.col("a").list.set_difference("b").alias("da"), pl.col("b").list.set_difference("a").alias("db"))
    d = d.filter((pl.col("da").list.len() == 1) & (pl.col("db").list.len() == 1))
    d = d.select(pl.col("da").list.first().alias("src"), pl.col("db").list.first().alias("dst"))
    d = d.filter((pl.col("src").str.len_chars() >= 2) & (pl.col("dst").str.len_chars() >= 2) & (pl.col("src") != pl.col("dst")))
    c = d.group_by("src", "dst").len()
    tot = c.group_by("src").agg(pl.col("len").sum().alias("tot"))
    best = c.sort("len", descending=True).group_by("src", maintain_order=True).first().join(tot, on="src")
    keep = best.filter((pl.col("len") >= MIN_N) & (pl.col("len") / pl.col("tot") >= MIN_SHARE))
    out[col] = dict(zip(keep["src"].to_list(), keep["dst"].to_list()))
    print(f"{col}: aligned 1-1 differences {d.height}, candidate words {tot.height}, kept mappings {keep.height}")
    print("  examples:", keep.sort("len", descending=True).head(25).select("src", "dst", "len").rows())
json.dump(out, open("data/translit_dict.json", "w"))

# coverage: share of S2/S3 name-word occurrences never seen in any S1 name that the dictionary resolves
s1_vocab = set(pl.read_parquet("data/train_s1.parquet", columns=["core"])["core"].str.split(" ").explode().unique().to_list())
toks = q.select(pl.col("core").str.split(" ")).explode("core").filter(pl.col("core").str.len_chars() >= 2)["core"]
unknown = toks.filter(~toks.is_in(list(s1_vocab)))
res = unknown.is_in(list(out["core"].keys())).mean()
print(f"name words never seen in S1: {unknown.len() / toks.len():.2%} of occurrences; dictionary resolves {res:.1%} of them")
