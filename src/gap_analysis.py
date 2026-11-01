"""Post-challenge gap analysis on the held-out fold: where the macro F0.5 is lost, and whether a decoy-offset rule helps.

  python src/gap_analysis.py [oof.parquet] [threshold]

Inputs (written by the stage-2 runs; not shipped):
  data/v5_oof.parquet        one row per held-out S2/S3 record: its best S1 candidate (s1_id), label, out-of-fold prob
  data/stage2_truth.parquet  every true (s1_id, entity_id) pair of the held-out S1 entities
  data/stage2_s1v.parquet    the held-out S1 entity ids
  dataset/train/*.tsv        raw records, for the decoy-offset check

Reports: the score at the threshold, the oracle score with perfect accept/reject on the existing best candidates
(the ceiling set by blocking and ranking), the share of true pairs that never become a best candidate, and the effect
of removing accepted pairs whose name is the S1 name plus one word at house-number offset +k, k in DECOY_OFFSETS
(the offsets used by the data generator for decoys).
"""
import sys
import polars as pl

DECOY_OFFSETS = [1, 2, 3, 4, 5, 7, 9, 11, 13, 21]
oof_p = sys.argv[1] if len(sys.argv) > 1 else "data/v5_oof.parquet"
thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7

oof = pl.read_parquet(oof_p)
truth = pl.read_parquet("data/stage2_truth.parquet")
s1_ids = pl.read_parquet("data/stage2_s1v.parquet")["entity_id"].to_list()
T = {}
for s, e in truth.iter_rows():
    T.setdefault(s, set()).add(e)


def f05(df):
    P = {}
    for s, e in df.select("s1_id", "entity_id").iter_rows():
        P.setdefault(s, set()).add(e)
    tot = 0.0
    for s in s1_ids:
        p, t = P.get(s, set()), T.get(s, set())
        if not p and not t:
            tot += 1.0
            continue
        tp = len(p & t)
        if tp == 0:
            continue
        pr, rc = tp / len(p), tp / len(t)
        tot += 1.25 * pr * rc / (0.25 * pr + rc)
    return tot / len(s1_ids)


acc = oof.filter(pl.col("prob") > thr)
base = f05(acc)
oracle = oof.filter(pl.col("label") == 1)
unreachable = truth.height - oracle.height
print(f"score @ {thr}: {base:.5f}  (accepted {acc.height:,}, false accepts {acc.filter(pl.col('label') == 0).height:,})")
print(f"oracle, perfect accept/reject on best candidates: {f05(oracle):.5f}")
print(f"true pairs never a best candidate: {unreachable:,} of {truth.height:,} ({unreachable / truth.height:.2%})")

rd = lambda f: pl.read_csv(f, separator="\t", quote_char=None, infer_schema_length=0)
s1 = rd("dataset/train/train_source1.tsv")
oth = pl.concat([rd("dataset/train/train_source2.tsv"), rd("dataset/train/train_source3.tsv")])
num = lambda c: pl.col(c).str.extract(r"(\d+)", 1).cast(pl.Int64, strict=False)
words = lambda c: (pl.col(c).fill_null("").str.to_lowercase().str.replace_all(r"[^\w\s]", " ")
                   .str.replace_all(r"\s+", " ").str.strip_chars().str.split(" "))
a = (acc.join(s1.select(pl.col("entity_id").alias("s1_id"), pl.col("business_name").alias("n1"),
                        pl.col("business_address").alias("a1")), on="s1_id")
     .join(oth.select("entity_id", pl.col("business_name").alias("n2"), pl.col("business_address").alias("a2")),
           on="entity_id")
     .with_columns(words("n1").alias("w1"), words("n2").alias("w2"), (num("a2") - num("a1")).alias("off")))
plus1 = (pl.col("w2").list.len() == pl.col("w1").list.len() + 1) & (pl.col("w1").list.set_difference(pl.col("w2")).list.len() == 0)
rule = (plus1 & pl.col("off").is_in(DECOY_OFFSETS)).fill_null(False)
hit = a.filter(rule)
mirror = a.filter(plus1 & pl.col("off").is_in([-k for k in DECOY_OFFSETS])).height
fa = a.filter(pl.col("label") == 0)
print(f"decoy-offset rule: removes {hit.height:,} (decoys {hit.filter(pl.col('label') == 0).height:,}, "
      f"true {hit.filter(pl.col('label') == 1).height:,}; mirror -k {mirror:,}) -> {f05(a.filter(~rule)):.5f} "
      f"({f05(a.filter(~rule)) - base:+.5f})")
print(f"false accepts by house-number relation: same number {fa.filter(pl.col('off') == 0).height:,}, "
      f"no number {fa.filter(pl.col('off').is_null()).height:,}, decoy offset {fa.filter(pl.col('off').is_in(DECOY_OFFSETS)).height:,}")
