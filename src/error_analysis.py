"""Break validation errors down by cause, country and source; print examples."""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import polars as pl
from pipeline import load, load_gt, eval_fold, top2, DATA

b = json.load(open(f"{DATA}/best.json"))
s1, q = load("train")
gt = load_gt()
s1v = s1.filter(eval_fold(pl.col("entity_id")))
vr = s1v.select("rid", pl.col("entity_id").alias("s1_id"), pl.col("country").alias("s1c"))
qq = q.select("qid", "entity_id", "country").join(gt, on="entity_id", how="left")
S = pl.read_parquet(f"{DATA}/train_scored.parquet")
top = top2(S)
cands = pl.read_parquet(f"{DATA}/train_cands.parquet").select("qid", "rid")

# true pairs whose S1 is in the validation fold
tp = qq.join(vr, on="s1_id").select("qid", "rid", "entity_id", "country")
tp = tp.join(cands.with_columns(pl.lit(True).alias("in_cand")), on=["qid", "rid"], how="left")
tp = tp.join(top.rename({"rid": "top_rid"}), on="qid", how="left")
tp = tp.with_columns(
    pl.when(pl.col("in_cand").is_null()).then(pl.lit("blocking_miss"))
    .when(pl.col("top_rid") != pl.col("rid")).then(pl.lit("ranked_wrong"))
    .when((pl.col("p1") <= b["thr"]) | (pl.col("p1") - pl.col("p2") < b["margin"])).then(pl.lit("below_thr"))
    .otherwise(pl.lit("found")).alias("outcome"),
    pl.col("entity_id").str.slice(0, 2).alias("src"))
print("TRUE PAIRS of val S1:", tp.height)
print(tp.group_by("outcome").len().with_columns((pl.col("len") / tp.height).alias("frac")).sort("len", descending=True))
print(tp.group_by("country", "src").agg((pl.col("outcome") == "found").mean().alias("recall"), pl.len()).sort("country", "src"))

# false merges onto val S1s
acc = top.filter((pl.col("p1") > b["thr"]) & (pl.col("p1") - pl.col("p2") >= b["margin"])).join(vr, on="rid")
acc = acc.join(qq.select("qid", "entity_id", "country", pl.col("s1_id").alias("true_s1")), on="qid")
fp = acc.filter(pl.col("true_s1").is_null() | (pl.col("true_s1") != pl.col("s1_id")))
print("ACCEPTED onto val S1:", acc.height, "false merges:", fp.height, f"precision={1 - fp.height / acc.height:.4f}")
print(fp.with_columns(pl.col("true_s1").is_null().alias("query_is_distractor")).group_by("country", "query_is_distractor").len())
sing = gt.select("s1_id").unique()
nsing = s1v.join(sing, left_on="entity_id", right_on="s1_id", how="anti").height
hit_sing = fp.join(sing, on="s1_id", how="anti")["s1_id"].n_unique()
print(f"val singletons: {nsing}, wrongly given a match: {hit_sing}")

# examples
raw = pl.concat([pl.read_parquet(f"{DATA}/train_s{k}.parquet").select("entity_id", "name", "addr") for k in (1, 2, 3)])
def show(df, title, n=12):
    print(f"\n=== {title}")
    for r in df.head(n).iter_rows(named=True):
        a = raw.filter(pl.col("entity_id") == r["q"]).row(0)
        s = raw.filter(pl.col("entity_id") == r["s"]).row(0)
        print(f"  p={r.get('p1', 0):.3f}  Q[{a[1]} | {a[2]}]\n            S1[{s[1]} | {s[2]}]")
show(fp.sample(min(12, fp.height), seed=1).select(pl.col("entity_id").alias("q"), pl.col("s1_id").alias("s"), "p1"), "FALSE MERGES")
m = tp.filter(pl.col("outcome") == "below_thr").sample(12, seed=1).join(vr, on="rid")
show(m.select(pl.col("entity_id").alias("q"), pl.col("s1_id").alias("s"), "p1"), "MISSED: below threshold")
m = tp.filter(pl.col("outcome") == "ranked_wrong").sample(12, seed=1).join(vr, on="rid")
show(m.select(pl.col("entity_id").alias("q"), pl.col("s1_id").alias("s"), "p1"), "MISSED: other S1 ranked higher (shows TRUE S1)")
