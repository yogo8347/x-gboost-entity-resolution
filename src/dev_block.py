"""Dev: measure blocking recall on a sample of train queries against full S1 index."""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import polars as pl
from blocking import build_index, generate

cap, rare, topk, pre = (int(x) for x in (sys.argv[1:5] if len(sys.argv) > 4 else (500, 8, 6, 30)))
t = time.time()
s1 = pl.read_parquet("data/train_s1.parquet").with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
q = pl.concat([pl.scan_parquet(f"data/train_s{k}.parquet")
               .filter(pl.col("entity_id").hash(seed=3) % 100 < 3) for k in (2, 3)]).collect()
gt = (pl.read_csv("dataset/train/train_ground_truth.tsv", separator="\t", infer_schema=False)
      .with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
      .drop_nulls().rename({"matched_entity_ids": "entity_id", "source1_entity_id": "s1_id"}))
q = q.join(gt, on="entity_id", how="left").with_row_index("qid").with_columns(pl.col("qid").cast(pl.UInt32))
print("queries", q.height, "load", round(time.time() - t), flush=True)
keys, dfreq = build_index(s1)
print("index keys", keys.height, "vocab", dfreq.height, round(time.time() - t), flush=True)
print(dfreq["df"].describe(), flush=True)
c = generate(keys, dfreq, q, s1.height, cap=cap, rare=rare, topk=topk, pre=pre, log=True)
print("pairs", c.height, "per query", c.height / q.height, round(time.time() - t), flush=True)
c = c.join(s1.select("rid", pl.col("entity_id").alias("cand_s1")), on="rid").join(
    q.select("qid", "s1_id"), on="qid")
pos = q.filter(pl.col("s1_id").is_not_null())
hit = c.filter(pl.col("cand_s1") == pl.col("s1_id"))
print("recall@K", hit["qid"].n_unique() / pos.height)
for k in (1, 3, 5, 6, 8, 10, 12, 16, 20):
    print(f"recall@{k}", hit.filter(pl.col("brank") < k)["qid"].n_unique() / pos.height)
miss = pos.join(hit.select("qid"), on="qid", how="anti").sample(15, seed=1)
s1i = s1.rename({"entity_id": "s1_id"})
for r in miss.join(s1i, on="s1_id", suffix="_1").iter_rows(named=True):
    print(f"  Q[{r['core']} | {r['addr']}]  S1[{r['core_1']} | {r['addr_1']}]")
