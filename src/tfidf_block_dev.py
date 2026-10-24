"""Recall experiment: character-3-gram TF-IDF top-k retrieval (Sparkly, VLDB 2023) as a SECOND candidate source,
merged with the current key-based candidates. Same 3% query sample as dev_block.py (hash seed 3, < 3%).

Text per record: core name + the main address words (translated with the learned dictionary), char 3-grams within
word boundaries, sublinear tf, per-country index. Top-K by cosine via sparse_dot_topn (multi-threaded).

  python src/tfidf_block_dev.py [K]
"""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

K = int(sys.argv[1]) if len(sys.argv) > 1 else 10
D = json.load(open("data/translit_dict.json"))


def text(df):
    tr = lambda col, d: pl.col(col).str.split(" ").list.eval(pl.element().replace(d)).list.join(" ")
    return df.select(tr("core", D["core"]).alias("c"), tr("awords", D["addr"]).alias("a")).select(
        (pl.col("c") + " " + pl.col("c") + " " + pl.col("a")).alias("t"))["t"].to_list()   # name counted twice


t0 = time.time()
s1 = pl.read_parquet("data/train_s1.parquet", columns=["entity_id", "country", "core", "awords"]).with_row_index("rid")
q = pl.concat([pl.scan_parquet(f"data/train_s{k}.parquet").select("entity_id", "country", "core", "awords")
               .filter(pl.col("entity_id").hash(seed=3) % 100 < 3) for k in (2, 3)]).collect().with_row_index("qid")
gt = (pl.read_csv("dataset/train/train_ground_truth.tsv", separator="\t", infer_schema=False)
      .with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").drop_nulls()
      .rename({"matched_entity_ids": "entity_id", "source1_entity_id": "s1_id"}))
q = q.join(gt, on="entity_id", how="left")
out = []
for ctry in s1["country"].unique().to_list():
    a = s1.filter(pl.col("country") == ctry); b = q.filter(pl.col("country") == ctry)
    if b.height == 0:
        continue
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), sublinear_tf=True, min_df=2, max_df=0.05, dtype=np.float32)
    A = vec.fit_transform(text(a)); B = vec.transform(text(b))
    M = sp_matmul_topn(B, A.T.tocsr(), top_n=K, threshold=0.05, n_threads=8).tocoo()
    out.append(pl.DataFrame({"qid": b["qid"].to_numpy()[M.row], "rid": a["rid"].to_numpy()[M.col], "cos": M.data}))
    print(f"{ctry}: S1 {a.height} queries {b.height} vocab {len(vec.vocabulary_)} pairs {M.nnz} ({time.time() - t0:.0f}s)", flush=True)
tf = pl.concat(out).with_columns(pl.col("qid").cast(pl.UInt32), pl.col("rid").cast(pl.UInt32))
tf = tf.join(s1.select(pl.col("rid").cast(pl.UInt32), pl.col("entity_id").alias("cand")), on="rid")
tf = tf.join(q.select(pl.col("qid").cast(pl.UInt32), "s1_id"), on="qid")
pos = q.filter(pl.col("s1_id").is_not_null()).height
hit_tf = tf.filter(pl.col("cand") == pl.col("s1_id")).select("qid").unique()
print(f"TF-IDF char-3gram top-{K}: recall {hit_tf.height / pos:.4f}  ({tf.height / q.height:.2f} candidates per record)")
# union with the EXISTING key-based candidates (data/train_cands.parquet), mapped by entity id (no index rebuild)
full_q = (pl.concat([pl.read_parquet(f"data/train_s{k}.parquet", columns=["entity_id"]) for k in (2, 3)])
          .with_row_index("fqid").with_columns(pl.col("fqid").cast(pl.UInt32)))
qmap = q.select(pl.col("qid").cast(pl.UInt32), "entity_id", "s1_id").join(full_q, on="entity_id")
kc = (pl.read_parquet("data/train_cands.parquet", columns=["qid", "rid"]).rename({"qid": "fqid"})
      .join(qmap, on="fqid").join(s1.select(pl.col("rid").cast(pl.UInt32), pl.col("entity_id").alias("cand")), on="rid"))
hit_k = kc.filter(pl.col("cand") == pl.col("s1_id")).select("qid").unique()
both = pl.concat([hit_k, hit_tf]).unique()
print(f"key blocker (current, top-8): recall {hit_k.height / pos:.4f} | UNION key + TF-IDF: recall {both.height / pos:.4f}")
print(f"records gaining new candidates: {tf.join(kc.select('qid', 'rid'), on=['qid', 'rid'], how='anti')['qid'].n_unique() / q.height:.1%}  "
      f"| total time {time.time() - t0:.0f}s")
