"""v11 hybrid blocking: key-based candidates (blocking.py) UNION a character-trigram TF-IDF leg, searched in both
directions, merged with adaptive K. Design follows the SABER hybrid blocker and Sparkly (VLDB 2023).

Trigram leg, per country (blocking never crosses countries):
  text  = core name (twice) + address words, both translated with the learned dictionary
  tfidf = char_wb 3-grams, sublinear tf, min_df 2, max_df 0.05 (fitted on S1)
  forward  record -> S1 : top-10, adaptive K: keep rank < 3 or cos >= top1 - 0.15
  reverse  S1 -> record : top-5 records per S1
Output pairs carry  x_tcos (trigram cosine, computed for EVERY merged pair), x_in_key, x_in_tf ; brank of
trigram-only pairs = 8 + trigram rank (key pairs keep their 0..7 rank); s_ncand recomputed on the merged set.

  python src/hybrid_block.py <split> <prep_dir> <key_cands.parquet> <out_cands.parquet>
"""
import gc, json, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

split, prep, key_p, out_p = sys.argv[1:5]
KMIN, KMAX, GAP, KREV, THR = 3, 10, 0.15, 5, 0.1
D = json.load(open("data/translit_dict.json"))
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def text(df):
    tr = lambda col, d: pl.col(col).fill_null("").str.split(" ").list.eval(pl.element().replace(d)).list.join(" ")
    return df.select((tr("core", D["core"]) + " " + tr("core", D["core"]) + " " + tr("awords", D["addr"])).alias("t"))["t"].to_list()


s1 = pl.read_parquet(f"{prep}/{split}_s1.parquet", columns=["country", "core", "awords"]).with_row_index("rid")
q = (pl.concat([pl.read_parquet(f"{prep}/{split}_s{k}.parquet", columns=["country", "core", "awords"]) for k in (2, 3)])
     .with_row_index("qid"))
parts, cos_parts = [], []
key = pl.read_parquet(key_p, columns=["qid", "rid", "bscore", "brank"])
for ctry in s1["country"].unique().sort().to_list():
    a = s1.filter(pl.col("country") == ctry); b = q.filter(pl.col("country") == ctry)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), sublinear_tf=True, min_df=2, max_df=0.05, dtype=np.float32)
    A = vec.fit_transform(text(a)).tocsr(); B = vec.transform(text(b)).tocsr()
    AT, BT = A.T.tocsr(), B.T.tocsr()
    arid, bqid = a["rid"].to_numpy(), b["qid"].to_numpy()
    log(f"{ctry}: S1 {A.shape[0]} records {B.shape[0]} vocab {A.shape[1]}")
    for st in range(0, B.shape[0], 400_000):                         # forward: record -> S1
        M = sp_matmul_topn(B[st:st + 400_000], AT, top_n=KMAX, threshold=THR, n_threads=8).tocoo()
        f = pl.DataFrame({"qid": bqid[st + M.row], "rid": arid[M.col], "c": M.data})
        f = f.with_columns(pl.col("c").rank("ordinal", descending=True).over("qid").alias("r"), pl.col("c").max().over("qid").alias("top"))
        parts.append(f.filter((pl.col("r") <= KMIN) | (pl.col("c") >= pl.col("top") - GAP)).select("qid", "rid", pl.col("r").cast(pl.UInt8).alias("trank")))
    log(f"{ctry}: forward done")
    for st in range(0, A.shape[0], 150_000):                         # reverse: S1 -> record
        M = sp_matmul_topn(A[st:st + 150_000], BT, top_n=KREV, threshold=THR, n_threads=8).tocoo()
        parts.append(pl.DataFrame({"qid": bqid[M.col], "rid": arid[st + M.row], "trank": np.full(M.nnz, 10, np.uint8)}))
    log(f"{ctry}: reverse done")
    # trigram cosine for every pair (key + trigram) of this country, via row-wise sparse dot products
    tf_c = pl.concat([p for p in parts if p.height]).group_by("qid", "rid").agg(pl.col("trank").min())
    kc = key.join(pl.DataFrame({"qid": bqid}), on="qid", how="semi").select("qid", "rid")
    allp = pl.concat([kc, tf_c.select("qid", "rid")]).unique()
    qpos = {v: i for i, v in enumerate(bqid)}; rpos = {v: i for i, v in enumerate(arid)}
    qi = np.fromiter((qpos[v] for v in allp["qid"].to_numpy()), np.int64, allp.height)
    ri = np.fromiter((rpos[v] for v in allp["rid"].to_numpy()), np.int64, allp.height)
    cs = np.zeros(allp.height, np.float32)
    for st in range(0, allp.height, 2_000_000):
        cs[st:st + 2_000_000] = np.asarray(B[qi[st:st + 2_000_000]].multiply(A[ri[st:st + 2_000_000]]).sum(1)).ravel()
    cos_parts.append(allp.with_columns(pl.Series("x_tcos", cs)))
    del A, B, AT, BT, M; gc.collect()
    log(f"{ctry}: cosine for {allp.height} pairs")

tf = pl.concat(parts).group_by("qid", "rid").agg(pl.col("trank").min())
cos = pl.concat(cos_parts)
m = (key.join(tf, on=["qid", "rid"], how="full", coalesce=True)
        .with_columns(pl.col("bscore").is_not_null().cast(pl.Float32).alias("x_in_key"),
                      pl.col("trank").is_not_null().cast(pl.Float32).alias("x_in_tf"),
                      pl.col("bscore").fill_null(0.0).cast(pl.Float32),
                      pl.coalesce(pl.col("brank"), (pl.col("trank") + 8)).cast(pl.UInt8).alias("brank"))
        .drop("trank").join(cos, on=["qid", "rid"], how="left").with_columns(pl.col("x_tcos").fill_null(0.0)))
m = m.with_columns(pl.col("qid").cast(pl.UInt32), pl.col("rid").cast(pl.UInt32),
                   pl.len().over("rid").cast(pl.Float32).alias("s_ncand"))
m.write_parquet(out_p)
log(f"merged candidates {m.height} ({m.height / q.height:.2f} per record) | key {key.height} | trigram-only {int((m['x_in_key'] == 0).sum())}")
