"""France patch: re-normalise France addresses (region / department names dropped, see normalize.FR_ADMIN),
then re-block and re-score France only. Blocking keys carry the country, so France records are only ever compared
with France S1 entities: a France-only rerun gives exactly what a full rerun would. Train data has no France, so the
trained models stay valid. Non-France rows are copied unchanged.

Writes data/frpatch/: test_s{1,2,3}.parquet, test_cands.parquet, test_scored.parquet, and output_v2fr/ (v2 stage 2).
  python src/france_patch.py
"""
import gc, os, shutil, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import polars as pl
import lightgbm as lgb

from normalize import norm_addr
from blocking import build_index, generate
from features import compute
from pipeline import KEEP, log, write_lists

OUT = "data/frpatch"
os.makedirs(OUT, exist_ok=True)
T0 = time.time()

# 1. re-normalise France addresses in the cached test parquets
for k in (1, 2, 3):
    dst = f"{OUT}/test_s{k}.parquet"
    if os.path.exists(dst):
        continue
    df = pl.read_parquet(f"data/test_s{k}.parquet")
    raw = pl.read_csv(f"dataset/test/test_source{k}.tsv", separator="\t", quote_char=None, infer_schema=False,
                      columns=["entity_id", "business_address"])
    fr = df.filter(pl.col("country") == "France").select("entity_id").join(raw, on="entity_id", how="left")
    res = [norm_addr(a, "France") for a in fr["business_address"].to_list()]
    fix = fr.select("entity_id").with_columns(pl.Series("addr_n", [r[0] for r in res]),
                                              pl.Series("nums_n", [" ".join(r[1]) for r in res]),
                                              pl.Series("awords_n", [" ".join(r[2]) for r in res]))
    df = (df.join(fix, on="entity_id", how="left")
            .with_columns(pl.coalesce("addr_n", "addr").alias("addr"), pl.coalesce("nums_n", "nums").alias("nums"),
                          pl.coalesce("awords_n", "awords").alias("awords"))
            .drop("addr_n", "nums_n", "awords_n"))
    df.write_parquet(dst)
    log(f"source {k}: re-normalised {fix.height} France rows")

s1 = pl.read_parquet(f"{OUT}/test_s1.parquet").with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
q = (pl.concat([pl.read_parquet(f"{OUT}/test_s{k}.parquet") for k in (2, 3)])
     .with_row_index("qid").with_columns(pl.col("qid").cast(pl.UInt32)))
fr_rid = s1.filter(pl.col("country") == "France").select("rid")
fr_q = q.filter(pl.col("country") == "France")
log("France S1", fr_rid.height, "France records", fr_q.height)

# 2. re-block France only (IDF uses the full S1 count, exactly as in the full run)
if not os.path.exists(f"{OUT}/fr_cands.parquet"):
    keys, dfreq = build_index(s1.filter(pl.col("country") == "France").select("rid", "country", "core", "dba", "nums", "awords"))
    c = generate(keys, dfreq, fr_q.select("qid", "country", "core", "dba", "nums", "awords"), s1.height)
    c = c.with_columns(pl.len().over("rid").cast(pl.Float32).alias("s_ncand"))
    c.write_parquet(f"{OUT}/fr_cands.parquet")
    del keys, dfreq; gc.collect()
frc = pl.read_parquet(f"{OUT}/fr_cands.parquet")
old = pl.scan_parquet("data/test_cands.parquet").join(fr_rid.lazy(), on="rid", how="anti")
pl.concat([old, frc.lazy().select(old.collect_schema().names())]).sink_parquet(f"{OUT}/test_cands.parquet")
log("France candidate pairs", frc.height, round(frc.height / fr_q.height, 2), "per record")

# 3. re-score France pairs with the unchanged stage-1 model
if not os.path.exists(f"{OUT}/fr_scored.parquet"):
    m = lgb.Booster(model_file="data/model.txt"); names = m.feature_name()
    qids = frc.select("qid").unique().sort("qid")["qid"]
    parts = []
    for i in range(0, len(qids), 300_000):
        sub = frc.join(pl.DataFrame({"qid": qids[i:i + 300_000]}), on="qid", how="semi")
        f = compute(sub, q, s1)
        p = m.predict(f.select(names).to_numpy(), num_threads=8)
        parts.append(f.select("qid", "rid", *KEEP).with_columns(pl.Series("p", p.astype(np.float32))))
        log("scored France chunk", i)
    pl.concat(parts).write_parquet(f"{OUT}/fr_scored.parquet")
frs = pl.read_parquet(f"{OUT}/fr_scored.parquet")
olds = pl.scan_parquet("data/test_scored.parquet").join(fr_rid.lazy(), on="rid", how="anti")
pl.concat([olds, frs.lazy().select(olds.collect_schema().names())]).sink_parquet(f"{OUT}/test_scored.parquet")
log("merged scored table written")

# 4. v2 stage 2 on the patched scores -> output_v2fr/ (both files, candidates from the patched blocking)
from stage2 import cluster_feats, decide
from pipeline import top2
import json
S = pl.read_parquet(f"{OUT}/test_scored.parquet")
t = cluster_feats(top2(S), q, s1)
m2 = lgb.Booster(model_file="data/model2.txt")
thr = json.load(open("data/best2.json"))["thr"]
d = decide(t, m2.predict(t.select(m2.feature_name()).to_numpy()), thr)
idq = q.select("qid", pl.col("entity_id").alias("e")); ids1 = s1.select("rid", pl.col("entity_id").alias("s1"))
os.makedirs("output_v2fr", exist_ok=True)
write_lists(s1, S.select("qid", "rid").join(idq, on="qid").join(ids1, on="rid"), "candidate_entity_ids", "output_v2fr/candidate_pairs.tsv")
a = d.join(idq, on="qid").join(ids1, on="rid")
res = write_lists(s1, a, "matched_entity_ids", "output_v2fr/matching_results.tsv")
fr_ids = set(s1.filter(pl.col("country") == "France")["entity_id"].to_list())
fr_n = a.filter(pl.col("s1").is_in(list(fr_ids))).height
log(f"output_v2fr written: matched {a.height} (France {fr_n}), empty rows {(res['matched_entity_ids'] == '').sum()}")
log("FRANCE PATCH DONE", round(time.time() - T0), "s")
