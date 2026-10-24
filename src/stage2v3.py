"""Stage 2 v3: richer cluster-aware verifier.

Adds to the v2 stage-2 inputs:
  * t1_*  the full stage-1 feature vector of the record's best S1 candidate, computed against the
          runner-up only (so "gap" features describe best vs runner-up)
  * t2_*  key similarities of the runner-up candidate (why did it lose?)
  * cl_*  coherence with EVERY other record assigned to the same S1 (not just one anchor):
          mean / min / max name and address similarity, agreement of house numbers with the cluster
          and with S1, share of confident members
Leakage: built only from out-of-sample stage-1 scores of held-out records (train) or test scores (test);
labels are attached afterwards for the held-out fold only.

  python src/stage2v3.py build train   -> data/stage2v3_train.parquet (+ label)
  python src/stage2v3.py build test    -> data/stage2v3_test.parquet
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

from features import compute
from pipeline import load_gt, eval_fold, log
from stage2 import cluster_feats, CONF
from decoy_markers import markers

DATA = os.environ.get("ER_DATA", "data")
PREP = os.environ.get("ER_PREP", "data")  # normalised source parquets
T1_KEEP = None  # all stage-1 features


def _cp(scorer, a, b):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)


def lean_load(split, qids=None):
    s1 = pl.read_parquet(f"{PREP}/{split}_s1.parquet").with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
    q = (pl.concat([pl.scan_parquet(f"{PREP}/{split}_s{k}.parquet") for k in (2, 3)])
         .with_row_index("qid").with_columns(pl.col("qid").cast(pl.UInt32)))
    if qids is not None:
        q = q.join(qids.lazy(), on="qid", how="semi")
    return s1, q.collect()


def top_two(split):
    S = pl.scan_parquet(f"{DATA}/{split}_scored.parquet").select("qid", "rid", "p")
    S = S.sort(["qid", "p"], descending=[False, True]).group_by("qid", maintain_order=True).head(2)
    return S.with_columns(pl.int_range(pl.len()).over("qid").alias("k")).collect(engine="streaming")


NB = 8  # buckets keyed by S1 (rid): every cluster lives in exactly one bucket -> identical results, 1/8 memory


def build(split):
    t0 = time.time()
    tt = top_two(split)
    log("top-2 pairs", tt.height)
    s1_ids = pl.read_parquet(f"{PREP}/{split}_s1.parquet", columns=["entity_id"]).with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
    top1 = tt.filter(pl.col("k") == 0)
    if split == "train":  # only records whose best candidate is a held-out S1
        vr = s1_ids.filter(eval_fold(pl.col("entity_id"))).select("rid")
        top1 = top1.join(vr, on="rid", how="semi")
    top1 = top1.with_columns((pl.col("rid").hash(seed=5) % NB).alias("b"))
    parts_dir = f"{DATA}/stage2v3_{split}_parts"
    os.makedirs(parts_dir, exist_ok=True)
    for b in range(NB):
        dst = f"{parts_dir}/{b}.parquet"
        if os.path.exists(dst):
            continue
        tb = top1.filter(pl.col("b") == b)
        _build_bucket(split, tt.join(tb.select("qid"), on="qid", how="semi"), tb).write_parquet(dst)
        log("bucket", b, "done", round(time.time() - t0), "s")
    out = pl.read_parquet(f"{parts_dir}/*.parquet")
    out.write_parquet(f"{DATA}/stage2v3_{split}.parquet")
    log("stage2v3", split, out.shape, round(time.time() - t0), "s")


def _build_bucket(split, tt, top1):
    s1, q = lean_load(split, top1.select("qid"))
    log("records", top1.height, "loaded", q.height)

    # --- v2 stage-2 inputs (cluster sizes, ranks, anchor similarity, KEEP features)
    from pipeline import top2
    S = pl.scan_parquet(f"{DATA}/{split}_scored.parquet").join(tt.select("qid").unique().lazy(), on="qid", how="semi").collect()
    base = cluster_feats(top2(S), q, s1)
    del S

    # --- t1_/t2_: stage-1 features of best and runner-up, computed on the top-2 set only
    pairs = tt.select("qid", "rid").with_columns(pl.lit(0.0, pl.Float32).alias("bscore"), pl.lit(0, pl.UInt8).alias("brank"))
    qs = pairs.select("qid").unique().sort("qid")["qid"]
    parts = []
    for i in range(0, len(qs), 1_000_000):  # bounded memory
        sub = pairs.join(pl.DataFrame({"qid": qs[i:i + 1_000_000]}), on="qid", how="semi")
        parts.append(compute(sub, q, s1))
    f = pl.concat(parts).join(tt.select("qid", "rid", "k"), on=["qid", "rid"])
    del parts
    fcols = [c for c in f.columns if c not in ("qid", "rid", "k", "bscore", "brank", "bscore_gap", "bscore_gap2")]
    t1 = f.filter(pl.col("k") == 0).select("qid", *[pl.col(c).alias("t1_" + c) for c in fcols])
    t2k = ["n_clean_ratio", "n_phon_ratio", "a_tset", "num_first_eq", "num_conflict", "comb2", "s_name_dup"]
    t2 = f.filter(pl.col("k") == 1).select("qid", *[pl.col(c).alias("t2_" + c) for c in t2k])
    # decoy markers on the best pair (legal-form conflict, house-number relation type, word substitution vs typo)
    b1 = (tt.filter(pl.col("k") == 0).select("qid", "rid")
          .join(q.select("qid", "name", "nums", "core"), on="qid")
          .join(s1.select("rid", pl.col("name").alias("s_name"), pl.col("nums").alias("s_nums"), pl.col("core").alias("s_core")), on="rid"))
    g = lambda c: b1[c].fill_null("").to_list()
    dm = markers(g("name"), g("s_name"), g("nums"), g("s_nums"), g("core"), g("s_core"))
    t1 = t1.join(b1.select("qid").with_columns([pl.Series(k, v) for k, v in dm.items()]), on="qid", how="left")
    log("pair features", f.height)
    del f

    # --- cl_: coherence with every other member of the same S1 cluster
    mem = (base.select("qid", "rid", "p1").join(q.select("qid", "core", "addr", "nums"), on="qid")
           .with_columns(pl.col("nums").str.split(" ").list.first().fill_null("").alias("n1")))
    s1n = s1.select("rid", pl.col("nums").str.split(" ").list.first().fill_null("").alias("s_n1"))
    mem = mem.join(s1n, on="rid", how="left")
    pr = mem.join(mem.select("rid", pl.col("qid").alias("o"), pl.col("core").alias("o_core"),
                             pl.col("addr").alias("o_addr"), pl.col("n1").alias("o_n1"), pl.col("p1").alias("o_p1")),
                  on="rid").filter((pl.col("qid") != pl.col("o")) & (pl.col("o_p1") >= CONF))  # confident members only (drift-robust)
    log("cluster member pairs", pr.height)
    g = lambda c: pr[c].fill_null("").to_list()
    pr = pr.with_columns(pl.Series("sn", _cp(fuzz.token_set_ratio, g("core"), g("o_core"))),
                         pl.Series("sa", _cp(fuzz.token_set_ratio, g("addr"), g("o_addr"))),
                         ((pl.col("n1") == pl.col("o_n1")) & (pl.col("n1") != "")).cast(pl.Float32).alias("same_n"),
                         (pl.col("o_p1") > 0.7).cast(pl.Float32).alias("o_conf"))
    cl = pr.group_by("qid").agg(
        pl.col("sn").mean().alias("cl_name_mean"), pl.col("sn").min().alias("cl_name_min"), pl.col("sn").max().alias("cl_name_max"),
        pl.col("sa").mean().alias("cl_addr_mean"), pl.col("sa").min().alias("cl_addr_min"), pl.col("sa").max().alias("cl_addr_max"),
        pl.col("same_n").mean().alias("cl_samenum_frac"), pl.col("o_conf").mean().alias("cl_conf_frac"),
        (pl.col("sn") * pl.col("o_p1")).sum().alias("cl_name_wsum"), pl.col("o_p1").sum().alias("cl_p_sum"))
    cl = cl.with_columns((pl.col("cl_name_wsum") / pl.col("cl_p_sum").clip(1e-6)).alias("cl_name_pw")).drop("cl_name_wsum")
    # does this record's house number agree with S1 while the cluster's does too?
    numagree = mem.with_columns(((pl.col("n1") == pl.col("s_n1")) & (pl.col("n1") != "")).cast(pl.Float32).alias("me_eq_s1"),
                                (pl.col("p1") >= CONF).cast(pl.Float32).alias("_c"))
    numagree = numagree.with_columns(
        ((pl.col("me_eq_s1") * pl.col("_c")).sum().over("rid") - pl.col("me_eq_s1") * pl.col("_c"))
        .truediv((pl.col("_c").sum().over("rid") - pl.col("_c")).clip(1)).alias("cl_oth_eq_s1"),
        (pl.col("n1") == "").cast(pl.Float32).alias("cl_me_nonum")).select("qid", "me_eq_s1", "cl_oth_eq_s1", "cl_me_nonum")

    text_cols = [c for c, dt in base.schema.items() if dt == pl.Utf8 and c not in ("src", "entity_id")]
    out = (base.drop(text_cols)
           .join(t1, on="qid", how="left").join(t2, on="qid", how="left")
           .join(cl, on="qid", how="left").join(numagree, on="qid", how="left"))
    out = out.join(s1.select("rid", pl.col("entity_id").alias("s1_id")), on="rid")
    if split == "train":
        gt = load_gt().rename({"s1_id": "true"})
        out = out.join(gt, on="entity_id", how="left").with_columns(
            (pl.col("true") == pl.col("s1_id")).fill_null(False).cast(pl.Int8).alias("label")).drop("true")
    return out


def apply(final_json=f"{DATA}/stage2v3_final.json", model_prefix=f"{DATA}/model2v3", out_dir="output_v3"):
    """test: v3 table -> tuned model(s) -> threshold + train-fold caps -> both TSVs in out_dir (v2 output untouched)."""
    import json
    import lightgbm as lgb
    import xgboost as xgb
    from stage2 import decide
    from pipeline import write_lists
    cfg = json.load(open(final_json))
    t = pl.read_parquet(f"{DATA}/stage2v3_test.parquet")
    Xt = t.select(cfg["feats"]).to_numpy().astype(np.float32)
    prob = np.zeros(t.height)
    if cfg["kind"] in ("lgb", "ensemble"):
        prob += cfg.get("w_lgb", 1.0) * lgb.Booster(model_file=model_prefix + "_lgb.txt").predict(Xt)
    if cfg["kind"] in ("xgb", "ensemble"):
        mx = xgb.Booster(); mx.load_model(model_prefix + "_xgb.json")
        prob += (1.0 - cfg.get("w_lgb", 0.0)) * mx.predict(xgb.DMatrix(Xt))
    d = decide(t, prob, cfg["thr"])
    s1 = pl.read_parquet("data/test_s1.parquet", columns=["entity_id"]).with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
    q = (pl.concat([pl.read_parquet(f"{PREP}/test_s{k}.parquet", columns=["entity_id"]) for k in (2, 3)])
         .with_row_index("qid").with_columns(pl.col("qid").cast(pl.UInt32)))
    idq = q.select("qid", pl.col("entity_id").alias("e")); ids1 = s1.select("rid", pl.col("entity_id").alias("s1"))
    os.makedirs(out_dir, exist_ok=True)
    cand = pl.read_parquet(f"{DATA}/test_scored.parquet", columns=["qid", "rid"]).join(idq, on="qid").join(ids1, on="rid")
    write_lists(s1, cand, "candidate_entity_ids", f"{out_dir}/candidate_pairs.tsv")
    a = d.join(idq, on="qid").join(ids1, on="rid")
    res = write_lists(s1, a, "matched_entity_ids", f"{out_dir}/matching_results.tsv")
    log("v3 written to", out_dir, "| matched", a.height, "| empty rows", (res["matched_entity_ids"] == "").sum())


if __name__ == "__main__":
    if sys.argv[1] == "apply":
        apply()
    else:
        build(sys.argv[2] if sys.argv[1] == "build" else sys.argv[1])
