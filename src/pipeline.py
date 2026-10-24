"""End-to-end pipeline.

  python src/pipeline.py block  <split>      # candidates for every S2/S3 record -> data/<split>_cands.parquet
  python src/pipeline.py train               # features on train-fold sample + LightGBM -> data/model.txt
  python src/pipeline.py infer  <split>      # score all candidates -> data/<split>_scored.parquet
  python src/pipeline.py evaluate            # macro F0.5 on held-out S1 fold, threshold sweep
  python src/pipeline.py submit [threshold]  # output/matching_results.tsv + output/candidate_pairs.tsv
"""
import gc, glob, json, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import polars as pl

from blocking import build_index, generate
from features import compute, feature_names
from metric import f05

DATA = os.environ.get("ER_DATA", "data")
PREP = os.environ.get("ER_PREP", "data")  # normalised source parquets  # ER_DATA=data/v1 re-evaluates the v1 artefacts
# stage-1 features carried into the scored table for stage 2
KEEP = ["n_clean_ratio", "n_phon_ratio", "a_tset", "num_first_eq", "num_conflict", "num_all_eq",
        "s_name_dup", "a_q_empty", "comb2_gap2"]
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


def load(split):
    s1 = pl.read_parquet(f"{PREP}/{split}_s1.parquet").with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
    q = pl.concat([pl.read_parquet(f"{PREP}/{split}_s{k}.parquet") for k in (2, 3)])
    q = q.with_row_index("qid").with_columns(pl.col("qid").cast(pl.UInt32))
    return s1, q


def load_gt():
    gt = (pl.read_csv("dataset/train/train_ground_truth.tsv", separator="\t", infer_schema=False)
          .with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
          .drop_nulls().rename({"matched_entity_ids": "entity_id", "source1_entity_id": "s1_id"}))
    return gt


def val_fold(ids):
    """20% of S1 entities held out from stage-1 training (deterministic hash)."""
    return ids.hash(seed=11) % 5 == 0


def eval_fold(ids):
    """Evaluation uses the FULL 20% held-out fold (same as val_fold)."""
    return val_fold(ids)


# ------------------------------------------------------------------ stage 1: blocking
def block(split):
    """Memory-lean: S1 with only blocking columns; queries streamed part by part from parquet."""
    cols = ["country", "core", "dba", "nums", "awords"]
    s1 = (pl.read_parquet(f"{PREP}/{split}_s1.parquet", columns=cols)
          .with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32)))
    qscan = (pl.concat([pl.scan_parquet(f"{PREP}/{split}_s{k}.parquet").select(cols) for k in (2, 3)])
             .with_row_index("qid").with_columns(pl.col("qid").cast(pl.UInt32)))
    nq = qscan.select(pl.len()).collect().item()
    log("loaded", s1.height, nq)
    keys, dfreq = build_index(s1)
    n1 = s1.height
    del s1; gc.collect()
    log("index", keys.height)
    parts = f"{DATA}/{split}_cands_parts"
    os.makedirs(parts, exist_ok=True)
    step = 1_000_000
    for st in range(0, nq, step):
        dst = f"{parts}/{st:09d}.parquet"
        if os.path.exists(dst):
            continue
        c = generate(keys, dfreq, qscan.slice(st, step).collect(), n1)
        c.write_parquet(dst)
        log("block", st, c.height)
        del c; gc.collect()
    del keys, dfreq; gc.collect()
    c = pl.read_parquet(f"{parts}/*.parquet")
    c = c.with_columns(pl.len().over("rid").cast(pl.Float32).alias("s_ncand"))
    c.write_parquet(f"{DATA}/{split}_cands.parquet")
    log("candidates", c.height, "per query", c.height / nq)


# ------------------------------------------------------------------ stage 2: training
TRAIN_TARGET = 1_500_000


def select_training_records(q, s1, call, target=TRAIN_TARGET):
    """LEAKAGE GUARD. q: records with entity_id + s1_id (true S1, null if unmatched); s1: rid + entity_id;
    call: all candidate pairs (qid, rid). A record is eligible for stage-1 training only if
      (a) its true S1 is in the training fold (unmatched records: their own id hashes to the training fold), AND
      (b) NONE of its candidates is a held-out S1.
    So no training pair involves a held-out S1, and every record that can land on a held-out S1 at
    evaluation time is excluded from training. Returns (training records, info)."""
    is_val_s1 = val_fold(s1["entity_id"])
    vals = s1.filter(is_val_s1)
    q = q.with_columns(pl.when(pl.col("s1_id").is_null()).then(val_fold(pl.col("entity_id")))
                       .otherwise(pl.col("s1_id").is_in(vals["entity_id"].implode())).alias("is_val"))
    touch_val = call.join(vals.select("rid"), on="rid", how="semi").select("qid").unique()
    elig = q.filter(~pl.col("is_val")).join(touch_val, on="qid", how="anti")
    frac = min(100, max(1, round(100 * target / max(elig.height, 1))))
    tq = elig.filter(pl.col("entity_id").hash(seed=5) % 100 < frac)
    return tq, {"eligible": elig.height, "frac": frac, "touch_val": touch_val.height}


def train():
    import lightgbm as lgb
    s1, q = load("train")
    gt = load_gt()
    q = q.join(gt, on="entity_id", how="left")
    s1 = s1.with_columns(val_fold(pl.col("entity_id")).alias("is_val"))
    call = pl.read_parquet(f"{DATA}/train_cands.parquet")
    tq, info = select_training_records(q, s1, call)
    log("eligible train records", info["eligible"], "sampled %", info["frac"])
    c = call.join(tq.select("qid"), on="qid", how="semi")
    del call
    log("train queries", tq.height, "pairs", c.height)
    feats = []
    for st in range(0, tq.height, 200_000):
        qq = tq.slice(st, 200_000)
        cc = c.join(qq.select("qid"), on="qid", how="semi")
        f = compute(cc, qq, s1)
        f = f.join(qq.select("qid", "s1_id"), on="qid").join(
            s1.select("rid", pl.col("entity_id").alias("cand")), on="rid")
        f = f.with_columns((pl.col("cand") == pl.col("s1_id")).fill_null(False).cast(pl.Int8).alias("label"))
        feats.append(f.drop("s1_id", "cand"))
        log("feats", st)
    F = pl.concat(feats)
    F.write_parquet(f"{DATA}/train_feats.parquet")
    names = feature_names(F)
    log("pairs", F.height, "pos rate", F["label"].mean(), "features", len(names))
    # hold out 10% of training queries for early stopping
    es = (F["qid"].hash(seed=9) % 10 == 0).to_numpy()
    X = F.select(names).to_numpy(); y = F["label"].to_numpy()
    params = dict(objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  num_threads=8, verbose=-1)
    dtr = lgb.Dataset(X[~es], y[~es], feature_name=names)
    dva = lgb.Dataset(X[es], y[es], reference=dtr)
    m = lgb.train(params, dtr, 1500, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    m.save_model(f"{DATA}/model.txt")
    imp = sorted(zip(names, m.feature_importance("gain")), key=lambda x: -x[1])
    log("top features", [(n, int(v)) for n, v in imp[:15]])


# ------------------------------------------------------------------ stage 3: inference
def infer(split):
    import lightgbm as lgb
    m = lgb.Booster(model_file=f"{DATA}/model.txt")
    names = m.feature_name()
    s1, q = load(split)
    c = pl.read_parquet(f"{DATA}/{split}_cands.parquet")
    if split == "train":  # evaluation only needs records that can land on a held-out S1
        vr = s1.filter(eval_fold(pl.col("entity_id"))).select("rid")
        c = c.join(c.join(vr, on="rid", how="semi").select("qid").unique(), on="qid", how="semi")
        log("train eval subset pairs", c.height)
    parts = f"{DATA}/{split}_scored_parts"
    os.makedirs(parts, exist_ok=True)
    step = 300_000
    for st in range(0, q.height, step):
        dst = f"{parts}/{st:09d}.parquet"
        if os.path.exists(dst):
            continue
        qq = q.slice(st, step)
        cc = c.filter(pl.col("qid").is_between(st, st + step - 1))
        if cc.height == 0:
            continue
        f = compute(cc, qq, s1)
        p = m.predict(f.select(names).to_numpy(), num_threads=8)
        f.select("qid", "rid", *KEEP).with_columns(pl.Series("p", p.astype(np.float32))).write_parquet(dst)
        if (st // step) % 5 == 0:
            log("infer", st, "/", q.height)
        del f; gc.collect()
    S = pl.read_parquet(f"{parts}/*.parquet")
    S.write_parquet(f"{DATA}/{split}_scored.parquet")
    log("scored pairs", S.height)


def top2(S):
    """per S2/S3 record: best S1 candidate, its prob p1, runner-up prob p2."""
    S = S.sort(["qid", "p"], descending=[False, True])
    extra = [pl.col(k).first() for k in KEEP if k in S.columns]
    return S.group_by("qid", maintain_order=True).agg(
        pl.col("rid").first(), pl.col("p").first().alias("p1"),
        pl.col("p").slice(1, 1).first().fill_null(0).alias("p2"),
        pl.len().alias("n_cand"), *extra)


def assign(top, thr, margin=0.0):
    """each S2/S3 record -> its argmax S1 if p1 > thr and p1 - p2 >= margin (1-to-1 by construction)."""
    return top.filter((pl.col("p1") > thr) & (pl.col("p1") - pl.col("p2") >= margin))


def evaluate():
    s1, q = load("train")
    top = top2(pl.read_parquet(f"{DATA}/train_scored.parquet"))
    gt = load_gt()
    s1v = s1.filter(eval_fold(pl.col("entity_id")))
    ids = s1v["entity_id"].to_list()
    truth = {}
    for a, b in gt.filter(pl.col("s1_id").is_in(ids)).iter_rows():
        truth.setdefault(a, set()).add(b)
    rid2 = dict(zip(s1v["rid"].to_list(), ids))
    topv = top.filter(pl.col("rid").is_in(list(rid2)))
    qid2 = q["entity_id"]
    best = (0, 0.5, 0.0)
    for thr in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        for margin in (0.0, 0.1, 0.3):
            a = assign(topv, thr, margin)
            pred = {}
            for qi, ri in zip(a["qid"].to_list(), a["rid"].to_list()):
                pred.setdefault(rid2[ri], set()).add(qid2[qi])
            sc = f05(pred, truth, ids)
            log(f"thr={thr} margin={margin} F0.5={sc:.5f}")
            if sc > best[0]:
                best = (sc, thr, margin)
    log(f"BEST F0.5={best[0]:.5f} thr={best[1]} margin={best[2]}  (val S1 entities: {len(ids)})")
    json.dump({"f05": best[0], "thr": best[1], "margin": best[2]}, open(f"{DATA}/best.json", "w"))


# ------------------------------------------------------------------ outputs
def write_lists(s1, pairs, col, path):
    lists = pairs.group_by("s1").agg(pl.col("e").unique().sort().str.join(",").alias(col))
    out = (s1.select(pl.col("entity_id").alias("source1_entity_id"))
           .join(lists, left_on="source1_entity_id", right_on="s1", how="left")
           .with_columns(pl.col(col).fill_null("")))
    out.write_csv(path, separator="\t", quote_style="never")
    return out


def submit():
    b = json.load(open(f"{DATA}/best.json"))
    s1, q = load("test")
    S = pl.read_parquet(f"{DATA}/test_scored.parquet")
    idq = q.select("qid", pl.col("entity_id").alias("e"))
    ids1 = s1.select("rid", pl.col("entity_id").alias("s1"))
    os.makedirs("output", exist_ok=True)
    cand = S.select("qid", "rid").join(idq, on="qid").join(ids1, on="rid")
    write_lists(s1, cand, "candidate_entity_ids", "output/candidate_pairs.tsv")
    a = assign(top2(S), b["thr"], b["margin"]).join(idq, on="qid").join(ids1, on="rid")
    res = write_lists(s1, a, "matched_entity_ids", "output/matching_results.tsv")
    log("written; matched", a.height, "empty rows", (res["matched_entity_ids"] == "").sum())


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "block":
        block(sys.argv[2])
    elif cmd == "train":
        train()
    elif cmd == "infer":
        infer(sys.argv[2])
    elif cmd == "evaluate":
        evaluate()
    elif cmd == "submit":
        submit()
