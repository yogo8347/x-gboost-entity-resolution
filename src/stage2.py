"""Stage 2: cluster-aware verification of each S2/S3 record's top S1 candidate.

Stage 1 scores pairs independently. Here every record is tentatively assigned to its argmax
S1, and an accept/reject model sees the *cluster* that forms around each S1:
  - how many other records chose the same S1 (per source) and how this one ranks among them
  - coherence: similarity of this record to the strongest other member of the same cluster
Trained out-of-fold on held-out S1 entities (stage-1 scores there are out-of-sample).

  python src/stage2.py cv      # 2-fold CV on validation fold: F0.5 + threshold -> data/best2.json
  python src/stage2.py fit     # fit on whole validation fold -> data/model2.txt
  python src/stage2.py apply   # test -> output/*.tsv
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

from pipeline import load, load_gt, eval_fold, val_fold, top2, write_lists, log, DATA, KEEP
from metric import f05

def _caps():
    """max matches per S1 per source, measured on the TRAINING fold only (no validation labels)."""
    s1ids = pl.read_parquet("data/train_s1.parquet", columns=["entity_id"])
    tr = s1ids.filter(~val_fold(pl.col("entity_id")))["entity_id"]
    g = load_gt().filter(pl.col("s1_id").is_in(tr))
    g = g.with_columns(pl.col("entity_id").str.slice(0, 2).alias("src")).group_by("s1_id", "src").len()
    return {r[0]: int(r[1]) for r in g.group_by("src").agg(pl.col("len").max()).iter_rows()}


CAP = _caps()


def _cp(scorer, a, b):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)


CONF = 0.3  # membership threshold for drift-robust cluster statistics
ROBUST_DROP = {"k_all", "k_src", "rank_src", "rank_all"}  # junk-sensitive originals, replaced by kc_* / rankc_*


def feats_present(t):
    return [c for c in FEATS if c in t.columns]


def cluster_feats(top, q, s1):
    """top: qid, rid, p1, p2 (argmax per record). Returns one row per record with features."""
    t = top.join(q.select("qid", "entity_id", "name", "core", "addr", "nums"), on="qid")
    t = t.with_columns(pl.col("entity_id").str.slice(0, 2).alias("src"))
    t = t.with_columns(
        (pl.col("p1") - pl.col("p2")).alias("gap"),
        (pl.len().over("rid") - 1).alias("k_all"),
        (pl.len().over(["rid", "src"]) - 1).alias("k_src"),
        pl.col("p1").rank("ordinal", descending=True).over(["rid", "src"]).alias("rank_src"),
        pl.col("p1").rank("ordinal", descending=True).over("rid").alias("rank_all"),
        (pl.col("p1").sum().over("rid") - pl.col("p1")).alias("sum_oth"),
        ((pl.col("p1") > 0.5).sum().over("rid") - (pl.col("p1") > 0.5)).alias("n_hi_oth"),
        ((pl.col("p1") > 0.5).sum().over(["rid", "src"]) - (pl.col("p1") > 0.5)).alias("n_hi_src_oth"),
    )
    # drift-robust cluster statistics: only records with real match evidence (p1 >= CONF) count as members, so
    # the number of junk/distractor records (27% in train vs ~37% in test) cannot change these values
    c = (pl.col("p1") >= CONF).cast(pl.Int32)
    t = t.with_columns((c.sum().over("rid") - c).alias("kc_all"),
                       (c.sum().over(["rid", "src"]) - c).alias("kc_src"))
    # rank among confident members only: junk records all rank below any confident record, so capping the plain
    # rank at (number of confident others + 1) removes their influence
    t = t.with_columns(pl.min_horizontal(pl.col("rank_all"), pl.col("kc_all") + 1).alias("rankc_all"),
                       pl.min_horizontal(pl.col("rank_src"), pl.col("kc_src") + 1).alias("rankc_src"))
    # anchor = strongest OTHER member of the same cluster (rank 1, or rank 2 if self is rank 1)
    ranked = t.select("rid", "qid", "p1", "core", "addr", "nums",
                      pl.col("p1").rank("ordinal", descending=True).over("rid").alias("r"))
    a1 = ranked.filter(pl.col("r") == 1).drop("r").rename(lambda c: c if c == "rid" else "a1_" + c)
    a2 = ranked.filter(pl.col("r") == 2).drop("r").rename(lambda c: c if c == "rid" else "a2_" + c)
    t = t.join(a1, on="rid", how="left").join(a2, on="rid", how="left")
    self_top = pl.col("a1_qid") == pl.col("qid")
    for c in ("core", "addr", "nums", "p1"):
        t = t.with_columns(pl.when(self_top).then(pl.col("a2_" + c)).otherwise(pl.col("a1_" + c)).alias("anc_" + c))
    t = t.drop([c for c in t.columns if c.startswith(("a1_", "a2_"))])
    has = t["anc_core"].is_not_null().to_numpy()
    g = lambda c: t[c].fill_null("").to_list()
    feats = {
        "anc_name": _cp(fuzz.token_set_ratio, g("core"), g("anc_core")),
        "anc_addr": _cp(fuzz.token_set_ratio, g("addr"), g("anc_addr")),
        "anc_nums": _cp(fuzz.token_set_ratio, g("nums"), g("anc_nums")),
    }
    for k in feats:
        feats[k][~has] = -1
    t = t.with_columns([pl.Series(k, v) for k, v in feats.items()])
    t = t.join(s1.select("rid", pl.col("core").str.len_chars().alias("s1_len")), on="rid")
    t = t.with_columns((pl.col("src") == "S3").cast(pl.Float32).alias("is_s3"),
                       pl.col("anc_p1").fill_null(-1))
    return t


FEATS = ["p1", "p2", "gap", "k_all", "k_src", "rank_src", "rank_all", "sum_oth", "n_hi_oth",
         "n_hi_src_oth", "anc_p1", "anc_name", "anc_addr", "anc_nums", "s1_len", "is_s3", "n_cand"] + KEEP


def decide(t, prob, thr):
    """accept if prob > thr, then enforce per-source caps (keep highest prob)."""
    d = t.select("qid", "rid", "src").with_columns(pl.Series("pr", prob))
    d = d.filter(pl.col("pr") > thr)
    d = d.with_columns(pl.col("pr").rank("ordinal", descending=True).over(["rid", "src"]).alias("r"))
    return d.filter(((pl.col("src") == "S2") & (pl.col("r") <= CAP["S2"])) |
                    ((pl.col("src") == "S3") & (pl.col("r") <= CAP["S3"])))


def _train(X, y, names):
    import lightgbm as lgb
    params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
                  feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, num_threads=8, verbose=-1)
    return lgb.train(params, lgb.Dataset(X, y, feature_name=names), 600)


def val_data():
    s1, q = load("train")
    gt = load_gt()
    top = top2(pl.read_parquet(f"{DATA}/train_scored.parquet"))
    s1v = s1.filter(eval_fold(pl.col("entity_id")))
    top = top.join(s1v.select("rid"), on="rid", how="semi")
    t = cluster_feats(top, q, s1)
    t = t.join(s1.select("rid", pl.col("entity_id").alias("s1_id")), on="rid")
    t = t.join(gt.rename({"s1_id": "true"}), on="entity_id", how="left")
    t = t.with_columns((pl.col("true") == pl.col("s1_id")).fill_null(False).cast(pl.Int8).alias("label"))
    return s1v, gt, t


def cv():
    s1v, gt, t = val_data()
    log("stage2 rows", t.height, "pos", t["label"].mean())
    # cache the (small) stage-2 table + truth so tuning / ensembling never reloads 10M records
    t.select("qid", "rid", "src", "entity_id", "s1_id", "label", *feats_present(t)).write_parquet(f"{DATA}/stage2_table.parquet")
    gt.filter(pl.col("s1_id").is_in(s1v["entity_id"].implode())).write_parquet(f"{DATA}/stage2_truth.parquet")
    s1v.select("entity_id").write_parquet(f"{DATA}/stage2_s1v.parquet")
    fold = (t["s1_id"].hash(seed=21) % 2).to_numpy()
    F = feats_present(t); X = t.select(F).to_numpy(); y = t["label"].to_numpy()
    prob = np.zeros(len(y))
    for k in (0, 1):
        m = _train(X[fold != k], y[fold != k], F)
        prob[fold == k] = m.predict(X[fold == k])
    ids = s1v["entity_id"].to_list()
    truth = {}
    for a, b in gt.filter(pl.col("s1_id").is_in(ids)).iter_rows():
        truth.setdefault(a, set()).add(b)
    best = (0, 0.5)
    for thr in np.arange(0.2, 0.95, 0.05):
        d = decide(t, prob, thr).join(t.select("qid", "entity_id", "s1_id"), on="qid")
        pred = {}
        for a, b in zip(d["s1_id"].to_list(), d["entity_id"].to_list()):
            pred.setdefault(a, set()).add(b)
        sc = f05(pred, truth, ids)
        log(f"stage2 thr={thr:.2f} F0.5={sc:.5f}")
        if sc > best[0]:
            best = (sc, float(thr))
    log(f"STAGE2 BEST F0.5={best[0]:.5f} thr={best[1]:.2f}")
    json.dump({"f05": best[0], "thr": best[1]}, open(f"{DATA}/best2.json", "w"))


def fit():
    _, _, t = val_data()
    m = _train(t.select(feats_present(t)).to_numpy(), t["label"].to_numpy(), feats_present(t))
    m.save_model(f"{DATA}/model2.txt")
    log("stage2 model saved")


def apply():
    import lightgbm as lgb
    s1, q = load("test")
    S = pl.read_parquet(f"{DATA}/test_scored.parquet")
    t = cluster_feats(top2(S), q, s1)
    final = f"{DATA}/stage2_final.json"
    if os.path.exists(final):  # tuned / ensembled stage 2 chosen on an untouched half (tune_stage2.py)
        import xgboost as xgb
        cfg = json.load(open(final))
        Xt = t.select(cfg["feats"]).to_numpy().astype(np.float32)
        prob = np.zeros(t.height)
        if cfg["kind"] in ("lgb", "ensemble"):
            w = cfg.get("w_lgb", 1.0)
            prob += w * lgb.Booster(model_file=f"{DATA}/model2_lgb.txt").predict(Xt)
        if cfg["kind"] in ("xgb", "ensemble"):
            w = 1.0 - cfg.get("w_lgb", 0.0)
            mx = xgb.Booster(); mx.load_model(f"{DATA}/model2_xgb.json")
            prob += w * mx.predict(xgb.DMatrix(Xt))
        thr = cfg["thr"]
        log("stage2 apply using", cfg["kind"], "thr", round(thr, 3))
    else:
        b = json.load(open(f"{DATA}/best2.json"))
        m = lgb.Booster(model_file=f"{DATA}/model2.txt")
        prob, thr = m.predict(t.select(m.feature_name()).to_numpy()), b["thr"]
    d = decide(t, prob, thr)
    idq = q.select("qid", pl.col("entity_id").alias("e"))
    ids1 = s1.select("rid", pl.col("entity_id").alias("s1"))
    os.makedirs("output", exist_ok=True)
    cand = S.select("qid", "rid").join(idq, on="qid").join(ids1, on="rid")
    write_lists(s1, cand, "candidate_entity_ids", "output/candidate_pairs.tsv")
    a = d.join(idq, on="qid").join(ids1, on="rid")
    res = write_lists(s1, a, "matched_entity_ids", "output/matching_results.tsv")
    log("stage2 written; matched", a.height, "empty rows", (res["matched_entity_ids"] == "").sum())
    log(a.join(s1.select(pl.col("entity_id").alias("s1"), "country"), on="s1")
        .group_by("country").len())


if __name__ == "__main__":
    {"cv": cv, "fit": fit, "apply": apply}[sys.argv[1]]()
