"""Leak-free Optuna tuning + ensembling of the stage-2 verifier, scored on an untouched half.

Protocol (all on the held-out 20% S1 fold, never on stage-1 training records):
  * held-out S1 entities are split by hash into A (tuning) and B (reporting).
  * Optuna maximises macro F0.5 on A using 2-fold CV *inside A* (threshold picked inside A).
  * every candidate (baseline, tuned LGBM, XGBoost, ensemble) is then trained on ALL of A, its threshold
    taken from A's out-of-fold predictions, and scored ONCE on B, which no choice ever looked at.
  * the winner is refit on A+B for the test set -> data/model2*.txt / .json, data/best2.json

  python src/tune_stage2.py [n_trials]
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import polars as pl
import lightgbm as lgb
import xgboost as xgb
import optuna

from metric import f05
from stage2 import CAP

DATA = os.environ.get("ER_DATA", "data")
T0 = time.time()
THREADS = int(os.environ.get("TUNE_THREADS", "4"))  # leave cores for the test pipeline running alongside


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


t = pl.read_parquet(os.environ.get("TABLE", f"{DATA}/stage2_table.parquet"))
truth_df = pl.read_parquet(f"{DATA}/stage2_truth.parquet")
s1v = pl.read_parquet(f"{DATA}/stage2_s1v.parquet")["entity_id"]
import re as _re
_EXCL = os.environ.get("EXCLUDE", "")  # regex of features to leave out (e.g. drift-sensitive ones)
FEATS = [c for c, dt in t.schema.items() if c not in ("qid", "rid", "src", "entity_id", "s1_id", "label") and dt.is_numeric()
         and not (_EXCL and _re.fullmatch(_EXCL, c))]
X = t.select(FEATS).to_numpy().astype(np.float32)
y = t["label"].to_numpy()
half = (t["s1_id"].hash(seed=77) % 2).to_numpy()          # 0 = A (tune), 1 = B (report)
inner = (t["s1_id"].hash(seed=21) % 2).to_numpy()         # CV folds inside A, keyed by S1 entity
s1_half = (s1v.hash(seed=77) % 2).to_numpy()
IDS = {0: s1v.filter(s1_half == 0).to_list(), 1: s1v.filter(s1_half == 1).to_list()}
TRUTH = {}
for a, b in truth_df.select("s1_id", "entity_id").iter_rows():
    TRUTH.setdefault(a, set()).add(b)
SRC = t["src"].to_numpy(); S1 = t["s1_id"].to_numpy(); EID = t["entity_id"].to_numpy()
# optional prior-shift correction: weight distractor rows (records matching no S1 at all) by DIST_WEIGHT during training,
# so the model is calibrated to a test set with more distractors (train ~27% vs test ~37% -> odds ratio ~1.6)
DW = float(os.environ.get("DIST_WEIGHT", "1.0"))
if DW != 1.0:
    _matched = set(pl.read_csv("dataset/train/train_ground_truth.tsv", separator="	", infer_schema=False)
                   .with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
                   .drop_nulls()["matched_entity_ids"].to_list())
    W = np.where(np.isin(EID, list(_matched)), 1.0, DW)
else:
    W = np.ones(len(t))
# optional covariate-shift weights (src/shift_weights.py): make training optimise for the test mix
if os.environ.get("SHIFT_WEIGHTS"):
    _sw = pl.read_parquet(os.environ["SHIFT_WEIGHTS"])
    W = W * t.select("qid").join(_sw, on="qid", how="left")["w"].fill_null(1.0).to_numpy()
    print("using shift weights", os.environ["SHIFT_WEIGHTS"], "mean", round(float(W.mean()), 3))
log("rows", t.height, "features", len(FEATS), "A rows", int((half == 0).sum()), "B rows", int((half == 1).sum()))


def score(mask, prob, thr, ids):
    """accept prob>thr, per-source caps (train-fold-derived), macro F0.5 over the given S1 ids."""
    idx = np.where(mask & (prob > thr))[0]
    d = pl.DataFrame({"s1": S1[idx], "e": EID[idx], "src": SRC[idx], "p": prob[idx]})
    d = d.with_columns(pl.col("p").rank("ordinal", descending=True).over(["s1", "src"]).alias("r"))
    d = d.filter(((pl.col("src") == "S2") & (pl.col("r") <= CAP["S2"])) | ((pl.col("src") == "S3") & (pl.col("r") <= CAP["S3"])))
    pred = {}
    for a, b in zip(d["s1"].to_list(), d["e"].to_list()):
        pred.setdefault(a, set()).add(b)
    return f05(pred, TRUTH, ids)


def best_thr(mask, prob, ids):
    grid = np.arange(0.20, 0.96, 0.025)
    sc = [score(mask, prob, th, ids) for th in grid]
    i = int(np.argmax(sc))
    return float(grid[i]), sc[i]


# ------------------------------------------------------------------ model factories
def fit_lgb(p, Xtr, ytr, wtr=None):
    params = dict(objective="binary", verbose=-1, num_threads=THREADS, bagging_freq=1, **{k: v for k, v in p.items() if k != "rounds"})
    return lgb.train(params, lgb.Dataset(Xtr, ytr, weight=wtr), p["rounds"])


def fit_xgb(p, Xtr, ytr, wtr=None):
    params = dict(objective="binary:logistic", tree_method="hist", nthread=THREADS, eval_metric="logloss",
                  **{k: v for k, v in p.items() if k != "rounds"})
    return xgb.train(params, xgb.DMatrix(Xtr, ytr, weight=wtr), p["rounds"])


def pred(kind, m, Xp):
    return m.predict(Xp) if kind == "lgb" else m.predict(xgb.DMatrix(Xp))


BASE_LGB = dict(learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.9,
                bagging_fraction=0.8, rounds=600)
BASE_XGB = dict(eta=0.05, max_depth=8, min_child_weight=5, subsample=0.8, colsample_bytree=0.9, rounds=600)


def oof_in_A(kind, p):
    """out-of-fold predictions for A rows (2 folds keyed by S1), zero elsewhere."""
    prob = np.zeros(len(y))
    for k in (0, 1):
        tr = (half == 0) & (inner != k); te = (half == 0) & (inner == k)
        m = (fit_lgb if kind == "lgb" else fit_xgb)(p, X[tr], y[tr], W[tr])
        prob[te] = pred(kind, m, X[te])
    return prob


def a_to_b(kind, p):
    """fit on all of A, threshold from A's OOF, score once on B. Returns (B F0.5, thr, B probs, A oof)."""
    oof = oof_in_A(kind, p)
    thr, a_sc = best_thr(half == 0, oof, IDS[0])
    m = (fit_lgb if kind == "lgb" else fit_xgb)(p, X[half == 0], y[half == 0], W[half == 0])
    pb = np.zeros(len(y)); pb[half == 1] = pred(kind, m, X[half == 1])
    return score(half == 1, pb, thr, IDS[1]), thr, a_sc, pb, oof


# ------------------------------------------------------------------ Optuna (on A only)
def objective(trial):
    p = dict(learning_rate=trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
             num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
             min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 20, 2000, log=True),
             feature_fraction=trial.suggest_float("feature_fraction", 0.5, 1.0),
             bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
             lambda_l2=trial.suggest_float("lambda_l2", 1e-3, 30, log=True),
             rounds=trial.suggest_int("rounds", 150, 1200, step=50))
    oof = oof_in_A("lgb", p)
    return best_thr(half == 0, oof, IDS[0])[1]


if __name__ == "__main__":
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    results = {}
    b_sc, thr, a_sc, pb_base, oof_base = a_to_b("lgb", BASE_LGB)
    results["baseline LGBM (current stage 2)"] = (b_sc, thr, a_sc); log("baseline", results)

    if not os.environ.get("FAST"):  # FAST=1 (deadline mode): baseline LGBM only; the baseline won on half B for v8 and v10
        if n_trials > 0:
            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False,
                           callbacks=[lambda s, tr: log(f"trial {tr.number} A-F0.5={tr.value:.5f} best={s.best_value:.5f}")])
            best_lgb = dict(study.best_params)
        else:  # reuse hyper-parameters found by an earlier Optuna study (chosen on half A only)
            best_lgb = json.load(open(os.environ["BEST_FROM"]))["lgb"]
            log("reusing Optuna params from", os.environ["BEST_FROM"], best_lgb)
        b_sc, thr, a_sc, pb_lgb, oof_lgb = a_to_b("lgb", best_lgb)
        results["Optuna-tuned LGBM"] = (b_sc, thr, a_sc); log("tuned", results["Optuna-tuned LGBM"])

        b_sc, thr, a_sc, pb_xgb, oof_xgb = a_to_b("xgb", BASE_XGB)
        results["XGBoost"] = (b_sc, thr, a_sc); log("xgb", results["XGBoost"])

        # ensemble = probability average (weights chosen on A's OOF only)
        best_w, best_a = 0.5, -1
        for w in np.arange(0, 1.01, 0.1):
            th, sc = best_thr(half == 0, w * oof_lgb + (1 - w) * oof_xgb, IDS[0])
            if sc > best_a:
                best_w, best_a, ens_thr = float(w), sc, th
        ens_b = score(half == 1, best_w * pb_lgb + (1 - best_w) * pb_xgb, ens_thr, IDS[1])
        results[f"Ensemble LGBM {best_w:.1f} + XGB {1 - best_w:.1f}"] = (ens_b, ens_thr, best_a)

    log("=" * 60)
    for k, (b, th, a) in results.items():
        log(f"{k:40s}  B(untouched) F0.5={b:.5f}   A(CV) F0.5={a:.5f}   thr={th:.3f}")
    winner = max(results, key=lambda k: results[k][0])
    log("WINNER on untouched half B:", winner)

    # refit winner on A+B for the test set; threshold from full 2-fold OOF (keyed by S1)
    full = np.ones(len(y), bool)
    def oof_full(kind, p):
        pr = np.zeros(len(y))
        for k in (0, 1):
            m = (fit_lgb if kind == "lgb" else fit_xgb)(p, X[inner != k], y[inner != k], W[inner != k])
            pr[inner == k] = pred(kind, m, X[inner == k])
        return pr
    ids_all = IDS[0] + IDS[1]
    if winner.startswith("Ensemble"):
        pr = best_w * oof_full("lgb", best_lgb) + (1 - best_w) * oof_full("xgb", BASE_XGB)
        cfg = {"kind": "ensemble", "w_lgb": best_w, "lgb": best_lgb, "xgb": BASE_XGB}
    elif winner == "XGBoost":
        pr = oof_full("xgb", BASE_XGB); cfg = {"kind": "xgb", "xgb": BASE_XGB}
    else:
        p = best_lgb if winner.startswith("Optuna") else BASE_LGB
        pr = oof_full("lgb", p); cfg = {"kind": "lgb", "lgb": p}
    thr, sc = best_thr(full, pr, ids_all)
    if cfg["kind"] in ("lgb", "ensemble"):
        fit_lgb(cfg["lgb"], X, y, W).save_model(os.environ.get("MODEL_PREFIX", f"{DATA}/model2") + "_lgb.txt")
    if cfg["kind"] in ("xgb", "ensemble"):
        fit_xgb(cfg["xgb"], X, y, W).save_model(os.environ.get("MODEL_PREFIX", f"{DATA}/model2") + "_xgb.json")
    cfg.update(thr=thr, cv_f05_full=sc, feats=FEATS,
               report={k: {"B_f05": v[0], "A_cv_f05": v[2], "thr": v[1]} for k, v in results.items()})
    json.dump(cfg, open(os.environ.get("FINAL", f"{DATA}/stage2_final.json"), "w"), indent=1)
    log("saved final stage-2 config:", cfg["kind"], "thr", round(thr, 3), "full-CV F0.5", round(sc, 5))
