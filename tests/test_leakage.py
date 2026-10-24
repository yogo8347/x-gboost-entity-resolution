"""Strict data-leakage tests for the entity-resolution pipeline.

Run from student_resource/:   python -m pytest tests/test_leakage.py -v -s

The validation design being tested:
  * S1 entities are split by a deterministic hash: 80% TRAIN fold, 20% HELD-OUT (validation) fold.
  * An S2/S3 record may be used to TRAIN stage 1 only if its true S1 is in the train fold
    (unmatched records: its own id hashes to the train fold) AND none of its blocking
    candidates is a held-out S1.
  * Every record that has a held-out S1 among its candidates is SCORED out-of-sample and used
    for F0.5 / threshold / stage-2 cross-validation.
  * The test split never touches ground truth.
Any violation of these rules is a leak, and a test below fails.
"""
import glob
import inspect
import json
import os
import re
import sys

import polars as pl
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)

import pipeline  # noqa: E402
from pipeline import eval_fold, load_gt, select_training_records, val_fold  # noqa: E402

DATA = "data"
need = pytest.mark.skipif


# ----------------------------------------------------------------------------- fixtures (lean)
@pytest.fixture(scope="session")
def s1():
    df = pl.read_parquet(f"{DATA}/train_s1.parquet", columns=["entity_id"]).with_row_index("rid")
    return df.with_columns(pl.col("rid").cast(pl.UInt32), val_fold(pl.col("entity_id")).alias("is_val"))


@pytest.fixture(scope="session")
def gt():
    return load_gt()


@pytest.fixture(scope="session")
def q(gt):
    """S2+S3 records in exactly the order pipeline.load() gives them (qid = row index)."""
    df = pl.concat([pl.read_parquet(f"{DATA}/train_s{k}.parquet", columns=["entity_id"]) for k in (2, 3)])
    df = df.with_row_index("qid").with_columns(pl.col("qid").cast(pl.UInt32))
    return df.join(gt, on="entity_id", how="left")


@pytest.fixture(scope="session")
def cands():
    return pl.read_parquet(f"{DATA}/train_cands.parquet", columns=["qid", "rid"])


@pytest.fixture(scope="session")
def train_pairs():
    return pl.read_parquet(f"{DATA}/train_feats.parquet", columns=["qid", "rid", "label"])


@pytest.fixture(scope="session")
def regenerated(q, s1, cands):
    tq, info = select_training_records(q, s1, cands)
    return tq, info


@pytest.fixture(scope="session")
def eval_records(cands, s1):
    """records that have at least one held-out S1 among their candidates (scored out-of-sample)."""
    return cands.join(s1.filter(pl.col("is_val")).select("rid"), on="rid", how="semi").select("qid").unique()


@pytest.fixture(scope="session")
def summary():
    out = {}
    yield out
    json.dump(out, open(f"{DATA}/split_summary.json", "w"), indent=1)
    print("\nSPLIT SUMMARY:", json.dumps(out, indent=1))


# ----------------------------------------------------------------------------- 1. the fold itself
def test_fold_is_20_percent_and_deterministic(s1, summary):
    frac = s1["is_val"].mean()
    again = val_fold(s1["entity_id"])
    summary.update(s1_total=s1.height, s1_train_fold=int((~s1["is_val"]).sum()), s1_heldout=int(s1["is_val"].sum()))
    assert 0.195 < frac < 0.205, frac
    assert (again == s1["is_val"]).all(), "fold assignment is not deterministic"


def test_evaluation_uses_the_full_heldout_fold(s1):
    assert (eval_fold(s1["entity_id"]) == s1["is_val"]).all(), "evaluation fold != full 20% held-out fold"


def test_every_s1_in_exactly_one_fold(s1):
    assert s1["entity_id"].n_unique() == s1.height, "duplicate S1 ids"
    both = s1.group_by("entity_id").agg(pl.col("is_val").n_unique().alias("k")).filter(pl.col("k") > 1)
    assert both.height == 0


# ----------------------------------------------------------------------------- 2. stage-1 training data
def test_saved_training_set_is_exactly_what_the_guard_selects(regenerated, train_pairs, cands, summary):
    """the model on disk was trained on precisely the records the leakage guard allows
    (records with zero blocking candidates cannot produce training pairs, so they are the only allowed gap)."""
    tq, info = regenerated
    saved = train_pairs.select("qid").unique()
    has_cand = cands.select("qid").unique()
    missing = tq.select("qid").join(saved, on="qid", how="anti")
    summary.update(eligible_train_records=info["eligible"], sampled_pct=info["frac"],
                   training_records=saved.height, training_pairs=train_pairs.height,
                   guard_selected_without_candidates=missing.height)
    assert saved.join(tq.select("qid"), on="qid", how="anti").height == 0, "model saw records outside the guard"
    assert missing.join(has_cand, on="qid", how="semi").height == 0, \
        "a guard-selected record WITH candidates is missing from the training pairs"


def test_no_training_pair_involves_a_heldout_s1(train_pairs, s1):
    bad = train_pairs.join(s1.filter(pl.col("is_val")).select("rid"), on="rid", how="semi")
    assert bad.height == 0, f"{bad.height} training pairs reference held-out S1 entities"


def test_no_training_record_belongs_to_a_heldout_s1(regenerated, s1):
    tq, _ = regenerated
    held = s1.filter(pl.col("is_val")).select(pl.col("entity_id").alias("s1_id"))
    bad = tq.drop_nulls("s1_id").join(held, on="s1_id", how="semi")
    assert bad.height == 0, f"{bad.height} training records are true matches of held-out S1"


def test_unmatched_training_records_hash_to_train_fold(regenerated):
    tq, _ = regenerated
    un = tq.filter(pl.col("s1_id").is_null())
    assert un.height > 0, "no distractors in training - model could not learn to reject"
    assert not val_fold(un["entity_id"]).any(), "a held-out distractor was used for training"


def test_training_and_evaluation_records_are_disjoint(regenerated, eval_records, q, summary):
    tq, _ = regenerated
    overlap = tq.select("qid").join(eval_records, on="qid", how="semi").height
    summary.update(eval_records=eval_records.height, s2s3_total=q.height,
                   unused_records=q.height - eval_records.height - tq.height)
    assert overlap == 0, f"{overlap} records are both trained on and evaluated"


def test_training_labels_match_ground_truth(train_pairs, q, s1):
    t = (train_pairs.join(q.select("qid", "s1_id"), on="qid")
         .join(s1.select("rid", pl.col("entity_id").alias("cand")), on="rid"))
    expect = (t["cand"] == t["s1_id"]).fill_null(False).cast(pl.Int8)
    assert (expect == t["label"]).all(), "label column disagrees with ground truth"


def test_training_pairs_come_from_blocking_only(train_pairs, cands):
    """no pair was injected from the ground truth (e.g. adding missed positives)."""
    extra = train_pairs.select("qid", "rid").join(cands, on=["qid", "rid"], how="anti")
    assert extra.height == 0, f"{extra.height} training pairs are not blocking candidates"


# ----------------------------------------------------------------------------- 3. features / model
def _src(obj):
    return inspect.getsource(obj)


def test_features_never_read_labels():
    import features, blocking
    for mod in (features, blocking):
        src = _src(mod)
        for bad in ("ground_truth", "load_gt", "s1_id", "matched_entity_ids"):
            assert bad not in src, f"{mod.__name__} references {bad!r}"


def test_model_features_have_no_label_columns():
    import lightgbm as lgb
    names = lgb.Booster(model_file=f"{DATA}/model.txt").feature_name()
    for bad in ("label", "s1_id", "true", "cand", "entity_id", "qid", "rid"):
        assert bad not in names, f"model uses forbidden column {bad!r}"


def test_country_is_not_a_feature():
    import lightgbm as lgb
    names = lgb.Booster(model_file=f"{DATA}/model.txt").feature_name()
    assert not any("country" in n or n in ("us", "india", "france") for n in names)


# ----------------------------------------------------------------------------- 4. test split never sees labels
@pytest.mark.parametrize("fn", ["block", "infer", "submit"])
def test_test_path_never_touches_ground_truth(fn):
    src = _src(getattr(pipeline, fn))
    assert "load_gt" not in src and "ground_truth" not in src, f"pipeline.{fn} reads labels"


def test_stage2_apply_never_touches_ground_truth():
    import stage2
    for fn in (stage2.apply, stage2.cluster_feats, stage2.decide):
        src = _src(fn)
        assert "load_gt" not in src and "val_data" not in src and "label" not in src, fn.__name__


def test_no_entity_ids_shared_between_train_and_test():
    tr = pl.concat([pl.read_parquet(f"{DATA}/train_s{k}.parquet", columns=["entity_id"]) for k in (1, 2, 3)])
    te = pl.concat([pl.read_parquet(f"{DATA}/test_s{k}.parquet", columns=["entity_id"]) for k in (1, 2, 3)])
    shared = tr.join(te, on="entity_id", how="semi").height
    assert shared == 0, f"{shared} entity ids appear in both train and test"


# ----------------------------------------------------------------------------- 5. stage 2
def test_stage2_caps_come_from_train_fold_only(gt, s1):
    import stage2
    tr = s1.filter(~pl.col("is_val")).select(pl.col("entity_id").alias("s1_id"))
    g = (gt.join(tr, on="s1_id", how="semi").with_columns(pl.col("entity_id").str.slice(0, 2).alias("src"))
         .group_by("s1_id", "src").len().group_by("src").agg(pl.col("len").max()))
    expect = {r[0]: int(r[1]) for r in g.iter_rows()}
    assert stage2.CAP == expect
    assert "~val_fold" in _src(stage2._caps), "caps must be computed on the train fold only"


def test_stage2_cv_folds_split_by_s1_entity():
    import stage2
    src = _src(stage2.cv)
    assert re.search(r't\["s1_id"\]\.hash\(seed=\d+\) % 2', src), "stage-2 CV folds must be keyed on the S1 entity"
    assert "prob[fold == k] = m.predict(X[fold == k])" in src, "stage-2 threshold must come from out-of-fold predictions"


def test_stage2_training_rows_are_heldout_s1_only():
    import stage2
    src = _src(stage2.val_data)
    assert "eval_fold" in src and 'join(s1v.select("rid"), on="rid", how="semi")' in src


# ----------------------------------------------------------------------------- 6. scored (out-of-sample) table
def _scored_qids():
    if os.path.exists(f"{DATA}/train_scored.parquet"):
        return pl.read_parquet(f"{DATA}/train_scored.parquet", columns=["qid"]).unique()
    parts = glob.glob(f"{DATA}/train_scored_parts/*.parquet")
    if not parts:
        pytest.skip("no scored parts yet")
    return pl.read_parquet(parts, columns=["qid"]).unique()


def test_scored_records_were_never_trained_on(regenerated):
    tq, _ = regenerated
    overlap = _scored_qids().join(tq.select("qid"), on="qid", how="semi").height
    assert overlap == 0, f"{overlap} scored (evaluation) records were in the training set"


# ----------------------------------------------------------------------------- 7. submission files
@pytest.mark.parametrize("out", ["output", "output_v3"])
def test_submission_only_uses_test_ids_and_matches_subset_of_candidates(out):
    if not os.path.exists(f"{out}/matching_results.tsv"):
        pytest.skip(f"{out} not written yet")
    rd = lambda f: pl.read_csv(f, separator="\t", infer_schema=False).fill_null("")
    m, c = rd(f"{out}/matching_results.tsv"), rd(f"{out}/candidate_pairs.tsv")
    s1t = pl.read_parquet(f"{DATA}/test_s1.parquet", columns=["entity_id"])["entity_id"]
    assert m.height == s1t.len() and set(m["source1_entity_id"]) == set(s1t)
    ids = set(pl.concat([pl.read_parquet(f"{DATA}/test_s{k}.parquet", columns=["entity_id"]) for k in (2, 3)])["entity_id"])
    cand = {a: set(b.split(",")) - {""} for a, b in c.iter_rows()}
    for a, b in m.iter_rows():
        got = set(b.split(",")) - {""}
        assert got <= ids and got <= cand.get(a, set()), a


# ----------------------------------------------------------------------------- 8. stage-2 v3 + tuning / ensemble
V3 = f"{DATA}/stage2v3_train.parquet"
needv3 = pytest.mark.skipif(not os.path.exists(V3), reason="v3 table not built")


@needv3
def test_v3_rows_are_heldout_s1_and_never_trained_on(s1, regenerated):
    t = pl.read_parquet(V3, columns=["qid", "rid"])
    held = s1.filter(pl.col("is_val")).select("rid")
    assert t.join(held, on="rid", how="anti").height == 0, "v3 table contains a row whose S1 is in the training fold"
    tq, _ = regenerated
    assert t.join(tq.select("qid"), on="qid", how="semi").height == 0, "v3 table contains stage-1 training records"


@needv3
def test_v3_labels_match_ground_truth(gt):
    t = pl.read_parquet(V3, columns=["entity_id", "s1_id", "label"])
    chk = t.join(gt.rename({"s1_id": "true"}), on="entity_id", how="left")
    expect = (chk["true"] == chk["s1_id"]).fill_null(False).cast(pl.Int8)
    assert (expect == chk["label"]).all()


@pytest.mark.skipif(not os.path.exists(f"{DATA}/stage2v3_final.json"), reason="v3 not tuned yet")
def test_v3_model_inputs_have_no_label_or_id_columns():
    feats = json.load(open(f"{DATA}/stage2v3_final.json"))["feats"]
    for bad in ("label", "s1_id", "entity_id", "true", "qid", "rid", "src", "country"):
        assert bad not in feats, f"v3 model uses forbidden column {bad!r}"


def test_v3_test_path_never_touches_ground_truth():
    import stage2v3
    assert "load_gt" not in _src(stage2v3.apply) and "label" not in _src(stage2v3.apply)
    bucket = _src(stage2v3._build_bucket)
    assert bucket.count("load_gt") == 1, "labels may only be read once, in the train branch"
    assert re.search(r'if split == "train":\s+gt = load_gt\(\)', bucket), \
        "labels may only be attached inside the train branch"


def test_tuning_reports_on_an_untouched_half():
    import tune_stage2
    src = _src(tune_stage2)
    assert 'half = (t["s1_id"].hash(seed=77) % 2)' in src, "A/B halves must be keyed by S1 entity"
    assert 'inner = (t["s1_id"].hash(seed=21) % 2)' in src, "inner CV must be keyed by S1 entity"
    assert "oof_in_A" in _src(tune_stage2.objective) and "half == 1" not in _src(tune_stage2.objective),         "Optuna objective must only see half A"
    assert "max(results, key=lambda k: results[k][0])" in src, "winner must be chosen on half B"


# ---------------------------------------------------------------- v10: cross-encoder feature
def test_v10_cross_encoder_folds_are_the_stage2_halves():
    """B's cross-encoder score must come from a model trained on half A only (and vice versa):
    the CE fold of every validation pair equals the stage-2 half (S1 hash, seed 77)."""
    v = pl.read_parquet("cloud_package/val_pairs.parquet", columns=["qid", "fold"])
    t = pl.read_parquet("data/stage2v10_train.parquet", columns=["qid", "s1_id"])
    j = v.join(t, on="qid")
    assert j.height == v.height
    assert j.select((pl.col("fold") == (pl.col("s1_id").hash(seed=77) % 2)).all()).item()


def test_v10_cross_encoder_scored_only_heldout_s1_pairs():
    """CE training pairs are best-candidate pairs of the held-out S1 fold only (never stage-1 training records)."""
    v = pl.read_parquet("cloud_package/val_pairs.parquet", columns=["qid"])
    t = pl.read_parquet("data/stage2v10_train.parquet", columns=["qid", "s1_id"]).join(v, on="qid", how="semi")
    assert t.select(val_fold(pl.col("s1_id")).all()).item()


def test_v10_test_pairs_carry_no_labels():
    cols = pl.read_parquet_schema("cloud_package/test_pairs.parquet")
    assert "label" not in cols and "fold" not in cols


def test_v10_model_inputs_have_no_label_or_id_columns():
    feats = json.load(open("data/stage2v10_final.json"))["feats"]
    bad = [f for f in feats if f in ("label", "qid", "rid", "entity_id", "s1_id", "fold") or "truth" in f]
    assert not bad, bad


# ---------------------------------------------------------------- v12 / final: transformer v2 training pairs
def test_ce2_training_pairs_never_touch_a_heldout_s1(s1):
    """ce2 is trained on stage-1 TRAINING records only: no pair may involve a held-out S1 entity (candidate side),
    and no query record whose true S1 is held out."""
    tp = pl.read_parquet("cloud_package2/train_pairs.parquet", columns=["qid", "rid"])
    s1r = pl.read_parquet("data/train_s1.parquet", columns=["entity_id"]).with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
    cand = tp.join(s1r, on="rid")
    assert cand.height == tp.height
    assert not cand.select(val_fold(pl.col("entity_id")).any()).item(), "a held-out S1 is in the ce2 training pairs"
    q = pl.concat([pl.read_parquet(f"data/train_s{k}.parquet", columns=["entity_id"]) for k in (2, 3)]) \
          .with_row_index("qid").with_columns(pl.col("qid").cast(pl.UInt32))
    rec = tp.select("qid").unique().join(q, on="qid").join(load_gt(), on="entity_id", how="left")
    held = s1r.filter(val_fold(pl.col("entity_id")))["entity_id"]
    assert not rec["s1_id"].is_in(held.implode()).any(), "a record of a held-out S1 is in the ce2 training pairs"


def test_ce2_training_records_disjoint_from_validation_records():
    tp = pl.read_parquet("cloud_package2/train_pairs.parquet", columns=["qid"]).unique()
    vp = pl.read_parquet("cloud_package2/val_pairs.parquet", columns=["qid"]).unique()
    assert tp.join(vp, on="qid", how="semi").height == 0


def test_llm_and_test_inputs_carry_no_labels():
    for f in ("cloud_package3/llm_test.parquet", "cloud_package2/test_pairs.parquet"):
        assert "label" not in pl.read_parquet_schema(f)
