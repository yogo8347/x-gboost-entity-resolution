# Business Entity Resolution — reproduction guide

Pipeline: **normalise → block (inverted index) → pair features → LightGBM → 1-to-1 assignment**.
CPU only, no external data or APIs. Tested on Windows 11, Python 3.11, 16 GB RAM, 8 cores.

## Setup
```bash
pip install -r requirements.txt
```
Place the challenge data so that `dataset/train/*.tsv` and `dataset/test/*.tsv` exist
next to `src/` (or set `DATASET=/path/to/dataset`).

## Run everything
```bash
bash src/run_all.sh
```
Each stage writes its output to `data/` and is skipped when that output already exists,
so the script can be re-run after an interruption.

| Stage | Command | Output | Time (8 cores) |
|---|---|---|---|
| Normalise all records | `python src/preprocess.py dataset train` / `... test` | `data/<split>_s{1,2,3}.parquet` | ~5 min / source |
| Blocking | `python src/pipeline.py block train` | `data/train_cands.parquet` | ~60 min |
| Train matcher | `python src/pipeline.py train` | `data/model.txt` | ~10 min |
| Score held-out fold | `python src/pipeline.py infer train` | `data/train_scored.parquet` | ~20 min |
| Validate + tune threshold | `python src/pipeline.py evaluate` | `data/best.json` | ~2 min |
| Blocking (test) | `python src/pipeline.py block test` | `data/test_cands.parquet` | ~50 min |
| Score test | `python src/pipeline.py infer test` | `data/test_scored.parquet` | ~40 min |
| Write submission | `python src/pipeline.py submit` | `output/matching_results.tsv`, `output/candidate_pairs.tsv` | ~2 min |

Then check the files:
```bash
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

## Source files (`src/`)
- `normalize.py`: transliteration (anyascii), legal suffix / DBA handling, address abbreviations for US, India and France
- `preprocess.py`: parallel normalisation of every record, cached to parquet
- `blocking.py`: IDF-weighted multi-key inverted index with two-stage rescoring
- `features.py`: 69 stage-1 pair features, none tied to a country
- `pipeline.py`: stages block / train / infer / evaluate / submit; `select_training_records()` is the leakage guard
- `stage2.py`: stage-2 cluster-aware verifier (`cv` / `fit` / `apply`), writes both output files
- `stage2v3.py`: richer stage-2 table (full stage-1 features of best + runner-up, coherence with every cluster member)
- `tune_stage2.py`: leak-free Optuna tuning + LightGBM/XGBoost ensemble, scored once on an untouched half of the held-out fold
- `metric.py`: macro F0.5 per Source-1 entity (singletons included)
- `run_all.sh`: resumable driver that runs every stage
- `make_zip.py`: builds the submission archive
- `dev_block.py`, `eda.py`, `eda2.py`: exploration and blocking-recall experiments (not needed to reproduce)

## Leakage tests
```bash
python -m pytest tests/test_leakage.py -v
```
22 checks: the 80/20 hash fold on S1 entities, training records disjoint from every record that can land on a
held-out S1, no training pair touching a held-out S1, labels only from ground truth, no feature or test-time
code reading labels, stage-2 folds keyed by S1 entity, caps from the training fold only.

## Licences
LightGBM (MIT), XGBoost (Apache-2.0), Optuna (MIT), polars (MIT), rapidfuzz (MIT), anyascii (ISC), numpy/pyarrow (BSD/Apache-2.0).
The model is a gradient-boosted tree ensemble with far fewer than 8B parameters.

## Reproducing the final submission (v10 / v12)
The table above builds the v2 baseline. The final file adds a stage-2 verifier, marker features and fine-tuned
cross-encoders. After `run_all.sh`:
```bash
python src/france_patch.py                      # France address re-normalisation (test only)
bash src/run_v5.sh; bash src/run_v6.sh; bash src/run_v7.sh; bash src/run_v8.sh   # stage-2 table + decoy/word/clean/translit markers
```
Cross-encoders (GPU; we used free Kaggle T4s, notebooks in `cloud_package*/`):
1. v10: `src/cross_encoder.py prep` writes uncertain best pairs (stage-1 p1 in [0.02, 0.995]); `cloud_package/xgboost_cross_encoder.py`
   fine-tunes `BAAI/bge-reranker-v2-m3` (Apache-2.0) in 2 folds equal to stage-2 halves A/B → `data/ce_val.parquet`, `data/ce_test.parquet`.
   Then `bash src/run_v10.sh` (adds the score, tunes stage 2 on half A, reports half B, applies to test).
2. v12: `python src/ce2_prep.py` (training pairs from stage-1 training records only) → `cloud_package2/xgboost_ce2.py`
   → `data/ce2_val.parquet`, `data/ce2_test.parquet` → `bash src/run_v12.sh`.
Checks: `python -m pytest tests/test_leakage.py` (36 leakage tests), `python src/overfit_check.py` (overfitting audit),
`python src/drift_check.py …` + `python src/drift_ce_check.py` (validation vs test drift), `utils/validate_submission.py`.
Experiments that were measured and rejected (kept for transparency): `v11_experiments.py`, `rank_rule.py`, LLM judge
(`cloud_package3/`), second architecture (`cloud_package4/`, `add_ce3.py`).
