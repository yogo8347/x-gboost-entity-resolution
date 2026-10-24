#!/usr/bin/env bash
# Resumable end-to-end driver: each stage is skipped if its output already exists.
set -e
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
PY="python -W ignore"
DATASET=${DATASET:-dataset}

for split in train test; do
  for k in 1 2 3; do [ -f data/${split}_s${k}.parquet ] || $PY src/preprocess.py "$DATASET" $split; done
done
[ -f data/train_cands.parquet ]  || $PY src/pipeline.py block train      # candidate generation (train)
[ -f data/model.txt ]            || $PY src/pipeline.py train            # stage-1 pair model
[ -f data/train_scored.parquet ] || $PY src/pipeline.py infer train      # out-of-sample scores, held-out S1 fold
[ -f data/best.json ]            || $PY src/pipeline.py evaluate         # stage-1-only F0.5 (reference)
[ -f data/best2.json ]           || $PY src/stage2.py cv                 # stage-2 cluster model, 2-fold CV F0.5
[ -f data/model2.txt ]           || $PY src/stage2.py fit
[ -f data/test_cands.parquet ]   || $PY src/pipeline.py block test       # candidate generation (test)
[ -f data/test_scored.parquet ]  || $PY src/pipeline.py infer test
$PY src/stage2.py apply                                                  # -> output/*.tsv
python utils/validate_submission.py --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv --test-dir "$DATASET/test"
echo "ALL DONE"
