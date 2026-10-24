#!/usr/bin/env bash
# Drift-robust stage 2 (v4): confident-only cluster statistics, drift-sensitive inputs excluded, drift gate.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 TUNE_THREADS=8
PY="python -W ignore"
EXCL='k_all|k_src|rank_src|rank_all|t1_.*_gap2?|t1_comb2?_rank|t1_q_ncand|t1_bscore.*|t1_brank'
$PY src/stage2v3.py build train                                                  || exit 1
$PY src/stage2v3.py build test                                                   || exit 1
TABLE=data/stage2v3_train.parquet FINAL=data/stage2v4_final.json MODEL_PREFIX=data/model2v4 \
  BEST_FROM=data/v3_old/stage2v3_final.json EXCLUDE="$EXCL" $PY src/tune_stage2.py 0 || exit 1
$PY src/drift_check.py data/stage2v4_final.json data/stage2v3_train.parquet data/stage2v3_test.parquet 0.10
echo "drift gate exit $?"
$PY -c "import sys; sys.path.insert(0,'src'); import stage2v3; stage2v3.apply('data/stage2v4_final.json','data/model2v4','output_v4')" || exit 1
python utils/validate_submission.py --matching output_v4/matching_results.tsv --candidate output_v4/candidate_pairs.tsv --test-dir dataset/test --check-ids
echo "V4 DONE"
