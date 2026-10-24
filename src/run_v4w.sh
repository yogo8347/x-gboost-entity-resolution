#!/usr/bin/env bash
# v4w: v4 features + distractor rows weighted x1.6 in training (prior-shift correction). Runs after v4.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 TUNE_THREADS=8
PY="python -W ignore"
until grep -q "V4 DONE" data/v4.log; do sleep 30; done
EXCL='k_all|k_src|rank_src|rank_all|t1_.*_gap2?|t1_comb2?_rank|t1_q_ncand|t1_bscore.*|t1_brank'
DIST_WEIGHT=1.6 TABLE=data/stage2v3_train.parquet FINAL=data/stage2v4w_final.json MODEL_PREFIX=data/model2v4w \
  BEST_FROM=data/v3_old/stage2v3_final.json EXCLUDE="$EXCL" $PY src/tune_stage2.py 0 || exit 1
$PY src/drift_check.py data/stage2v4w_final.json data/stage2v3_train.parquet data/stage2v3_test.parquet 0.10
echo "drift gate exit $?"
$PY -c "import sys; sys.path.insert(0,'src'); import stage2v3; stage2v3.apply('data/stage2v4w_final.json','data/model2v4w','output_v4w')" || exit 1
python utils/validate_submission.py --matching output_v4w/matching_results.tsv --candidate output_v4w/candidate_pairs.tsv --test-dir dataset/test --check-ids
echo "V4W DONE"
