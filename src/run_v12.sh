#!/usr/bin/env bash
# v12: v10 + cross-encoder v2 (trained on stage-1 training-fold pairs). Waits for data/ce2_val.parquet + data/ce2_test.parquet.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 TUNE_THREADS=8
PY="python -W ignore -u"
until [ -f data/ce2_val.parquet ] && [ -f data/ce2_test.parquet ]; do sleep 30; done
$PY src/add_ce2.py data/stage2v10_train.parquet data/ce2_val.parquet data/stage2v12_train.parquet || exit 1
$PY src/add_ce2.py data/frpatch/stage2v10_test.parquet data/ce2_test.parquet data/frpatch/stage2v12_test.parquet || exit 1
EXCL='k_all|k_src|rank_src|rank_all|t1_.*_gap2?|t1_comb2?_rank|t1_q_ncand|t1_bscore.*|t1_brank'
TABLE=data/stage2v12_train.parquet FINAL=data/stage2v12_final.json MODEL_PREFIX=data/model2v12 \
  BEST_FROM=data/v3_old/stage2v3_final.json EXCLUDE="$EXCL" FAST=1 $PY src/tune_stage2.py 0 || exit 1
$PY src/drift_check.py data/stage2v12_final.json data/stage2v12_train.parquet data/frpatch/stage2v12_test.parquet 0.10
echo "drift gate exit $?"
$PY src/apply_table.py data/stage2v12_final.json data/model2v12 data/frpatch/stage2v12_test.parquet output_v12 0.85 0.92 || exit 1
for d in output_v12 output_v12_t85 output_v12_t92; do python utils/validate_submission.py --matching $d/matching_results.tsv --candidate $d/candidate_pairs.tsv --test-dir dataset/test | tail -1; done
echo "V12 DONE"
