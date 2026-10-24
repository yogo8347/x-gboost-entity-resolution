#!/usr/bin/env bash
# v10: v8 stage-2 table + Kaggle cross-encoder probability. Waits for data/ce_val.parquet and data/ce_test.parquet.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 TUNE_THREADS=8
PY="python -W ignore"
until [ -f data/ce_val.parquet ] && [ -f data/ce_test.parquet ] && ! grep -q "V9 RUNNING" /dev/null; do sleep 30; done
$PY src/add_ce.py data/stage2v8_train.parquet data/ce_val.parquet data/stage2v10_train.parquet || exit 1
$PY src/add_ce.py data/frpatch/stage2v8_test.parquet data/ce_test.parquet data/frpatch/stage2v10_test.parquet || exit 1
EXCL='k_all|k_src|rank_src|rank_all|t1_.*_gap2?|t1_comb2?_rank|t1_q_ncand|t1_bscore.*|t1_brank'
TABLE=data/stage2v10_train.parquet FINAL=data/stage2v10_final.json MODEL_PREFIX=data/model2v10 \
  BEST_FROM=data/v3_old/stage2v3_final.json EXCLUDE="$EXCL" $PY src/tune_stage2.py 0 || exit 1
$PY src/drift_check.py data/stage2v10_final.json data/stage2v10_train.parquet data/frpatch/stage2v10_test.parquet 0.10
echo "drift gate exit $?"
$PY src/apply_table.py data/stage2v10_final.json data/model2v10 data/frpatch/stage2v10_test.parquet output_v10 0.85 0.92 || exit 1
for d in output_v10 output_v10_t85 output_v10_t92; do python utils/validate_submission.py --matching $d/matching_results.tsv --candidate $d/candidate_pairs.tsv --test-dir dataset/test | tail -1; done
echo "V10 DONE"
