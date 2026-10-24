#!/usr/bin/env bash
# v8: v7 + translated comparison (learned dictionary). Waits for v7.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 TUNE_THREADS=8
PY="python -W ignore"
until grep -q "V7 DONE" data/v7.log; do sleep 30; done
$PY src/translit_markers.py data/stage2v7_train.parquet data/stage2v8_train.parquet data train                      || exit 1
$PY src/translit_markers.py data/frpatch/stage2v7_test.parquet data/frpatch/stage2v8_test.parquet data/frpatch test || exit 1
EXCL='k_all|k_src|rank_src|rank_all|t1_.*_gap2?|t1_comb2?_rank|t1_q_ncand|t1_bscore.*|t1_brank'
TABLE=data/stage2v8_train.parquet FINAL=data/stage2v8_final.json MODEL_PREFIX=data/model2v8 \
  BEST_FROM=data/v3_old/stage2v3_final.json EXCLUDE="$EXCL" $PY src/tune_stage2.py 0                          || exit 1
$PY src/drift_check.py data/stage2v8_final.json data/stage2v8_train.parquet data/frpatch/stage2v8_test.parquet 0.10
echo "drift gate exit $?"
$PY src/apply_table.py data/stage2v8_final.json data/model2v8 data/frpatch/stage2v8_test.parquet output_v8 0.85 0.92 || exit 1
for d in output_v8 output_v8_t85 output_v8_t92; do python utils/validate_submission.py --matching $d/matching_results.tsv --candidate $d/candidate_pairs.tsv --test-dir dataset/test | tail -1; done
echo "V8 DONE"
