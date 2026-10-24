#!/usr/bin/env bash
# v9: v8 + covariate-shift weights (adversarial validation). Waits for v8.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 TUNE_THREADS=8
PY="python -W ignore"
until grep -q "V8 DONE" data/v8.log; do sleep 30; done
$PY src/shift_weights.py data/stage2v8_final.json data/stage2v8_train.parquet data/frpatch/stage2v8_test.parquet data/shift_w_v8.parquet || exit 1
EXCL='k_all|k_src|rank_src|rank_all|t1_.*_gap2?|t1_comb2?_rank|t1_q_ncand|t1_bscore.*|t1_brank'
SHIFT_WEIGHTS=data/shift_w_v8.parquet TABLE=data/stage2v8_train.parquet FINAL=data/stage2v9_final.json MODEL_PREFIX=data/model2v9 \
  BEST_FROM=data/v3_old/stage2v3_final.json EXCLUDE="$EXCL" $PY src/tune_stage2.py 0 || exit 1
$PY src/drift_check.py data/stage2v9_final.json data/stage2v8_train.parquet data/frpatch/stage2v8_test.parquet 0.10
echo "drift gate exit $?"
$PY src/apply_table.py data/stage2v9_final.json data/model2v9 data/frpatch/stage2v8_test.parquet output_v9 0.85 0.92 || exit 1
for d in output_v9 output_v9_t85 output_v9_t92; do python utils/validate_submission.py --matching $d/matching_results.tsv --candidate $d/candidate_pairs.tsv --test-dir dataset/test | tail -1; done
echo "V9 DONE"
