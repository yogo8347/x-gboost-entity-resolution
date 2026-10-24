#!/usr/bin/env bash
# v7: v6 + clean-comparison markers (vocabulary audit) + French decoy words. Waits for v6.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 TUNE_THREADS=8
PY="python -W ignore"
until grep -q "V6 DONE" data/v6.log; do sleep 30; done
$PY src/clean_markers.py data/stage2v6_train.parquet data/stage2v7_train.parquet data train                      || exit 1
$PY src/clean_markers.py data/frpatch/stage2v6_test.parquet data/frpatch/stage2v7_test.parquet data/frpatch test || exit 1
EXCL='k_all|k_src|rank_src|rank_all|t1_.*_gap2?|t1_comb2?_rank|t1_q_ncand|t1_bscore.*|t1_brank'
TABLE=data/stage2v7_train.parquet FINAL=data/stage2v7_final.json MODEL_PREFIX=data/model2v7 \
  BEST_FROM=data/v3_old/stage2v3_final.json EXCLUDE="$EXCL" $PY src/tune_stage2.py 0                          || exit 1
$PY src/drift_check.py data/stage2v7_final.json data/stage2v7_train.parquet data/frpatch/stage2v7_test.parquet 0.10
echo "drift gate exit $?"
$PY src/apply_table.py data/stage2v7_final.json data/model2v7 data/frpatch/stage2v7_test.parquet output_v7 0.85 0.92 || exit 1
for d in output_v7 output_v7_t85 output_v7_t92; do python utils/validate_submission.py --matching $d/matching_results.tsv --candidate $d/candidate_pairs.tsv --test-dir dataset/test | tail -1; done
echo "V7 DONE"
