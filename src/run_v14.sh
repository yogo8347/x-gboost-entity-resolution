#!/usr/bin/env bash
# v14: best table so far (v13 > v12 > v10) + second transformer ce3. Waits for data/ce3_val.parquet + data/ce3_test.parquet.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 TUNE_THREADS=8
PY="python -W ignore -u"
until [ -f data/ce3_val.parquet ] && [ -f data/ce3_test.parquet ]; do sleep 30; done
# never tune two tables at once: wait for any v12 / v13 run that has its inputs
while { [ -f data/ce2_val.parquet ] && ! grep -qE "V12 DONE|Traceback" data/v12.log 2>/dev/null; } || \
      { [ -f data/llm_val.parquet ] && ! grep -qE "V13 DONE|Traceback" data/v13.log 2>/dev/null; }; do sleep 30; done
if grep -q "V13 DONE" data/v13.log 2>/dev/null; then B=v13; elif grep -q "V12 DONE" data/v12.log 2>/dev/null; then B=v12; else B=v10; fi
echo "base table: $B"
$PY src/add_ce3.py data/stage2${B}_train.parquet data/ce3_val.parquet data/stage2v14_train.parquet || exit 1
$PY src/add_ce3.py data/frpatch/stage2${B}_test.parquet data/ce3_test.parquet data/frpatch/stage2v14_test.parquet || exit 1
EXCL='k_all|k_src|rank_src|rank_all|t1_.*_gap2?|t1_comb2?_rank|t1_q_ncand|t1_bscore.*|t1_brank'
TABLE=data/stage2v14_train.parquet FINAL=data/stage2v14_final.json MODEL_PREFIX=data/model2v14 \
  BEST_FROM=data/v3_old/stage2v3_final.json EXCLUDE="$EXCL" FAST=1 $PY src/tune_stage2.py 0 || exit 1
$PY src/drift_check.py data/stage2v14_final.json data/stage2v14_train.parquet data/frpatch/stage2v14_test.parquet 0.10
echo "drift gate exit $?"
$PY src/apply_table.py data/stage2v14_final.json data/model2v14 data/frpatch/stage2v14_test.parquet output_v14 0.85 0.92 || exit 1
for d in output_v14 output_v14_t85 output_v14_t92; do python utils/validate_submission.py --matching $d/matching_results.tsv --candidate $d/candidate_pairs.tsv --test-dir dataset/test | tail -1; done
echo "V14 DONE"
