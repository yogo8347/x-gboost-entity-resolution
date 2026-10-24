#!/usr/bin/env bash
# v13: best stage-2 table so far (v12 if it finished, else v10) + LLM judge. Waits for data/llm_val.parquet + data/llm_test.parquet.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 TUNE_THREADS=8
PY="python -W ignore -u"
until [ -f data/llm_val.parquet ] && [ -f data/llm_test.parquet ]; do sleep 30; done
while [ -f data/ce2_val.parquet ] && ! grep -qE "V12 DONE|Traceback" data/v12.log 2>/dev/null; do sleep 30; done  # never tune two tables at once
if grep -q "V12 DONE" data/v12.log 2>/dev/null; then B=v12; else B=v10; fi
echo "base table: $B"
$PY src/add_llm.py data/stage2${B}_train.parquet data/llm_val.parquet data/stage2v13_train.parquet || exit 1
$PY src/add_llm.py data/frpatch/stage2${B}_test.parquet data/llm_test.parquet data/frpatch/stage2v13_test.parquet || exit 1
EXCL='k_all|k_src|rank_src|rank_all|t1_.*_gap2?|t1_comb2?_rank|t1_q_ncand|t1_bscore.*|t1_brank'
TABLE=data/stage2v13_train.parquet FINAL=data/stage2v13_final.json MODEL_PREFIX=data/model2v13 \
  BEST_FROM=data/v3_old/stage2v3_final.json EXCLUDE="$EXCL" FAST=1 $PY src/tune_stage2.py 0 || exit 1
$PY src/drift_check.py data/stage2v13_final.json data/stage2v13_train.parquet data/frpatch/stage2v13_test.parquet 0.10
echo "drift gate exit $?"
$PY src/apply_table.py data/stage2v13_final.json data/model2v13 data/frpatch/stage2v13_test.parquet output_v13 0.85 0.92 || exit 1
for d in output_v13 output_v13_t85 output_v13_t92; do python utils/validate_submission.py --matching $d/matching_results.tsv --candidate $d/candidate_pairs.tsv --test-dir dataset/test | tail -1; done
echo "V13 DONE"
