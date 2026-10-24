#!/usr/bin/env bash
# FINAL: one stage-2 model on ALL signals that have arrived: v10 table + ce2 (transformer v2) + llm (LLM judge) + ce3
# (second transformer) + ensemble averages. Waits until all three Kaggle outputs are in data/ or until CUTOFF, whichever
# comes first, then builds one table, tunes once (A/B protocol), drift-checks and writes the submission.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 TUNE_THREADS=8
PY="python -W ignore -u"
CUTOFF=${CUTOFF:-21:15}
have() { [ -f data/$1_val.parquet ] && [ -f data/$1_test.parquet ]; }
until { have ce2 && have ce3 && have ce4; } || [[ "$(date +%H:%M)" > "$CUTOFF" ]]; do sleep 30; done
have ce2 || { echo "no ce2 -> nothing to add beyond v10"; exit 1; }
while ! grep -qE "V12 DONE|Traceback" data/v12.log 2>/dev/null; do sleep 30; done   # never two heavy jobs at once
echo "$(date +%H:%M) building final table: ce2=yes llm=dropped ce3=$(have ce3 && echo yes || echo no) ce4=$(have ce4 && echo yes || echo no)"
TR=data/stage2v12_train.parquet; TE=data/frpatch/stage2v12_test.parquet
if [ -n "$USE_LLM" ] && have llm; then  # LLM judge dropped: validation AUC 0.4776 on hard pairs (no signal)
  $PY src/add_llm.py $TR data/llm_val.parquet data/stage2vF${TAG}1_train.parquet || exit 1
  $PY src/add_llm.py $TE data/llm_test.parquet data/frpatch/stage2vF${TAG}1_test.parquet || exit 1
  TR=data/stage2vF${TAG}1_train.parquet; TE=data/frpatch/stage2vF${TAG}1_test.parquet
fi
for C in ${CE_COLS:-ce3 ce4}; do
  if have $C; then
    $PY src/add_ce_col.py $TR data/${C}_val.parquet $C data/stage2vF${TAG}_${C}_train.parquet || exit 1
    $PY src/add_ce_col.py $TE data/${C}_test.parquet $C data/frpatch/stage2vF${TAG}_${C}_test.parquet || exit 1
    TR=data/stage2vF${TAG}_${C}_train.parquet; TE=data/frpatch/stage2vF${TAG}_${C}_test.parquet
  fi
done
cp $TR data/stage2vF${TAG}_train.parquet; cp $TE data/frpatch/stage2vF${TAG}_test.parquet
EXCL='k_all|k_src|rank_src|rank_all|t1_.*_gap2?|t1_comb2?_rank|t1_q_ncand|t1_bscore.*|t1_brank'
TABLE=data/stage2vF${TAG}_train.parquet FINAL=data/stage2vF${TAG}_final.json MODEL_PREFIX=data/model2vF${TAG} \
  BEST_FROM=data/v3_old/stage2v3_final.json EXCLUDE="$EXCL" FAST=1 $PY src/tune_stage2.py 0 || exit 1
$PY src/drift_check.py data/stage2vF${TAG}_final.json data/stage2vF${TAG}_train.parquet data/frpatch/stage2vF${TAG}_test.parquet 0.10
echo "drift gate exit $?"
$PY src/apply_table.py data/stage2vF${TAG}_final.json data/model2vF${TAG} data/frpatch/stage2vF${TAG}_test.parquet output_vF${TAG} 0.85 0.92 || exit 1
for d in output_vF${TAG}_t85; do python utils/validate_submission.py --matching $d/matching_results.tsv --candidate $d/candidate_pairs.tsv --test-dir dataset/test | tail -1; done
echo "FINAL${TAG} DONE"
