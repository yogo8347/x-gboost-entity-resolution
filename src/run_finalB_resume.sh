#!/usr/bin/env bash
# Resume FINAL-B (v12 + ce4) from its already-built tables: tune (FAST), drift check, apply at 0.85, validate.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 TUNE_THREADS=8
PY="python -W ignore -u"
while [ ! -f data/zip_done.flag ]; do sleep 20; done          # never alongside the final zip build
EXCL='k_all|k_src|rank_src|rank_all|t1_.*_gap2?|t1_comb2?_rank|t1_q_ncand|t1_bscore.*|t1_brank'
TABLE=data/stage2vFB_train.parquet FINAL=data/stage2vFB_final.json MODEL_PREFIX=data/model2vFB \
  BEST_FROM=data/v3_old/stage2v3_final.json EXCLUDE="$EXCL" FAST=1 $PY src/tune_stage2.py 0 || exit 1
$PY src/apply_table.py data/stage2vFB_final.json data/model2vFB data/frpatch/stage2vFB_test.parquet output_vFB 0.85 || exit 1
python utils/validate_submission.py --matching output_vFB_t85/matching_results.tsv --candidate output_vFB_t85/candidate_pairs.tsv --test-dir dataset/test --check-ids | tail -1
echo "FINALB DONE"
