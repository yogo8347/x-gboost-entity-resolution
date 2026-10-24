#!/usr/bin/env bash
# v6: v5 + word markers (decoy extra words, alias markers). Uses the existing v5 stage-2 tables.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 TUNE_THREADS=8
PY="python -W ignore"
$PY src/word_markers.py data/stage2v3_train.parquet data/stage2v6_train.parquet data train                      || exit 1
$PY src/word_markers.py data/frpatch/stage2v3_test.parquet data/frpatch/stage2v6_test.parquet data/frpatch test || exit 1
EXCL='k_all|k_src|rank_src|rank_all|t1_.*_gap2?|t1_comb2?_rank|t1_q_ncand|t1_bscore.*|t1_brank'
TABLE=data/stage2v6_train.parquet FINAL=data/stage2v6_final.json MODEL_PREFIX=data/model2v6 \
  BEST_FROM=data/v3_old/stage2v3_final.json EXCLUDE="$EXCL" $PY src/tune_stage2.py 0                         || exit 1
$PY src/drift_check.py data/stage2v6_final.json data/stage2v6_train.parquet data/frpatch/stage2v6_test.parquet 0.10
echo "drift gate exit $?"
cp data/frpatch/stage2v6_test.parquet data/frpatch/stage2v3_test_v6.parquet
ER_DATA=data/frpatch ER_PREP=data/frpatch $PY -c "
import sys, os; sys.path.insert(0,'src'); import stage2v3, polars as pl
# apply reads DATA/stage2v3_test.parquet: point it at the v6 table via a temporary swap
os.replace('data/frpatch/stage2v3_test.parquet', 'data/frpatch/stage2v3_test_v5.parquet')
os.replace('data/frpatch/stage2v3_test_v6.parquet', 'data/frpatch/stage2v3_test.parquet')
try:
    stage2v3.apply('data/stage2v6_final.json', 'data/model2v6', 'output_v6')
finally:
    os.replace('data/frpatch/stage2v3_test.parquet', 'data/frpatch/stage2v3_test_v6.parquet')
    os.replace('data/frpatch/stage2v3_test_v5.parquet', 'data/frpatch/stage2v3_test.parquet')
" || exit 1
python utils/validate_submission.py --matching output_v6/matching_results.tsv --candidate output_v6/candidate_pairs.tsv --test-dir dataset/test --check-ids
echo "V6 DONE"
