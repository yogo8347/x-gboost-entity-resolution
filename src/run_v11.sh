#!/usr/bin/env bash
# v11: hybrid blocking (key + char-trigram TF-IDF, both directions, adaptive K) -> stage 1 retrained on the merged
# candidates -> out-of-sample scores -> stage 2 with all markers -> drift gate -> files. Every step is resumable.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 TUNE_THREADS=8
PY="python -W ignore"
EXP="$1"; [ -n "$EXP" ] && until grep -qE "UNION|Traceback" "$EXP"; do sleep 20; done
step() { echo "=== $1  $(date +%H:%M)"; }
step "hybrid blocking train"; [ -f data11/train_cands.parquet ] || $PY src/hybrid_block.py train data data/train_cands.parquet data11/train_cands.parquet || exit 1
export ER_DATA=data11
step "stage-1 training";      [ -f data11/model.txt ] || $PY src/pipeline.py train || exit 1
step "held-out scoring";      [ -f data11/train_scored.parquet ] || $PY src/pipeline.py infer train || exit 1
step "stage-1 evaluate";      [ -f data11/best.json ] || $PY src/pipeline.py evaluate || exit 1
step "hybrid blocking test";  [ -f data11/test_cands.parquet ] || $PY src/hybrid_block.py test data/frpatch data/frpatch/test_cands.parquet data11/test_cands.parquet || exit 1
step "test scoring";          [ -f data11/test_scored.parquet ] || ER_PREP=data/frpatch $PY src/pipeline.py infer test || exit 1
step "stage-2 tables";        [ -f data11/stage2v3_train.parquet ] || $PY src/stage2v3.py build train || exit 1
                              [ -f data11/stage2v3_test.parquet ] || ER_PREP=data/frpatch $PY src/stage2v3.py build test || exit 1
step "markers"
$PY src/word_markers.py data11/stage2v3_train.parquet data11/t_w.parquet data train || exit 1
$PY src/clean_markers.py data11/t_w.parquet data11/t_wc.parquet data train || exit 1
$PY src/translit_markers.py data11/t_wc.parquet data11/stage2_train_full.parquet data train || exit 1
$PY src/word_markers.py data11/stage2v3_test.parquet data11/e_w.parquet data/frpatch test || exit 1
$PY src/clean_markers.py data11/e_w.parquet data11/e_wc.parquet data/frpatch test || exit 1
$PY src/translit_markers.py data11/e_wc.parquet data11/stage2_test_full.parquet data/frpatch test || exit 1
step "stage-2 tuning"
EXCL='k_all|k_src|rank_src|rank_all|t1_.*_gap2?|t1_comb2?_rank|t1_q_ncand|t1_bscore.*|t1_brank'
TABLE=data11/stage2_train_full.parquet FINAL=data11/final.json MODEL_PREFIX=data11/model2 \
  BEST_FROM=data/v3_old/stage2v3_final.json EXCLUDE="$EXCL" $PY src/tune_stage2.py 0 || exit 1
$PY src/drift_check.py data11/final.json data11/stage2_train_full.parquet data11/stage2_test_full.parquet 0.10
echo "drift gate exit $?"
step "files"
ER_PREP=data/frpatch $PY -c "
import sys; sys.path.insert(0,'src'); import os, polars as pl
from pipeline import write_lists
s1 = pl.read_parquet('data/frpatch/test_s1.parquet', columns=['entity_id']).with_row_index('rid').with_columns(pl.col('rid').cast(pl.UInt32))
q = pl.concat([pl.read_parquet(f'data/frpatch/test_s{k}.parquet', columns=['entity_id']) for k in (2, 3)]).with_row_index('qid').with_columns(pl.col('qid').cast(pl.UInt32))
c = pl.read_parquet('data11/test_scored.parquet', columns=['qid', 'rid']).join(q.rename({'entity_id': 'e'}), on='qid').join(s1.rename({'entity_id': 's1'}), on='rid')
os.makedirs('data11/cands', exist_ok=True); write_lists(s1, c, 'candidate_entity_ids', 'data11/cands/candidate_pairs.tsv')" || exit 1
CAND_TSV=data11/cands/candidate_pairs.tsv ER_PREP=data/frpatch $PY src/apply_table.py data11/final.json data11/model2 data11/stage2_test_full.parquet output_v11 0.85 0.92 || exit 1
for d in output_v11 output_v11_t85 output_v11_t92; do python utils/validate_submission.py --matching $d/matching_results.tsv --candidate $d/candidate_pairs.tsv --test-dir dataset/test | tail -1; done
echo "V11 DONE $(date +%H:%M)"
