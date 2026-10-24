# X_Gboost — Project Guide (Amazon ML Challenge 2026, Business Entity Resolution)

Read this first. It explains **what we built, why, and how to run it**. The day-by-day log with every experiment and
leaderboard score is in `STATUS.md` (see its "Gains ledger" table). The formal write-up for the organisers is
`Documentation_template.md`.

---

## 1. The problem in one paragraph
Three sources of business records (name, address, country). Source 1 (S1) is a clean reference list; Sources 2 and 3
(S2/S3) are noisy copies plus distractors. For every S1 entity we must list the S2/S3 records of the same business.
Score = F0.5 computed **per S1 entity and averaged** (precision counts twice as much as recall; an S1 with no true
matches scores 1.0 only if we predict nothing). Train: 2.2M S1 / 10.3M S2+S3 (US, India). Test: 1.7M S1 / 10.0M S2+S3
(US, India **and France, which never appears in training**).

## 2. Key facts we discovered (they drive every design decision)
| Fact | Consequence |
|---|---|
| Every S2/S3 record matches **at most one** S1 | Decide per record: take its best S1, accept or reject |
| ~26% of S2/S3 (train) / ~37% (test) are distractors | Rejecting is as important as matching; test is harder than train |
| 5.6% of S1 have no match (singletons) | A single false merge on them costs a full 1.0 |
| 31% of S1 names repeat | Address + house number are essential |
| **Decoys**: near-copies of an S1 with one change — legal form swapped (Inc→LLC), house number digit changed (7765→7769), a business word appended (Pvt, Holdings, Group, Ventures…), a name word swapped | Dedicated "decoy marker" features |
| Hindi re-transliteration (gret indastrij = great industries), leetspeak (hea1th), state codes (mh = Maharashtra) | Normalisation + phonetic keys + learned translation dictionary |
| France addresses mix region and department names (Hauts-de-France vs Nord) | France-only address patch |

## 3. Pipeline architecture
```
raw TSV ──► normalise ──► BLOCKING ──► STAGE 1 (pair model) ──► STAGE 2 (verifier) ──► threshold + caps ──► TSVs
            normalize.py   blocking.py    features.py (69 feats)    stage2v3.py + markers     apply_table.py
            preprocess.py  (~7 cands/rec) pipeline.py (LightGBM)    tune_stage2.py (LGBM/XGB)
```
1. **Normalise** (`normalize.py`, `preprocess.py`): transliterate any script to Latin (anyascii), lowercase, strip legal
   forms into a "core" name, split DBA / "formerly" names, canonical address abbreviations, US states, France admin names.
2. **Blocking** (`blocking.py`): inverted index of rare keys, always prefixed by country. Key families: name tokens,
   consonant skeletons, phonetic codes, full name, name×address word, house number×address word, address bigrams,
   concatenated names (domains), number×name prefix, address pairs. Probe each record's 12 rarest keys, rescore with
   IDF, keep top 8. Recall ≈ 97.5% of true pairs.
3. **Stage 1** (`features.py`, `pipeline.py`): 69 similarity features per candidate pair (fuzzy name/address scores,
   phonetic, cleaned names, number agreement, competition with the other candidates) → LightGBM probability.
4. **Stage 2** (`stage2.py`, `stage2v3.py`, markers): for each record's best S1 only, a second model sees the full
   stage-1 features of best + runner-up, **cluster coherence** with the other confident records of the same S1, and the
   **decoy / word / clean / translated markers**. LightGBM (+ XGBoost ensemble), tuned with Optuna.
5. **Decision** (`apply_table.py`): accept if probability > threshold (0.85 is best on test), at most 5 S2 / 6 S3
   matches per S1 (caps learned from the training fold).

## 4. How we validate (and why it is trustworthy)
* S1 entities split 80/20 by hash. Stage 1 trains only on records whose true S1 is in the 80% **and none of whose
  candidates is a held-out S1** (`select_training_records` in `pipeline.py`) — so validation is out-of-sample.
* Stage-2 tuning uses half A of the held-out fold; the chosen model is scored once on the untouched half B.
* `tests/test_leakage.py` — 29 pytest checks prove there is no leakage (`python -m pytest tests/test_leakage.py -v`).
* **Drift gate** (`drift_check.py`): every model input must look the same on validation and test (PSI < 0.10).
  Added after v3 scored 0.910 on the leaderboard despite a great validation score (its cluster features depended on how
  many junk records a dataset has, which differs between train and test).

## 5. Version history (leaderboard = public score on Unstop)
| Version | What changed | Validation | Leaderboard |
|---|---|---|---|
| v2 | blocking + stage 1 + first stage 2 | 0.9777 | 0.963 |
| v3 | rich cluster features | 0.9805 | **0.910 ✗** (drift) |
| v4 | drift-robust stage 2 (confident members only) | 0.9787 | – |
| v5 | + decoy markers (legal-form conflict, house-number change type) + France address patch | 0.9805 | 0.968 |
| v5 @ thr 0.85 | stricter threshold (test has more decoys) | – | 0.971 |
| v7 @ 0.85 | + word markers (decoy words, alias words) + cleaned comparisons (Indian states, US noise, French decoy words) | 0.9811 | **0.973** |
| v8 @ 0.85 | + learned translation dictionary (14.7k name + 28.8k address mappings from training pairs) | 0.9813 | **0.973** |
Diagnostic uploads (not attempts): France emptied = 0.831 (→ France ≈ 0.94), all empty = 0.056 (singleton share).

## 6. File map (`src/`)
| File | Role |
|---|---|
| normalize.py / preprocess.py | text normalisation → `data/*_s{1,2,3}.parquet` |
| blocking.py | key-based candidate generation (+ optional translated / typo-tolerant keys via env flags) |
| features.py | stage-1 pair features |
| pipeline.py | `block` / `train` (leakage guard) / `infer` / `evaluate` |
| stage2.py | first stage-2 verifier, caps, decide() |
| stage2v3.py | stage-2 table builder (best + runner-up features, confident-member cluster stats), France-aware |
| decoy_markers.py, word_markers.py, clean_markers.py, translit_markers.py | the marker feature families |
| translit_dict.py | learns the translation dictionary from training-fold pairs |
| tune_stage2.py | Optuna / XGBoost / ensemble, A/B protocol; optional shift weights and distractor weights |
| drift_check.py | drift gate |
| apply_table.py | applies a tuned stage 2 to test and writes both TSVs at several thresholds |
| france_patch.py | France-only address re-normalisation, re-blocking and re-scoring |
| shift_weights.py | covariate-shift (adversarial validation) weights — v9, not yet run |
| cross_encoder.py, ce_probe.py | small transformer (MiniLM) experiments on the local GPU |
| hybrid_block.py, tfidf_block_dev.py | character-trigram TF-IDF blocking (after the SABER hybrid-blocking design) — too slow on CPU, see §8 |
| metric.py | macro F0.5 exactly as the leaderboard computes it |
| run_*.sh | resumable drivers for each version |

## 7. How to run (Windows, Git Bash, Python 3.11, `pip install -r requirements.txt`)
```bash
python src/preprocess.py dataset train && python src/preprocess.py dataset test
bash src/run_all.sh            # blocking, stage 1, held-out scoring, first stage 2, test scoring  (v2)
python src/france_patch.py     # France address patch
bash src/run_v5.sh; bash src/run_v6.sh; bash src/run_v7.sh; bash src/run_v8.sh
python utils/validate_submission.py --matching output_v8_t85/matching_results.tsv --candidate output_v8_t85/candidate_pairs.tsv --test-dir dataset/test
```
Full runs take many hours on a 16 GB laptop (blocking ~1.5 h per split, scoring ~3 h per split). The trained models in
this package let you skip training: `data/model.txt` (stage 1), `data/model2v8_lgb.txt` + `data/stage2v8_final.json`
(stage 2 v8).

## 8. Lessons and open items
* **Validation overshoots the leaderboard by ~1 point**: test has more decoys and France. Validation is a filter;
  the leaderboard is the judge. Stricter thresholds (0.85–0.92) help on test.
* **Remaining gap is recall of candidate generation** (our candidates allow at most 0.989; the SABER hybrid blocker
  reaches 0.998 on a 2.5% sample). Its character-trigram leg is far too slow on a laptop CPU at full size (≈14 h for India
  alone); it needs a GPU (SABER uses CUDA sparse products + region partitioning).
* **Open:** Kaggle transformer (bge-reranker-v2-m3) on uncertain pairs → merge as a stage-2 feature (v10);
  covariate-shift weights (v9); a full SABER-style run on a GPU.

## 9. Timeline — what we did, in order (25 Sept → 27 Sept)
**25 Sept (day 1) — foundation**
1. EDA: one-S1-per-record structure, 26% distractors, 5.6% singletons, repeated names, noise patterns, France only in test.
2. Normaliser for US / India / France; all 24M records normalised to parquet (~5 min per source).
3. Blocking: word keys gave only 81.8% recall → compound keys (name×address, number×word, bigrams) 95.8% → phonetic +
   concatenated-name keys 96.0% → typo/number-prefix/address-pair keys, top-8 → 97.5%.
4. Stage-1 LightGBM (44 → 69 features) and first stage-2 cluster verifier. v1 validation 0.9651 — later found to contain
   a **leak** (training pairs could include held-out S1s); fixed with the leakage guard + 22 (now 29) pytest checks.
5. Laptop issues solved along the way: out-of-memory crashes (chunking), sessions killing jobs (detached processes),
   swap thrashing (one heavy job at a time).

**26 Sept (day 2) — leaderboard feedback loop**
6. 04:30 v2 submission → leaderboard **0.963** (validation 0.9777): ~1.5-point gap.
7. Diagnostic uploads: France emptied → France ≈ 0.94; all-empty → singleton share 0.056 (as in train). Audit: no bugs,
   but test has ~37% distractors vs 27% and twice the house-number conflicts (more decoys).
8. v3 (rich cluster features) validated 0.9805 but scored **0.910** → root cause: features depending on how many junk
   records a dataset has. Built the **drift gate**; v4 = drift-robust stage 2.
9. Error analysis → **decoy markers** (legal-form conflict 46× more common in decoys; house-number digit substitution
   8×) + France address patch → v5 **0.968**; stricter threshold 0.85 → **0.971** (0.92 also 0.971).
10. Full vocabulary audit of every word in the data → word markers (decoys append pvt/holdings/group/ventures…; aliases
   dba/formerly/aka only on true copies), Indian state normalisation, US city/CDP/township noise, French decoy words →
   v7 **0.973**. Learned translation dictionary (43k mappings from training pairs) → v8 **0.973** (plateau).

**27 Sept (day 3) — recall and transformers**
11. Research into the SABER hybrid blocker: remaining gap is candidate recall (0.989 ceiling vs SABER's 0.998 on a sample).
   Character-trigram TF-IDF leg ported (hybrid_block.py) but measured ~14 h for India alone on CPU → needs a GPU.
12. Transformer: local MiniLM on the MX450 too slow/weak alone (blend +0.0014 AUC); Kaggle notebook with
   bge-reranker-v2-m3 on uncertain pairs prepared and launched (results → stage-2 feature, "v10").
13. Deadline: 27 Sept 11:59 PM IST. Best file so far: `releases/v8_t85/matching_results.tsv` (0.973).
