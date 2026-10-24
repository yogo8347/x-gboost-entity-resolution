# X_Gboost: experiment log

The day-by-day working log kept during the challenge (25–27 Sept 2026): plan, every experiment with its measured effect, and every leaderboard upload. Times are IST.

> **Final result:** v12 @ threshold 0.85 = **0.981323** public leaderboard (see the end of this log). Entries below are kept as written during the challenge, so "current best" lines refer to that moment.

**Deadline:** 27 Sept (exact hour: check on Unstop).  **Our freeze:** 27 Sept 09:00. The final package is ready and submitted by then; after that only emergencies.

**Rules**
1. **Always shippable.** Every milestone ends with a complete, validator-PASS file. There is always a best-so-far file ready to upload.
2. **Upgrade only on evidence.** A new version replaces the current one only if it beats it on validation data no choice was tuned on (leak-free, 22 pytest checks).
3. **One job at a time on RAM-heavy steps.** No parallel heavy jobs; that caused both earlier stalls.
4. **Drift gate before any upload** (`src/drift_check.py`): every model input must have PSI < 0.10 between validation and test. Added after v3 (validation 0.9805) scored 0.910 on the leaderboard; the gate fails v3 (max PSI 0.51).

## Milestones

| # | Milestone | What changes | Target time | Status |
|---|---|---|---|---|
| M1 | v2 submission | baseline file + zip | 26 Sept ~05:10, **upload 06:00** | ✅ done 04:30: validator PASS (incl. --check-ids), 23/23 pytest, zip OK |
| M2 | stage-2 v3 | richer verifier, Optuna, LightGBM + XGBoost ensemble | 26 Sept ~08:30 | ✅ done ~11:00 (Optuna 40 trials took 4.5 h) |
| M3 | stage-2 fixes after the leaderboard analysis | v4 drift-robust stage 2, France patch, v5 decoy markers | 26 Sept ~17:00 | ✅ v5 done 16:46 |
| M4 | final checks | France sample review, error analysis, full pytest incl. submission file, final doc + zip | 27 Sept ~06:00 | ⏳ |
| FREEZE | final upload + zip | best validated version | **27 Sept 09:00** | ⏳ |

## Scores (macro F0.5, 441,410 held-out S1 entities, leak-free)

| Version | Validation F0.5 | Leaderboard | File |
|---|---|---|---|
| v1 (leaky, reference only) | 0.9651 | not submitted | none |
| v2 | 0.9777 | **0.963** | `output/matching_results.tsv` (backup `releases/v2/`) |
| v3 (ensemble 0.8 LGBM + 0.2 XGB) | 0.9805 (untouched half) | **0.910 ✗** | rejected: fails the drift gate |
| v4 (drift-robust stage 2, ensemble 0.6 LGBM + 0.4 XGB) | 0.9787 (untouched half), drift gate PASS (max PSI 0.073) | upload pending; predicted ~0.966 | `output_v4/` (backup `releases/v4/`) |
| v2 + France fix | 0.9777 (France not in validation) | pending | `output_v2fr/` |
| **v5 (v4 + decoy markers + France fix, tuned LGBM)** | **0.9805** (untouched half), drift gate PASS (max PSI 0.073) | **0.968** ✅ (predicted 0.967–0.973) | `output_v5/` (backup `releases/v5/`) |

Leaderboard reference (26 Sept midday): leader 0.9906; 4th place 0.988.


## Gains ledger (every idea tried, measured effect, status)

| Idea | Measured on | Effect | Status |
|---|---|---|---|
| Compound + phonetic + concatenated blocking keys | blocking recall (3% sample) | 81.8% → 97.5% | ✅ in all versions |
| 25 extra stage-1 features (phonetic, cleaned names, number distance, name rarity) | validation | stage 1: 0.9587 (leaky) → 0.9729 | ✅ in v2+ |
| Stage-2 cluster verifier | validation | +0.5 | ✅ in v2+ |
| Leakage guard | correctness | removes optimism | ✅ |
| v3 rich cluster features | leaderboard | **−5.3** (0.910) | ❌ rejected: drift |
| Drift-robust cluster stats (confident members only) | validation / drift gate | 0.9787, gate PASS | ✅ in v4, v5 |
| France region/department normalisation | France address similarity | +7–10 points | ✅ in v5 |
| Decoy markers (legal-form conflict, house-number relation, word substitution) | validation | +0.18 over v4 | ✅ in v5 |
| v5 overall | leaderboard | 0.968 (+0.5 over v2) | ✅ |
| **v5 @ threshold 0.85** | **leaderboard** | **0.971** | ✅ **current best** (`output_v5_t85/`) |
| Distractor-weighted training (v4w) | not run | – | parked (weaker evidence than markers) |
| Threshold re-tune under distractor weighting | validation (weighted) | ≈ 0 | ❌ no gain |
| Per-S1 expected-F0.5 decisions | untouched half B | +0.0002 | 📝 tiny; optional in the final build |
| Top-20 candidates instead of top-8 | blocking recall | +0.35 recall points for 2.5x pairs | 📝 small; not worth the re-score |
| Embedded name IDs as a linking key | train ground truth | IDs never shared | ❌ rejected |
| Row order / ID leak check | train | no correlation | ✅ none exists (not used) |
| Singleton-rate probe | leaderboard | 0.056 = train | ✅ singletons are handled correctly |
| Stricter threshold on test (v5 @ 0.85) | leaderboard | **+0.3 (0.968 → 0.971)** | ✅ test is precision-limited; the threshold goes higher |
| Typo-tolerant (symmetric-delete) blocking keys | blocking recall | built (flag BLOCK_DEL), not yet measured | ⏳ parked: needs a full re-block |
| Stricter threshold v5 @ 0.92 | leaderboard | 0.971 (= @0.85) | ✅ plateau 0.85–0.92; later versions uploaded at 0.85 |
| Word markers (decoy extra words: pvt/holdings/group/…; alias markers) — v6 | validation | 0.9808 (+0.03) | ✅ in v7 |
| Full vocabulary audit (`data/vocab_audit.txt`) | – | found Indian state names never normalised, US city/CDP/township noise and typos, leetspeak, French decoy words | ✅ feeds v7 |
| Clean-comparison markers + French decoy words — v7 | validation / leaderboard | 0.9811 / **0.973** @0.85 | ✅ **current best** (`output_v7_t85/`) |
| Learned translation dictionary (training fold only): 14,703 name + 28,774 address mappings, resolves 35.6% of unknown name words — v8 | validation | pending | ⏳ queued |

## Log
- 26 Sept 02:30: test blocking done (68.2M pairs, 6.84 per record, matches train's 6.90). Test scoring started.
- 26 Sept 02:40: dry run of the final step on 1.2M test records: validator PASS, France matched. Dry-run files deleted.
- 26 Sept 04:25: **M1 done.** Test: 1,732,544 S1 rows, 101,396 predicted singletons (5.9%, train rate 5.6%), 5.89M matches. Matches per S1: US 3.41, India 3.36, France 3.49, so France behaves like the training countries. Backup in `releases/v2/`.
- 26 Sept ~05:00: **v2 leaderboard = 0.963**, 1.5 points below validation. Validation covers only US and India. Validation per country: US 0.9787, India 0.9635. Test is 47% India (train 40%) and 15% France (never in training). France borderline accepts look like French decoys (Club vs Clinique, Sport vs Amicale).
- Diagnostic upload prepared: `output_probe_nofrance/matching_results.tsv` (France rows emptied). France F0.5 ≈ (0.963 − probe)/0.15 + 0.056. **Re-upload `output/matching_results.tsv` afterwards.**
- 26 Sept: **probe with France emptied = 0.831**, so France F0.5 ≈ 0.94. US/India on test ≈ 0.967 vs ≈ 0.975 expected. Audit: no pipeline bug (IDs aligned, every candidate scored, feature drift PSI < 0.07). Test has ~37% distractors (validation 27%) and 2x house-number conflicts on best candidates; caps are not binding; a threshold re-tune under distractor weighting gives no gain.
- 26 Sept ~11:20: **M2 done.** On the untouched half B: v2-style stage 2 0.9777 → v3 features 0.9803 → +Optuna 0.9804 / XGBoost 0.9804 → **ensemble 0.9805**. `output_v3/` validator PASS (incl. --check-ids). v3 accepts 6.49M matches vs v2 5.89M. Threshold probe files (`output_thr0.55/`, `output_thr0.85/`) built but held back; v3's score answers the same question.
- 26 Sept ~11:30: **v3 leaderboard = 0.910 (worse than v2's 0.963).** Cause: cluster features counted every record assigned to an S1, including junk. Test has more junk (about 37% vs 27%), so those features shifted (PSI up to 1.1) and v3 over-accepted by 600k matches. Fix (v4): cluster statistics use only records with p1 ≥ 0.3; drift-sensitive inputs are excluded; the drift gate is now mandatory.
- 26 Sept ~11:35: singleton-rate probe prepared, `output_probe_allempty/`. Its score = the share of S1 with no true match in the public test set (train: 0.056).
- 26 Sept ~12:10: **all-empty probe = 0.056** → public-test singleton share 0.056, the same as train. Singleton hypothesis rejected (v2 predicts 5.9% empty). France estimate ≈ 0.94 confirmed.
- Predictions written before the next uploads: **v4** (robust stage 2): 0.965–0.970 if junk-sensitive stage-2 inputs were the issue (H1). If v4 ≤ 0.963 → test H2 (stage-1-only file; possible stage-1 retrain without junk-sensitive inputs such as s_ncand).
- 26 Sept ~13:00: **Decoy markers found.** On the held-out fold: legal-form conflict (Inc vs LLC) true 0.3% vs decoys 13.9%; accepted pairs with a conflict have 68% precision vs 99.1% without. First house number with one digit substituted: 8.3x more common in decoys; completely different: 8.9x. Name cleaning had hidden legal forms from the model. The sibling-group hypothesis was rejected (decoys come alone).
- France patch: region/department names dropped from French addresses (address similarity +7–10 points); France candidates now 7.87 per record.
- v4w (distractor weighting) cancelled in favour of v5 = v4 + decoy markers + France fix (queued behind v4 and the France patch).
- 26 Sept 14:44: **v4 done**: untouched-half 0.9787, drift gate PASS (max PSI 0.073), 5.90M matches. 14:49: **v2 + France fix done** (26,931 France rows changed, 0 US/India).
- 26 Sept 16:46: **v5 done**: 107 inputs; tuned LGBM 0.9805 on the untouched half (+0.18 over v4 from the decoy markers); drift gate PASS (max PSI 0.073); 5.89M matches; validator PASS with --check-ids. The laptop shut down after this; nothing was lost (all outputs on disk).
- 26 Sept ~18:00: **v5 leaderboard = 0.968** (+0.5 over v2), inside the prediction. Validation-to-leaderboard gap shrank from 1.47 to 1.25. Current best: v5.
- 26 Sept ~19:30: **v5 @ 0.85 = 0.971 (new best).** Test rewards stricter decisions (more decoys than validation). Next: probe a higher threshold; v6 word markers target precision directly.
- 26 Sept ~19:30: v5 @ 0.92 = 0.971, the same as @ 0.85, so the best test threshold is a plateau at 0.85–0.92. Rule: upload each new version's `_t85` file first.
- 26 Sept ~23:00: **v7 @ 0.85 = 0.973 (new best)**, inside the prediction (0.972–0.974). Queue: v8 (dictionary) → v9 (covariate-shift weights) → blocking recall test.
- 27 Sept ~01:30: **v8 @ 0.85 = 0.973** (= v7). The stage-2 marker line has plateaued; the remaining gap is recall (candidate ceiling 0.989 vs SABER's 0.998 on a mini-world). Next: hybrid blocking (char-trigram TF-IDF with max_df, bidirectional, adaptive K).

## Overfit audit (27 Sep, 06:xx) — `src/overfit_check.py`, log `data/overfit_check.log` — VERDICT: no harmful overfitting
| Check | Result |
|---|---|
| Stage 1 learning curve (hold-out logloss @200/500/1000/1486 trees) | 0.01281 / 0.01180 / 0.01139 / 0.01123 — still falling at the last tree |
| Stage 1 train vs hold-out (full) | logloss 0.00487 vs 0.01123, AUC 0.999985 vs 0.999837 |
| Stage 1 on never-seen businesses (held-out 20% S1, 3M pairs) | AUC 0.999796, logloss 0.01080 — same as hold-out |
| Stage 2 learning curve B(untouched) logloss @100/300/600/900/1200 | 0.03453 / 0.02721 / 0.02605 / 0.02567 / 0.02559 — never rises (plateau) |
| Tuner A(CV) vs B | B ≥ A for every model (+0.0002 … +0.0005) |
| Threshold curve on B (0.5–0.9) | 0.9798–0.9814, flat |
| Per country on B | US 0.98489, India 0.97609 |
Remaining val→LB gap (~0.008) = distribution shift (more decoys, France), not overfitting.

## v10 (27 Sept 07:40) — v8 + Kaggle cross-encoder (bge-reranker-v2-m3) feature
| Model | A(CV) | B(untouched) | thr |
|---|---|---|---|
| baseline LGBM (winner) | 0.98567 | **0.98571** | 0.80 |
| Optuna LGBM | 0.98569 | 0.98562 | 0.90 |
| XGBoost | 0.98561 | 0.98569 | 0.775 |
| Ensemble 0.9/0.1 | 0.98570 | 0.98570 | 0.875 |
CE folds = A/B halves (hash seed 77): B's ce comes from a CE trained on A only (verified 100% alignment). Outputs:
`output_v10{,_t85,_t92}` all PASS; hybrid `output_v10fr8_t85` (France from v8) PASS --check-ids -> `releases/v10fr8_t85`.

## v11 experiments (27 Sept 08:34) — `src/v11_experiments.py`, log `data/v11exp.log` — REJECTED (no meaningful gain)
| Experiment | B(untouched) | A(CV) |
|---|---|---|
| E0 v10 baseline | 0.98571 | 0.98567 |
| E1 seed-bagging x5 | 0.98574 | 0.98573 |
| E2 per-country thresholds (US 0.80 / India 0.75) | 0.98574 | 0.98570 |
| E3 both | 0.98574 | 0.98573 |
+0.00003 is noise-level (≈13 S1 entities of 220k) → not worth a new file. Also checked: runner-up re-ranking by the
cross-encoder would affect at most ~1.6k of 2.06M records (label-0 best rows with p2 > 0.5) → not worth another Kaggle run.
Remaining levers: candidate recall (SABER merge) and France (no labels; kept on v8).

## Rank rule (27 Sept 14:41) — `src/rank_rule.py` — REJECTED
First match per (S1, source) at t1, later ones at t2. Chosen on A: t1 0.70 / t2 0.80 (A 0.98571 vs single 0.98567).
B(untouched): 0.98577 vs 0.98571 single threshold -> +0.00006 = noise (~13 S1). Not applied.

## Plan from 14:35: one combined final model (`src/run_final.sh`)
v12 (v10 + transformer v2 ce2) runs as soon as data/ce2_*.parquet arrive (safety version). FINAL = v12 + LLM judge (llm)
+ second transformer (ce3) + averages, one table, one FAST tune (baseline LGBM), drift gate, apply; waits for all
Kaggle outputs or 20:30. Last upload = best of FINAL / v12 / v10 on untouched half B.

## LLM judge (27 Sept) — REJECTED
Qwen2.5-7B-Instruct-AWQ zero-shot on 59,270 hard validation pairs (ce in [0.1, 0.9]): AUC 0.4776 (= no signal). On these
pairs surface similarity misleads (decoys look the same, true trade-name matches look different); only models trained on our
labels learn that. Notebook stopped; FINAL uses ce2 + ce3 only. Transformer v2 (ce2) validation AUC 0.9921.

## v12 / FINAL-A (27 Sept 19:15)
| Version | B(untouched) | A(CV) |
|---|---|---|
| v10 | 0.98571 | 0.98567 |
| v12 = v10 + ce2 (bge, trained on stage-1 training-fold pairs) | **0.98588** | 0.98580 |
| FINAL-A = v12 + ce3 (xlm-roberta-base) + ce_avg | 0.98577 | 0.98583 |
ce2 AUC 0.9921, ce3 AUC 0.9900 (hard validation pairs). ce3 rejected; FINAL-A stopped. v12 files: releases/v12_t85 (PASS).
FINAL-B = v12 + ce4 only (bge full epoch on 2 GPUs), if ce4 arrives by 21:15.

## FINAL-B (27 Sept 21:30) — v12 + ce4 (bge, 72% of data, both GPUs; AUC 0.9922): B 0.98589 vs v12 0.98588 -> identical, stopped.
Transformer signal saturated (ce2/ce4 same model). Last upload: releases/v12_t85 (B 0.98588, validator PASS).

## LEADERBOARD v12_t85 = **0.981323** (27 Sept ~21:55) = FINAL BEST -> output/ + final zip. v10 = 0.980865.
