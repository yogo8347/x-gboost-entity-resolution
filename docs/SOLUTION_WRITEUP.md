# ML Challenge 2026: Business Entity Resolution Solution

**Entry:** X_Gboost  
**Author:** Yojit Goyal  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary
We treat the task as **assignment** rather than pairwise matching. In the training data every Source-2/3 record belongs to at most one Source-1 entity. So for every S2/S3 record we (1) retrieve a few S1 candidates with a multi-key, IDF-weighted inverted index, (2) score each pair with a LightGBM classifier on 70+ country-agnostic similarity features, and (3) accept only the best candidate after a **second, cluster-aware LightGBM** checks it. That second model looks at the group of records that chose the same S1: how many there are, where this record ranks among them, and how similar it is to the strongest other member. The one-match-per-record rule and caps on matches per source (learned from the training data) protect precision, which F0.5 weights twice as heavily as recall. The pipeline runs on a CPU and uses no external data.

---

## 2. Methodology

### 2.1 Problem Analysis
Findings from exploring the 2.2M S1 / 5.0M S2 / 5.3M S3 training records:

| Finding | Number | Consequence |
|---|---|---|
| Each S2/S3 id appears in the ground truth at most once | 7.64M matched ids, all unique | Decide per S2/S3 record: take the argmax S1, accept or reject |
| S2/S3 records that match nothing (distractors) | ≈26% of S2/S3 | A candidate always exists, so an explicit reject decision is needed |
| S1 singletons (no match) | 5.6% of S1 | One false merge on a singleton costs a full 1.0 |
| Matches per S1 | mean 3.5; S2 ≤ 5, S3 ≤ 6 | Hard caps per source |
| Repeated normalised S1 names | 31% of S1 | The name alone is often ambiguous; address evidence is essential |
| Missing S2/S3 address | ≈3.4% | Such records are often impossible to resolve; the model must abstain |
| Hard-negative decoys | most false merges in v1 | Near-copies of an S1 entity with one letter or one number changed, labelled as non-matches |

Noise patterns seen: junk prefixes (`--`, `>>`, `#`), duplicated or reordered tokens, typos including leetspeak (`re1iable`, `5teel`), DBA/"trading as" forms, domain-style names (`corinneconnorstrategic.com`), names in Devanagari/Kannada/Tamil script, **English names re-transliterated through Hindi** (`gret indastrij` = great industries, `praibhet` = private), zero-padded or truncated house numbers, shuffled address components, `null`/`N/A` placeholders, and abbreviation swaps (St/Street/Saint, state name/code).

### 2.2 Solution Strategy
**Approach Type:** Blocking + two-stage classifier (pairwise matcher, then cluster-aware verifier) with constrained assignment.  
**Core Innovation:**
1. **Transliteration-robust blocking and features:** a Soundex-style phonetic code without the first letter, which maps `yunik mitiya` and `unique media` to the same code, plus consonant skeletons and regex removal of transliterated legal words.
2. **Cluster-aware verification (stage 2):** a record is accepted only if it fits with the other records that chose the same S1. This is how the model rejects decoys.
3. **Structure-aware assignment:** at most one S1 per record, and caps on matches per source.

Pipeline: `normalise → block → stage-1 pair features + LightGBM → argmax per record → stage-2 cluster features + LightGBM → threshold + caps → output`.

---

## 3. Candidate Generation (Blocking)
All keys carry the country as a prefix, so candidates never cross countries (the country is treated as an open-ended label). Keys are hashed to 64-bit integers to save memory.

| Key family | Example | Purpose |
|---|---|---|
| `n:` name token | `n:keystone` | basic lexical match |
| `k:` consonant skeleton | `k:kystn` | typos, vowel variants |
| `p:` phonetic code / `q:` phonetic code of first two tokens | `q:52\|53` | Hindi back-transliteration |
| `f:` full core name | `f:olive tours` | exact names |
| `j:` concatenated name prefixes | `j:corinneconnorstrategic` | domain-style names |
| `c:` name token × address word | `c:ironclad\|dallas` | common names, kept rare by location |
| `a:` house number × address word (incl. state code) | `a:4351\|watson` | renamed or DBA records |
| `m:` house number × name prefix | `m:11610\|swe` | typos in every address word |
| `b:` / `w:` address word bigrams / pairs | `w:bahia\|scottsdale` | Indian addresses without numbers; renamed businesses |

**Retrieval** is IDF-weighted in two stages:
1. Probe each record's 12 rarest keys, skipping keys shared by more than 500 S1 records, and keep the 50 best S1 records by summed IDF.
2. Rescore those 50 using *all* shared keys and keep the **top 8**.

- **Candidate pairs generated:** 71,236,960 on train (6.9 per S2/S3 record); test similar (about 69M)
- **Blocking recall** (share of true pairs that land among the candidates), on a 3% sample: v1 81.8% → 95.8% (compound keys) → 96.0% (phonetic and concatenated keys) → **97.5%** (v2: number × name prefix, address pairs, deeper probing). Recall@1 = 95.5%.
- **How we avoided losing true matches:** compound keys stay rare even when each part is common. Probing the rarest keys bounds the cost. The second-stage rescoring fixes ordering errors from the probe. Every change was checked against measured recall@K.

---

## 4. Matching Model

### Stage 1 — pairwise matcher (LightGBM, 70+ features)
- **Name:** rapidfuzz ratio / token-set / token-sort / partial / Jaro-Winkler on the full, core (legal suffixes removed) and DBA-part names. Also: ratio after regex removal of legal words (including transliterated `praibhet`, `limitet`) and honorifics (`m/s`, `sri`, `smt`); phonetic-code ratio; consonant-skeleton ratio; space-free ratio (for domain names); raw Levenshtein distances; token Jaccard and containment in both directions; name lengths.
- **Name rarity:** how many S1 records share this S1's exact core name, and the query's core name. Generic names need address evidence.
- **Address:** token-set / sort / partial / plain ratios; address-word Jaccard and containment; empty-address flags.
- **Numbers:** first number equal, close (truncated or 1-edit), shared, exact set equality, extra or missing numbers on each side, log absolute difference of house numbers, conflict flag. These target the decoys ("268 14" vs "268 1").
- **Competition:** the gap to the best and second-best candidate of the same record for several scores, rank among the record's candidates, number of candidates, and how many records compete for this S1.
- **Source:** S2 vs S3. **The country is deliberately not a feature**, so the model transfers to France, which is absent from training.

Training uses 1.51M records chosen by the leakage guard in §5: true S1 in the training fold and no held-out S1 among their candidates. The objective is binary log-loss with early stopping on a held-out 10% of records.

### Stage 2 — cluster-aware verifier (LightGBM)
Each record is tentatively assigned to its best candidate. For that assignment:
- stage-1 p1, p2 and the gap between them
- key stage-1 similarities
- the number of other records assigned to the same S1 (overall and per source), and this record's rank among them
- the sum of the others' probabilities, and how many of them are highly confident
- **coherence:** name, address and number similarity to the strongest *other* member of the cluster
- S1 name length

Trained with 2-fold cross-validation over held-out S1 entities, on out-of-sample stage-1 scores.

**Threshold selection:** a sweep maximising macro F0.5 on the held-out S1 fold. Per-source caps (S2 ≤ 5, S3 ≤ 6) are then applied, keeping the highest-probability records.

---

## 5. Results & Error Analysis
**Validation protocol (leak-free, verified by 22 pytest checks in `tests/test_leakage.py`):**
- S1 entities are split by a deterministic hash: 80% training fold (1,765,411), **20% held out (441,410)**. Every record of one business falls on the same side.
- A record may train stage 1 only if its true S1 is in the training fold **and none of its blocking candidates is a held-out S1**. 2,476,019 records qualify; 1,506,998 are used (8.0M labelled pairs).
- All 7,723,230 records with a held-out S1 among their candidates are scored out-of-sample. They alone drive F0.5, the thresholds and stage-2 training (2-fold CV keyed by S1).
- Stage-2 caps (S2 ≤ 5, S3 ≤ 6) are measured on the training fold only. Test-time code never reads labels.

| Version | What changed | Validation F0.5 | Public leaderboard |
|---|---|---|---|
| v1 | first pipeline (contained a leak, later fixed) | 0.9651 | – |
| v2 | leak-free stage 1 (69 features) + stage-2 cluster verifier | 0.9777 | 0.963 |
| v3 | rich cluster features | 0.9805 | 0.910 (rejected: drift, see below) |
| v4 | drift-robust stage 2 (cluster statistics over confident members only) | 0.9787 | – |
| v5 | + decoy markers + France address patch | 0.9805 | 0.968 (0.971 at threshold 0.85) |
| v7 | + word markers, cleaned comparisons, French decoy words | 0.9811 | 0.973 @ 0.85 |
| **v8** | + learned translation dictionary | **0.9813** | **0.973 @ 0.85** |
| v10 hybrid | + cross-encoder score (fine-tuned bge-reranker-v2-m3) on uncertain pairs; France kept on v8 | 0.9857 | 0.9799 @ 0.85 |
| v10 | same, cross-encoder used for France too | 0.9857 | 0.9809 @ 0.85 |
| **v12 (final)** | + second bge cross-encoder trained on stage-1 training-fold pairs | **0.9859** | **0.9813 @ 0.85** |

Validation = macro F0.5 on the untouched half of the 20% held-out S1 fold. Leaderboard numbers are the public Unstop
scores. Diagnostic uploads (France emptied: 0.831; all empty: 0.056) were used to measure France (≈0.94) and the
public-test singleton share (0.056).

**Validation vs. test.** Validation overshoots the leaderboard by ~1 point because the test mix differs: ~37% distractors
(vs 27% in train), twice as many house-number conflicts on best candidates, and France (absent from training). Two
safeguards follow from this: (1) a **drift gate** (`src/drift_check.py`) that requires every model input to have
PSI < 0.10 between validation and test (v3 failed it with PSI 0.51 and scored 0.910); (2) a stricter decision threshold
(0.85) chosen on the leaderboard plateau (0.85 and 0.92 both 0.971 for v5).

**Decoy markers (v5–v8).** Error analysis showed most false merges are *decoys*: near-copies of an S1 entity. Measured on
the held-out fold: legal-form conflict (Inc vs LLC) occurs in 0.3% of true pairs vs 13.9% of decoys; first house number
with one digit substituted 1.2% vs 9.8%; an appended business word (pvt, holdings, group, ventures, public, exports,
industries, enterprises, overseas, infratech) 4 true vs 1,070 decoy pairs; alias words (dba, formerly, aka, fka, née)
appear only on true copies. Word lists were derived on one half of the held-out fold and evaluated on the other.

**Vocabulary audit and translation dictionary (v7–v8).** A full audit of every word found Indian state names never
normalised (mh / mharastr = Maharashtra, …), US address noise (city / CDP / township and their typos, PO box, PMB),
leetspeak (hea1th, lnc), and French decoy words (snc, holding, participations, …). A dictionary of 14,703 name and
28,774 address translations was learned from training-fold true pairs only.

**Cross-encoder feature (v10).** A multilingual cross-encoder (`BAAI/bge-reranker-v2-m3`, 568M parameters, Apache-2.0)
reads both records as text and outputs a match probability. It was fine-tuned on a free Kaggle GPU (1 epoch, max length 64)
on best-candidate pairs with stage-1 probability in [0.02, 0.995] from the held-out fold, in two folds that coincide with
stage-2 halves A and B, so half B's scores come from a model trained on half A only. On uncertain held-out pairs the AUC
rises from 0.9885 (stage 2) to 0.9969 (blend); the stage-2 score on the untouched half rises from 0.9813 to 0.9857.
The transformer never saw French text and is visibly unsure on French pairs (mean score 0.65 on pairs stage 1 is sure
about, vs 0.90–0.92 for US/India), so we first kept the v8 decision for France (`src/compose_country.py`, leaderboard 0.9799); using the transformer on France as well scored higher (0.9809), so the final file uses it everywhere. The standard drift gate
flags the score (PSI 1.35) because it is bimodal and quantile bins collapse; with fixed bins PSI is 0.092 on US + India
and 0.021 after matching the stage-1 mix (`src/drift_ce_check.py`).

**Overfitting audit (`src/overfit_check.py`).** Stage 1: hold-out log-loss falls monotonically with the number of trees
(0.0128 at 200 → 0.0112 at 1486); AUC on businesses never used in training 0.99980 vs 0.99984 on the early-stopping
hold-out. Stage 2: log-loss on the untouched half falls from 0.0345 (100 trees) to 0.0256 (1200) and never rises; for
every model the untouched-half score is equal to or above the tuning-half score (+0.0002 to +0.0005); F0.5 varies only
0.9798–0.9814 across thresholds 0.5–0.9. The validation–leaderboard gap is distribution shift, not overfitting.

**Further transformer experiments (27 Sept, all scored once on the untouched half B).** A second bge-reranker trained on
568k best-candidate pairs of the stage-1 *training* records (never a held-out S1; one model scores validation and test
alike, `src/ce2_prep.py`, `tests/test_leakage.py`) reached AUC 0.9921 and moved half B from 0.98571 to 0.98588 (v12).
A second architecture (xlm-roberta-base, AUC 0.9900) lowered it to 0.98577 and was rejected. A zero-shot LLM judge
(Qwen2.5-7B-Instruct-AWQ, Apache-2.0) scored AUC 0.4776 on the hard pairs: on this data, true matches under unrelated trade
names and near-identical decoys contradict general knowledge, so only models trained on our labels help; it was rejected.
Seed bagging, per-country thresholds and a rank-aware acceptance rule each changed half B by at most +0.00006 (noise)
and were not used. Remaining headroom is candidate recall (97.5%), which needs GPU dense/trigram retrieval.

**Where the remaining error lies (v2, oracle analysis on the held-out fold):**

| Scenario | F0.5 |
|---|---|
| v2 as submitted | 0.9777 |
| perfect accept/reject on our existing best candidates | 0.9894 |
| + perfect ranking of every blocked true match | 0.9916 |

Blocking recall on held-out true pairs is 97.45% (38,892 misses out of 1,526,382). Most of the remaining loss is in accept/reject decisions: 15,076 false accepts and 33,107 rejected true matches. Rejected true matches often have an empty address (23%) and a name shared by other S1 entities.

- **Common false positives (wrong merges):** in v1, 85% were *decoys*, near-copies of an S1 entity with a one-letter name change (`olaneon` vs `olaneus`, `kpf` vs `kvf`) or a changed number (`268 14` vs `268 1`, `46` vs `465 main st`), labelled as distinct businesses. v2's number-distance features and stage 2 target these.
- **Common false negatives (missed matches):** (a) empty address plus a name shared by several S1 entities, which is often impossible to resolve, so the model abstains on purpose; (b) heavy Hindi back-transliteration of English names; (c) records renamed to unrelated trade names with off-by-one house numbers; (d) typos in every token.

## 6. Conclusion
Treating the problem as constrained assignment, with high-recall multi-key blocking and a second stage that looks at the whole cluster, gave most of the gains. The largest single improvements came from compound blocking keys (+14 points of blocking recall) and cluster-aware verification. The main lesson: in entity resolution scored on precision, the structure of the data (one S1 per record, caps per source, coherence of each cluster) matters as much as the pairwise similarity model.

---

## Appendix

### A. Code Artefacts
Repository root (see also [REPRODUCTION.md](REPRODUCTION.md)):
- `src/normalize.py`: normalisation (anyascii transliteration, legal and DBA handling, US/India/France address abbreviations)
- `src/preprocess.py`: parallel normalisation to parquet
- `src/blocking.py`: multi-key inverted index with two-stage retrieval
- `src/features.py`: stage-1 pair features
- `src/pipeline.py`: `block` / `train` / `infer` / `evaluate` / `submit`
- `src/stage2.py`: cluster-aware verifier (`cv` / `fit` / `apply`), which writes both output files
- `src/metric.py`: macro F0.5
- `src/run_all.sh`: **entry point**, a resumable end-to-end driver; `bash src/run_all.sh` regenerates `output/matching_results.tsv` and `output/candidate_pairs.tsv`
- `src/error_analysis.py`, `src/dev_block.py`, `src/eda*.py`: analysis scripts

Environment: Python 3.11, polars 1.44.2, rapidfuzz 3.14.6, lightgbm 4.7.0 (MIT), anyascii 0.3.3 (ISC), numpy 1.26.4. CPU only (8 cores, 16 GB RAM). No external data, APIs or geocoding.

### B. Additional Results
Blocking recall progression (3% query sample, 309k S2/S3 records):

| Configuration | Recall@K | Recall@1 | Candidates / record |
|---|---|---|---|
| name tokens + skeleton + number×word, cap 500, 3 probes | 81.8% | 71.6% | 6.4 |
| + full-name, name×address, address bigrams, 2-stage rescoring | 95.8% | 92.8% | 7.4 |
| + phonetic codes, concatenated prefixes | 96.0% | 93.4% | 7.5 |
| + number×name-prefix, address pairs, 12 probes, top-8 | 97.5% | 95.5% | 8.4 |
