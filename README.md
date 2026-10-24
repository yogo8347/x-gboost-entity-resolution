# Business Entity Resolution at Scale: Amazon ML Challenge 2026

**Yojit Goyal** (entry: X_Gboost) · public leaderboard **0.9813** macro F0.5 · 12.5M training records, 11.7M test records · US, India and France (France unseen in training)

Given business records (name, address, country) from three sources that share no identifiers, find every Source 2 and Source 3 record that refers to the same real-world business as each Source 1 entity. Names and addresses are noisy: typos, leetspeak, legal-suffix swaps, DBA names, Devanagari / Tamil / Kannada scripts, English re-transliterated through Hindi (`gret indastrij` = *great industries*), and deliberate **decoys** that are near-copies of a real business with one detail changed.

Scoring is macro F0.5 per Source 1 entity, so precision counts twice as much as recall, and one false merge on an entity with no true match costs a full point for that entity.

## Results

| Version | What changed | Validation F0.5 | Public leaderboard |
|---|---|---|---|
| v2 | multi-key blocking + 69-feature LightGBM + cluster-aware stage 2 | 0.9777 | 0.963 |
| v3 | rich cluster features | 0.9805 | 0.910 ✗ (distribution drift, rejected) |
| v5 | drift-robust stage 2 + decoy markers + France address patch | 0.9805 | 0.971 |
| v7 | word markers, cleaned comparisons, French decoy words | 0.9811 | 0.973 |
| v8 | translation dictionary learned from training pairs | 0.9813 | 0.973 |
| v10 | + fine-tuned cross-encoder (`bge-reranker-v2-m3`) on uncertain pairs | 0.9857 | 0.9809 |
| **v12** | **+ second cross-encoder trained on stage-1 training-fold pairs** | **0.9859** | **0.9813** |

Validation is measured on an untouched half of a 20% held-out fold (441,410 entities) that no tuning decision ever saw.

## How it works

```
raw TSV ─► normalise ─► blocking ─► stage 1: pair model ─► cross-encoders ─► stage 2: verifier ─► threshold + caps ─► submission
           anyascii,    inverted    69 similarity           bge-reranker      best + runner-up,     0.85, ≤5 S2 / ≤6 S3
           legal forms, index,      features, LightGBM      on uncertain      cluster coherence,    per Source 1 entity
           addresses    ~7 cands                            pairs (GPU)       decoy markers
```

1. **Normalisation** (`src/normalize.py`): transliterates any script to Latin, splits legal forms and DBA / "formerly" names off a core name, and canonicalises address abbreviations for the US, India and France.
2. **Blocking** (`src/blocking.py`): an IDF-weighted inverted index over about ten key families (name tokens, consonant skeletons, phonetic codes, name × address word, house number × street, address bigrams, concatenated domain-style names), always prefixed by country. It keeps about 7 candidates per record at **97.5% recall**. Compound keys alone lifted recall from 81.8% to 95.8%.
3. **Stage 1** (`src/features.py`, `src/pipeline.py`): 69 country-agnostic pair features (fuzzy name and address scores, phonetic agreement, number distance, name rarity, competition with the other candidates), scored by LightGBM.
4. **Assignment, not pairwise matching**: every Source 2 or 3 record matches at most one Source 1 entity, so each record keeps only its best candidate and the model decides accept or reject.
5. **Stage 2** (`src/stage2v3.py`, `src/tune_stage2.py`): a second LightGBM sees the best and runner-up candidates, the coherence of the cluster each Source 1 entity is forming, and **decoy markers**. Examples: a legal-form conflict (Inc vs LLC) appears in 0.3% of true pairs but 13.9% of decoys, and a one-digit house-number change is about 8× more common in decoys.
6. **Cross-encoders** (`cloud_package*/`): `BAAI/bge-reranker-v2-m3` fine-tuned on free Kaggle T4 GPUs reads both records as text. Its score is added as a stage-2 feature and raises AUC on uncertain pairs from 0.9885 to 0.9969.

## What made the difference

- **Leak-free validation.** Our first version validated at 0.9651 but contained a leak, because training pairs could include held-out entities. `select_training_records()` now prevents it, and `tests/test_leakage.py` proves it with 36 checks.
- **A drift gate.** v3 validated at 0.9805 but scored 0.910, because its cluster features depended on the share of junk records, which is higher in test (37% vs 27%). Every model input must now pass PSI < 0.10 between validation and test (`src/drift_check.py`) before an upload.
- **Diagnostic uploads.** Emptying France measured France's F0.5 at about 0.94, and an all-empty file measured the share of entities with no match (0.056). Two uploads told us where the error was.
- **Error analysis over model tuning.** The biggest late gains came from reading the misses (decoy markers, a full vocabulary audit, the France patch), not from hyperparameter search.
- **Measured negatives.** A second transformer architecture (xlm-roberta-base), a zero-shot LLM judge (Qwen2.5-7B, AUC 0.48), seed bagging and per-country thresholds were each measured and rejected. They are all recorded in [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md).

## Repository layout

| Path | Contents |
|---|---|
| `src/` | the pipeline: normalisation, blocking, features, stage 1 and 2, markers, drift and overfitting checks, `run_*.sh` drivers for each version |
| `tests/test_leakage.py` | 36 leakage checks (`python -m pytest tests -v`) |
| `cloud_package*/` | Kaggle GPU notebooks for the cross-encoders (`cloud_package2` = v12) and the rejected experiments |
| `data/` | trained models: `model.txt` (stage 1), `model2v12_lgb.txt` + `stage2v12_final.json` (final stage 2) |
| `docs/SOLUTION_WRITEUP.md` | full technical write-up: analysis, methodology, results, error analysis |
| `docs/REPRODUCTION.md` | stage-by-stage commands and run times |
| `docs/PROJECT_GUIDE.md` | key data findings, architecture and file map |
| `docs/EXPERIMENT_LOG.md` | day-by-day log of every experiment and leaderboard upload |

## Running it

The challenge data is not included. Get it from the Amazon ML Challenge 2026 student kit and place it so that `dataset/train/*.tsv` and `dataset/test/*.tsv` sit next to `src/`.

```bash
pip install -r requirements.txt
python src/preprocess.py dataset train && python src/preprocess.py dataset test
bash src/run_all.sh                       # blocking, stage 1, held-out scoring, first stage 2 (v2)
python src/france_patch.py                # France address patch
bash src/run_v5.sh; bash src/run_v6.sh; bash src/run_v7.sh; bash src/run_v8.sh
```

The cross-encoder steps for v10 and v12 run on a GPU (Kaggle notebooks in `cloud_package/` and `cloud_package2/`). See [docs/REPRODUCTION.md](docs/REPRODUCTION.md) for the full sequence. Stage 1 and 2 run on a CPU and were developed on a 16 GB, 8-core laptop. A full run takes several hours.

## Author

Yojit Goyal ([@yogo8347](https://github.com/yogo8347))

## License

Code: MIT (see [LICENSE](LICENSE)). Libraries: LightGBM (MIT), XGBoost (Apache-2.0), Optuna (MIT), polars (MIT), rapidfuzz (MIT), anyascii (ISC). Model: `BAAI/bge-reranker-v2-m3` (Apache-2.0). The challenge dataset belongs to the organisers and is not redistributed here.
