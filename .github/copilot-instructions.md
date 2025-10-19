# Copilot instructions

## Architecture overview
- Ensemble fake-text detection for the ESA challenge: see `models/main.py` for multi-model orchestration and `scripts/main_submission_esa.py` for the unified pipeline used in practice.
- Data arrives as pair directories like `article_0001/file_1.txt`; `utils/data_loader.create_unified_dataloaders` converts them into single-text samples with binary labels and `sample_id` markers (`"{pair}_{1|2}"`).
- Feature extraction lives in `models/extractors.py` (layer-wise embeddings, pooling helpers) and `models/text_features.py` (perplexity + intrinsic dimension calculators). Classifiers and trajectory metrics are implemented in `models/classifiers.py`.
- Outputs are reconciled back to Kaggle submissions via `utils/data_loader.reconstruct_pairs_from_predictions`, and helper scripts drop CSVs into the repo root or `scripts/results/`.

## Core workflows
- Unified training loop: call `create_unified_dataloaders(...)`, flatten batches with `extract_training_data`, and keep the per-text fake probability convention (0 = real, 1 = fake).
- Embedding path: instantiate `EmbeddingExtractor(model_id, device)`, run `get_all_layer_embeddings` (returns a list of dicts keyed by layer), then produce `(n_texts, hidden)` arrays with `pool_embeds_from_layer` before feeding them into a classifier.
- Binary detectors: `BinaryDetector.fit(...)` supports `classifier_type` values `svm`, `lr`, `xgb`, and `neural` (the last one wraps `DeepBinaryDetector` and requires CUDA). `predict` yields `(preds, probs, distances)` when `return_probabilities`/`return_distances` are true; most scripts only keep the fake probabilities.
- Legacy pairwise path: `OutlierDetections` (unsupervised fake spotting) and `TrajectoryClassifier` (token-level angle features) still power ensemble experiments in `models/main.py`; both expect precomputed text lists and an `EmbeddingExtractor` instance.
- Additional features: `PerplexityCalculator` and `TextIntrinsicDimensionCalculator` load heavy HuggingFace models in `float16`. They assume GPU by default, so set `device="cpu"` explicitly for small tests.

## Running and tooling
- Install with `pip install -e .[dev]`; packaging metadata still references `scripts/main.py`, so prefer direct module execution (`python scripts/main_submission_esa.py ...`).
- Primary entry point example: `python scripts/main_submission_esa.py --model_name Qwen/Qwen2.5-0.5B --analysis_type embedding --layer 22 --train_path data/data_esa/train --train_labels_path data/data_esa/train.csv --test_path data/data_esa/test`.
- Batch configs live in `scripts/run_configs.yaml`; run them via `python scripts/run_experiment.py embedding_baseline` (logs land under `scripts/logs/`).
- Reconstruct Kaggle submissions with `reconstruct_pairs_from_predictions`; saved CSVs follow `submission_{model}_{layer}.csv`. Keep the lower-fake-probability → real-text convention intact.
- GPU hygiene matters: `clear_gpu_memory()` wraps `torch.cuda.empty_cache()` + `gc.collect()`, and most scripts default to `batch_size=8` to stay within the VRAM budget.

## Conventions & gotchas
- Always sanitize blank strings to a single space before encoding (see `get_features`); the HuggingFace tokenizers crash on empty input.
- Preserve DataLoader batch keys (`text`, `label`, `sample_id`, metadata lengths) when extending loaders so downstream helpers keep working.
- HuggingFace models download on demand; set `LOCAL_RANK`/`CUDA_VISIBLE_DEVICES` or pass `device` explicitly when running on shared clusters.
- Minimal regression coverage exists (`test_pooling.py`); run `pytest test_pooling.py` after touching embedding utilities.
- Large artifacts (`out_llms/`, data dumps) are ignored by default—write new outputs to that folder or `scripts/results/` to avoid polluting the repo root.
