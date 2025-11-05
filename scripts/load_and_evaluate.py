#!/usr/bin/env python3
"""
Script to load saved detectors and evaluate them on different datasets for cross-dataset evaluation.
"""

import sys
import gc
import argparse
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Ensure project modules are discoverable
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from models.extractors import EmbeddingExtractor, TFIDFExtractor, pool_embeds_from_layer
from models.text_features import PerplexityCalculator, TextIntrinsicDimensionCalculator
from models.classifiers import BinaryDetector


def clear_gpu_memory():
    """Utility to clear GPU caches when working with large HF models.

    Safe on CPU-only environments (no-op for CUDA parts when unavailable).
    """
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
    except Exception:
        # torch may be installed without CUDA; ignore
        pass
    gc.collect()


def sanitize_texts(texts: List[str]) -> List[str]:
    """Ensure every text is a non-empty string accepted by tokenizers."""
    return [t.strip() if isinstance(t, str) and t.strip() else " " for t in texts]


def replace_nan_with_column_means(features: np.ndarray) -> np.ndarray:
    """Replace non-finite values (NaN/Inf) with column means; fallback to zeros.

    If a scipy.sparse matrix is provided (as in TF-IDF), return unchanged since
    sparse matrices typically don't contain NaNs and require different handling.
    """
    try:
        from scipy.sparse import issparse  # type: ignore
        if issparse(features):
            return features
    except Exception:
        pass

    arr = np.asarray(features, dtype=np.float32)
    if np.isfinite(arr).all():
        return arr
    finite_mask = np.isfinite(arr)
    arr_masked = arr.copy()
    arr_masked[~finite_mask] = np.nan
    col_means = np.nanmean(arr_masked, axis=0)
    if np.isscalar(col_means):
        col_means = np.array([col_means])
    col_means = np.nan_to_num(col_means, nan=0.0)
    bad_rows, bad_cols = np.where(~finite_mask)
    if bad_rows.size > 0:
        arr[bad_rows, bad_cols] = col_means[bad_cols]
    return arr


def load_saved_detector(model_path: str) -> Tuple[BinaryDetector, Dict[str, Any]]:
    """Load a saved detector and its metadata."""
    model_path = Path(model_path)
    
    # Load the detector
    with open(model_path, 'rb') as f:
        detector = pickle.load(f)
    
    # Load metadata if it exists
    metadata_path = model_path.parent / f"{model_path.stem}_metadata.pkl"
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
    
    # Attach model_path for downstream utilities (e.g., loading TF-IDF vectorizer)
    metadata['model_path'] = str(model_path)
    return detector, metadata


def instantiate_extractor_from_metadata(metadata: Dict[str, Any], device: str = "cuda:0", model_path: str | None = None):
    """Recreate the feature extractor from saved metadata."""
    analysis_type = metadata.get('analysis_type', 'embedding')
    model_name = metadata.get('model_name', 'Qwen/Qwen2.5-0.5B')
    layer = metadata.get('layer', 22)
    
    if analysis_type == "embedding":
        return EmbeddingExtractor(model_name, device=device)
    elif analysis_type == "perplexity":
        return PerplexityCalculator(model_name, device=device)
    elif analysis_type == "phd":
        return TextIntrinsicDimensionCalculator(
            model_name,
            device=device,
            layer_idx=layer,
        )
    elif analysis_type == "tfidf":
        # Load the fitted TF-IDF vectorizer saved alongside the detector
        if model_path is None:
            raise ValueError("model_path is required to load TF-IDF vectorizer for evaluation")
        vec_path = Path(model_path).with_name(Path(model_path).stem + "_vectorizer.pkl")
        if not vec_path.exists():
            raise FileNotFoundError(f"Expected TF-IDF vectorizer file not found: {vec_path}")
        import pickle as _pkl
        with open(vec_path, 'rb') as f:
            fitted_vec = _pkl.load(f)
        extractor = TFIDFExtractor(
            max_features=getattr(fitted_vec, 'max_features', None),
            ngram_range=getattr(fitted_vec, 'ngram_range', (1, 2)),
            lowercase=getattr(fitted_vec, 'lowercase', True),
            min_df=getattr(fitted_vec, 'min_df', 1),
            max_df=getattr(fitted_vec, 'max_df', 1.0),
            use_idf=getattr(fitted_vec, 'use_idf', True),
            norm=getattr(fitted_vec, 'norm', 'l2'),
            sublinear_tf=getattr(fitted_vec, 'sublinear_tf', False),
            stop_words=getattr(fitted_vec, 'stop_words', None),
        )
        # Replace with fitted vectorizer and mark as fitted
        extractor.vectorizer = fitted_vec
        extractor._is_fitted = True
        return extractor
    else:
        raise ValueError(f"Unsupported analysis_type {analysis_type}")


def get_features_from_metadata(texts: List[str], extractor, metadata: Dict[str, Any], 
                              batch_size: int = 8, max_length: int = 512, 
                              show_progress: bool = True) -> np.ndarray:
    """Extract features using the same parameters as during training."""
    processed_texts = sanitize_texts(texts)
    analysis_type = metadata.get('analysis_type', 'embedding')
    layer = metadata.get('layer', 22)
    pooling = metadata.get('pooling', 'mean')
    normalize = bool(metadata.get('normalize', False))
    
    print(f"Extracting {analysis_type} features for {len(processed_texts)} texts...")
    
    if analysis_type == "embedding":
        embeds_all = extractor.get_all_layer_embeddings(
            processed_texts,
            batch_size=batch_size,
            max_length=max_length,
            show_progress=show_progress,
        )
        # Be robust to layer index if not available
        available_layers = sorted(list(embeds_all[0].keys())) if embeds_all else []
        chosen_layer = layer
        if chosen_layer not in available_layers and available_layers:
            fallback = available_layers[-1]
            print(f"⚠️  Requested layer {chosen_layer} not available. Using last available layer {fallback}.")
            chosen_layer = fallback
        layer_embeds = [embeds[chosen_layer] for embeds in embeds_all]
        if normalize:
            layer_embeds = [F.normalize(torch.from_numpy(e), p=2, dim=1).numpy() for e in layer_embeds]
        features = pool_embeds_from_layer(layer_embeds, pooling=pooling)

    elif analysis_type == "perplexity":
        features = np.array(
            extractor.calculate_batch_perplexity(processed_texts, max_length=max_length)
        ).reshape(-1, 1)

    elif analysis_type == "phd":
        features = np.array(
            extractor.calculate_batch(processed_texts, max_length=max_length)
        ).reshape(-1, 1)

    elif analysis_type == "tfidf":
        # Use the fitted vectorizer to transform into sparse features
        # Keep sparse to let BinaryDetector apply SVD/MaxAbsScaler as trained
        features = extractor.transform(processed_texts, dense=False)

    else:
        raise ValueError(f"Unknown analysis_type: {analysis_type}")

    return features


def load_mercor_ai_dataset(data_path: str) -> Tuple[List[str], np.ndarray]:
    """Load the Mercor AI dataset."""
    df = pd.read_csv(data_path)
    texts = sanitize_texts(df['answer'].astype(str).tolist())
    labels = df['is_cheating'].astype(int).to_numpy()  # 0=human, 1=AI
    return texts, labels


def load_human_ai_dataset(data_path: str) -> Tuple[List[str], np.ndarray]:
    """Load the AI_Human.csv dataset."""
    df = pd.read_csv(data_path)
    texts = sanitize_texts(df['text'].astype(str).tolist())
    labels = df['generated'].astype(int).to_numpy()  # 0=human, 1=AI
    return texts, labels


def load_dataset(dataset_name: str, data_path: str, text_col: str = None, label_col: str = None) -> Tuple[List[str], np.ndarray]:
    """Load dataset based on its name and format."""
    if dataset_name == "mercor_ai":
        return load_mercor_ai_dataset(data_path)
    elif dataset_name == "human_ai":
        return load_human_ai_dataset(data_path)
    else:
        # Generic CSV loader
        df = pd.read_csv(data_path)
        if text_col and label_col:
            texts = sanitize_texts(df[text_col].astype(str).tolist())
            labels = df[label_col].astype(int).to_numpy()
        else:
            # Try to infer columns
            possible_text_cols = ['text', 'answer', 'content', 'passage']
            possible_label_cols = ['label', 'is_cheating', 'generated', 'target']
            
            text_col = None
            for col in possible_text_cols:
                if col in df.columns:
                    text_col = col
                    break
            
            label_col = None
            for col in possible_label_cols:
                if col in df.columns:
                    label_col = col
                    break
            
            if not text_col or not label_col:
                raise ValueError(f"Could not infer text and label columns from {list(df.columns)}")
            
            texts = sanitize_texts(df[text_col].astype(str).tolist())
            labels = df[label_col].astype(int).to_numpy()
        
        return texts, labels


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> Dict[str, float]:
    """Compute comprehensive evaluation metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='binary', zero_division=0),
    }
    
    if y_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics['roc_auc'] = float('nan')
    
    return metrics


def _sweep_thresholds(y_true: np.ndarray, y_proba: np.ndarray, metric: str = "f1") -> Tuple[float, float]:
    """Return (best_threshold, best_metric_value) by sweeping thresholds in [0,1].

    Currently supports metric="f1" only.
    """
    if y_proba is None:
        return 0.5, float('nan')

    thresholds = np.linspace(0.0, 1.0, 101)
    best_t = 0.5
    best_val = -1.0
    for t in thresholds:
        pred = (y_proba >= t).astype(int)
        if metric == "f1":
            val = f1_score(y_true, pred, average='binary', zero_division=0)
        else:
            # default fallback
            val = f1_score(y_true, pred, average='binary', zero_division=0)
        if val > best_val:
            best_val = val
            best_t = t
    return float(best_t), float(best_val)


def evaluate_detector_on_dataset(detector: BinaryDetector, metadata: Dict[str, Any], 
                                texts: List[str], labels: np.ndarray, 
                                device: str = "cuda:0", batch_size: int = 8, 
                                max_length: int = 512,
                                threshold: float | None = None,
                                optimize_threshold: str | None = None,
                                optimize_split: float = 0.2,
                                random_state: int = 42) -> Dict[str, Any]:
    """Evaluate a saved detector on a dataset."""
    print("Creating feature extractor...")
    extractor = instantiate_extractor_from_metadata(metadata, device=device, model_path=metadata.get('model_path'))
    
    # Extract features
    features = get_features_from_metadata(
        texts, extractor, metadata, batch_size=batch_size, 
        max_length=max_length, show_progress=True
    )
    features = replace_nan_with_column_means(features)
    
    # Make predictions
    print("Making predictions...")
    results = detector.predict(
        features, return_probabilities=True, return_distances=True
    )
    
    # Unpack results based on what was returned
    if isinstance(results, list) and len(results) >= 2:
        predictions = results[0]
        probabilities = results[1]
        distances = results[2] if len(results) > 2 else None
    else:
        predictions = results
        probabilities = None
        distances = None
    
    # Handle probabilities format
    if probabilities is not None:
        if probabilities.ndim == 1:
            # Already probability of fake class
            prob_fake = probabilities
        elif probabilities.shape[1] == 2:
            # Binary classification probabilities [prob_real, prob_fake]
            prob_fake = probabilities[:, 1]
        else:
            prob_fake = probabilities.flatten()
    else:
        prob_fake = None
    
    applied_threshold = None
    optimized_info: Dict[str, Any] = {}

    # Apply manual or optimized thresholding if probabilities are available
    if prob_fake is not None:
        if optimize_threshold is not None:
            # Option 1: small validation split to find threshold, then apply to full set
            try:
                X_idx = np.arange(len(labels))
                # ensure stratify only when both classes exist
                stratify = labels if len(np.unique(labels)) > 1 else None
                idx_train, idx_val = train_test_split(
                    X_idx, test_size=optimize_split, random_state=random_state, stratify=stratify
                ) if 0 < optimize_split < 1.0 and len(labels) > 10 else (X_idx, [])
                if len(idx_val) > 0:
                    best_t, best_val = _sweep_thresholds(labels[idx_val], prob_fake[idx_val], metric=optimize_threshold)
                else:
                    best_t, best_val = _sweep_thresholds(labels, prob_fake, metric=optimize_threshold)
                applied_threshold = best_t
                predictions = (prob_fake >= applied_threshold).astype(int)
                optimized_info = {
                    'optimize_metric': optimize_threshold,
                    'optimize_split': float(optimize_split),
                    'chosen_threshold': float(best_t),
                    'chosen_metric_value': float(best_val),
                }
            except Exception:
                # Fallback: ignore optimization on error
                pass

        if threshold is not None and applied_threshold is None:
            applied_threshold = float(threshold)
            predictions = (prob_fake >= applied_threshold).astype(int)

    # Compute metrics
    metrics = compute_metrics(labels, predictions, prob_fake)
    if applied_threshold is not None:
        metrics['applied_threshold'] = float(applied_threshold)
    if optimized_info:
        metrics.update(optimized_info)
    
    # Clean up extractor
    del extractor
    clear_gpu_memory()
    
    return {
        'metrics': metrics,
        'predictions': predictions,
        'probabilities': probabilities,
        'features_shape': features.shape,
        'metadata': metadata
    }


def print_evaluation_results(results: Dict[str, Any], model_name: str, dataset_name: str):
    """Print formatted evaluation results."""
    metrics = results['metrics']
    metadata = results['metadata']
    
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Training Dataset: {metadata.get('dataset_used', 'Unknown')}")
    print(f"Analysis Type: {metadata.get('analysis_type', 'Unknown')}")
    print(f"Classifier: {metadata.get('classifier_type', 'Unknown')}")
    print(f"Features Shape: {results['features_shape']}")
    print(f"\nMetrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    if 'roc_auc' in metrics and not np.isnan(metrics['roc_auc']):
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    # Threshold info if available
    if 'applied_threshold' in metrics:
        print(f"  Threshold: {metrics['applied_threshold']:.3f}")
    if 'chosen_threshold' in metrics:
        print(f"  Best Threshold (opt): {metrics['chosen_threshold']:.3f} ({metrics.get('optimize_metric','')}: {metrics.get('chosen_metric_value', float('nan')):.4f})")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Load saved detector and evaluate on dataset")
    
    # Model loading
    parser.add_argument("--model_path", type=str, required=True, 
                       help="Path to saved detector (.pkl file)")
    
    # Dataset configuration
    parser.add_argument("--dataset_name", type=str, required=True, 
                       choices=["mercor_ai", "human_ai", "generic"],
                       help="Dataset type to load")
    parser.add_argument("--data_path", type=str, required=True, 
                       help="Path to evaluation dataset CSV")
    parser.add_argument("--text_column", type=str, default=None, 
                       help="Name of text column (for generic datasets)")
    parser.add_argument("--label_column", type=str, default=None, 
                       help="Name of label column (for generic datasets)")
    
    # Inference parameters
    parser.add_argument("--device", type=str, default="cuda:0", 
                       help="Device to run inference on")
    parser.add_argument("--batch_size", type=int, default=8, 
                       help="Batch size for feature extraction")
    parser.add_argument("--max_length", type=int, default=512, 
                       help="Maximum sequence length")

    # Thresholding options
    parser.add_argument("--threshold", type=float, default=None,
                       help="Override decision threshold on P(fake); if set, predictions = (P(fake) >= threshold)")
    parser.add_argument("--optimize_threshold", type=str, default=None, choices=["f1"],
                       help="Optimize a threshold on a validation split of the evaluation set using the given metric (e.g., 'f1')")
    parser.add_argument("--optimize_split", type=float, default=0.2,
                       help="Fraction of eval set used as validation to select the threshold when --optimize_threshold is set")
    parser.add_argument("--random_state", type=int, default=42,
                       help="Random seed for threshold optimization split")
    
    # Output options
    parser.add_argument("--save_predictions", action="store_true", 
                       help="Save predictions to CSV file")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", 
                       help="Directory to save results")
    
    args = parser.parse_args()
    
    # Load detector and metadata
    print(f"Loading detector from {args.model_path}...")
    detector, metadata = load_saved_detector(args.model_path)
    
    # Load evaluation dataset
    print(f"Loading {args.dataset_name} dataset from {args.data_path}...")
    texts, labels = load_dataset(args.dataset_name, args.data_path, 
                                args.text_column, args.label_column)
    
    print(f"Loaded {len(texts)} samples")
    print(f"Label distribution: {np.bincount(labels)} (0=real, 1=fake)")
    
    # Evaluate
    results = evaluate_detector_on_dataset(
        detector, metadata, texts, labels, 
        device=args.device, batch_size=args.batch_size, 
        max_length=args.max_length,
        threshold=args.threshold,
        optimize_threshold=args.optimize_threshold,
        optimize_split=args.optimize_split,
        random_state=args.random_state
    )
    
    # Print results
    model_name = Path(args.model_path).stem
    print_evaluation_results(results, model_name, args.dataset_name)
    
    # Save results if requested
    if args.save_predictions:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save predictions CSV
        prob_fake = None
        if results.get('probabilities') is not None:
            probs_np = np.asarray(results['probabilities'])
            if probs_np.ndim == 1:
                prob_fake = probs_np
            elif probs_np.ndim == 2:
                if probs_np.shape[1] == 2:
                    prob_fake = probs_np[:, 1]
                elif probs_np.shape[1] == 1:
                    prob_fake = probs_np[:, 0]
                else:
                    prob_fake = probs_np.ravel()
            else:
                prob_fake = probs_np.ravel()

        data_dict = {
            'prediction': results['predictions'],
            'true_label': labels
        }
        if prob_fake is not None:
            data_dict['probability_fake'] = prob_fake
        
        pred_df = pd.DataFrame(data_dict)
        pred_file = output_dir / f"predictions_{model_name}_{args.dataset_name}.csv"
        pred_df.to_csv(pred_file, index=False)
        
        # Save metrics JSON
        import json
        metrics_file = output_dir / f"metrics_{model_name}_{args.dataset_name}.json"
        with open(metrics_file, 'w') as f:
            json.dump(results['metrics'], f, indent=2)
        
        print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()