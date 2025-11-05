import sys
import gc
import argparse
import pickle
from pathlib import Path
from typing import List, Tuple
import re

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import torch.nn.functional as F

# Ensure project modules are discoverable
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from models.extractors import EmbeddingExtractor, pool_embeds_from_layer, TFIDFExtractor
from models.text_features import PerplexityCalculator, TextIntrinsicDimensionCalculator
from models.classifiers import BinaryDetector, OutlierDetections
import torch.nn.functional as F


def clear_gpu_memory():
    """Utility to clear GPU caches when working with large HF models."""
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()


def sanitize_texts(texts: List[str]) -> List[str]:
    """Ensure every text is a non-empty string accepted by tokenizers."""
    return [t.strip() if isinstance(t, str) and t.strip() else " " for t in texts]


def replace_nan_with_column_means(features: np.ndarray) -> np.ndarray:
    """Replace non-finite values (NaN/Inf) column-wise with column means; fallback to zeros.

    Handles both dense numpy arrays and sparse matrices. For sparse inputs, TF-IDF
    typically contains no NaNs and scipy.sparse doesn't support NaN ops; we simply
    return the matrix unchanged.
    """
    # Gracefully skip sparse inputs
    try:
        from scipy.sparse import issparse  # type: ignore
        if issparse(features):
            return features
    except Exception:
        pass

    arr = np.asarray(features, dtype=np.float32)

    # If no non-finite values, return as-is
    if np.isfinite(arr).all():
        return arr

    # Compute column means over finite values only
    finite_mask = np.isfinite(arr)
    # Avoid empty slices by setting invalid to NaN then using nanmean
    arr_masked = arr.copy()
    arr_masked[~finite_mask] = np.nan
    col_means = np.nanmean(arr_masked, axis=0)
    if np.isscalar(col_means):  # handle 1D features (shape: (n,))
        col_means = np.array([col_means])
    col_means = np.nan_to_num(col_means, nan=0.0)

    # Replace NaN/Inf with column means
    bad_rows, bad_cols = np.where(~finite_mask)
    if bad_rows.size > 0:
        arr[bad_rows, bad_cols] = col_means[bad_cols]
    return arr


def get_features(texts: List[str], extractor, args, show_progress: bool = True):
    """Run the selected feature extractor on a list of texts.

    Memory efficient embedding path: if args.memory_efficient is True and analysis_type=embedding,
    we call extractor.get_pooled_layer_embeddings to avoid materializing every layer & sequence.
    Normalization can now be applied directly in the memory-efficient path.
    """
    processed_texts = sanitize_texts(texts)
    print(f"Extracting features for {len(processed_texts)} texts...")

    if args.analysis_type == "embedding":
        if getattr(args, 'memory_efficient', False):
            # Memory-efficient path: can apply normalization directly
            normalize_flag = getattr(args, 'normalize', False)
            print(f"Using memory-efficient single-layer pooled extraction path" + 
                  (f" with L2 normalization" if normalize_flag else "."))
            features = extractor.get_pooled_layer_embeddings(
                processed_texts,
                layer_idx=args.layer,
                pooling=args.pooling,
                batch_size=args.batch_size,
                max_length=args.max_length,
                show_progress=show_progress,
                normalize=normalize_flag,
            )
        else:
            # Full embeddings path (loads all layers, more memory-intensive)
            embeds_all = extractor.get_all_layer_embeddings(
                processed_texts,
                batch_size=args.batch_size,
                max_length=args.max_length,
                show_progress=show_progress,
            )
            available_layers = sorted(list(embeds_all[0].keys())) if embeds_all else []
            chosen_layer = args.layer
            if chosen_layer not in available_layers:
                if available_layers:
                    fallback = available_layers[-1]
                    print(f"⚠️  Requested layer {chosen_layer} not available. Using last available layer {fallback}.")
                    chosen_layer = fallback
                    try:
                        args.layer = chosen_layer
                    except Exception:
                        pass
                else:
                    raise ValueError("No layers available in embeddings.")

            layer_embeds = [embeds[chosen_layer] for embeds in embeds_all]
            if getattr(args, 'normalize', False):
                layer_embeds = [F.normalize(torch.from_numpy(e), p=2, dim=1).numpy() for e in layer_embeds]
            features = pool_embeds_from_layer(layer_embeds, pooling=args.pooling)

    elif args.analysis_type == "perplexity":
        features = np.array(
            extractor.calculate_batch_perplexity(processed_texts, max_length=args.max_length)
        ).reshape(-1, 1)

    elif args.analysis_type == "phd":
        features = np.array(
            extractor.calculate_batch(processed_texts, max_length=args.max_length)
        ).reshape(-1, 1)

    elif args.analysis_type == "tfidf":
        dense = bool(getattr(args, 'tfidf_dense', False))
        features = extractor.fit_transform(processed_texts, dense=dense)

    else:
        raise ValueError(f"Unknown analysis_type: {args.analysis_type}")

    return features


def instantiate_extractor(args):
    """Factory that returns the appropriate feature extractor."""
    if args.analysis_type == "embedding":
        return EmbeddingExtractor(
            args.model_name,
            device=args.device,
            log_memory=getattr(args, 'log_memory', False),
            memory_interval=getattr(args, 'memory_log_interval', 1)
        )
    if args.analysis_type == "perplexity":
        return PerplexityCalculator(args.model_name, device=args.device)
    if args.analysis_type == "phd":
        return TextIntrinsicDimensionCalculator(
            args.model_name,
            device=args.device,
            layer_idx=args.layer,
        )
    if args.analysis_type == "tfidf":
        return TFIDFExtractor(
            max_features=getattr(args, 'tfidf_max_features', 20000),
            ngram_range=(getattr(args, 'tfidf_ngram_min', 1), getattr(args, 'tfidf_ngram_max', 2)),
            lowercase=True,
            min_df=getattr(args, 'tfidf_min_df', 1),
            max_df=getattr(args, 'tfidf_max_df', 1.0),
            use_idf=True,
            norm="l2",
            sublinear_tf=False,
            stop_words=getattr(args, 'tfidf_stop_words', None),
        )
    raise ValueError(f"Unsupported analysis_type {args.analysis_type}")


def generate_model_name(args, dataset_name: str) -> str:
    """Generate standardized, filesystem-safe model filename."""
    def sanitize_for_filename(s: str) -> str:
        # Replace any character that's not alphanumeric, dash, underscore, or dot with underscore
        s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))
        # Collapse repeated underscores and trim leading/trailing separators
        s = re.sub(r"_+", "_", s).strip("._-")
        return s or "model"

    if args.analysis_type == "embedding":
        norm_tag = "_l2norm" if getattr(args, 'normalize', False) else ""
        feature_type = f"embedding_layer{args.layer}_{args.pooling}{norm_tag}"
    elif args.analysis_type == "perplexity":
        feature_type = "perplexity"
    elif args.analysis_type == "phd":
        feature_type = f"phd_layer{args.layer}"
    else:
        feature_type = args.analysis_type

    safe_dataset = sanitize_for_filename(dataset_name)
    safe_model = sanitize_for_filename(getattr(args, 'model_name', 'model'))
    safe_feature = sanitize_for_filename(feature_type)
    safe_classifier = sanitize_for_filename(getattr(args, 'classifier_type', 'clf'))

    return f"{safe_dataset}_{safe_model}_{safe_feature}_{safe_classifier}"


def save_detector_and_metadata(detector: BinaryDetector, args, save_dir: Path, model_name: str, extractor=None):
    """Save the trained detector and metadata for later use."""
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the detector
    detector_path = save_dir / f"{model_name}.pkl"
    detector_path.parent.mkdir(parents=True, exist_ok=True)
    with open(detector_path, 'wb') as f:
        pickle.dump(detector, f)
    
    # Save metadata for reproducibility
    metadata = {
        'model_name': args.model_name,
        'analysis_type': args.analysis_type,
        'classifier_type': args.classifier_type,
        'layer': getattr(args, 'layer', None),
        'pooling': getattr(args, 'pooling', None),
        'normalize': getattr(args, 'normalize', False),
        'batch_size': args.batch_size,
        'max_length': args.max_length,
        'validation_split': args.validation_split,
        'dataset_used': args.dataset_name,
        'text_column': args.text_column,
        'label_column': args.label_column,
    }

    if args.analysis_type == "tfidf":
        metadata.update({
            'tfidf_max_features': getattr(args, 'tfidf_max_features', None),
            'tfidf_ngram_min': getattr(args, 'tfidf_ngram_min', None),
            'tfidf_ngram_max': getattr(args, 'tfidf_ngram_max', None),
            'tfidf_min_df': getattr(args, 'tfidf_min_df', None),
            'tfidf_max_df': getattr(args, 'tfidf_max_df', None),
            'tfidf_dense': getattr(args, 'tfidf_dense', False),
            'svd_components': getattr(args, 'svd_components', None),
        })
    
    metadata_path = save_dir / f"{model_name}_metadata.pkl"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"✅ Saved detector to {detector_path}")
    print(f"✅ Saved metadata to {metadata_path}")

    # Save fitted TF-IDF vectorizer alongside the detector when applicable
    if getattr(args, 'analysis_type', None) == 'tfidf' and extractor is not None:
        try:
            vec = getattr(extractor, 'vectorizer', None)
            is_fitted = getattr(extractor, '_is_fitted', False)
            if vec is not None and is_fitted:
                vec_path = save_dir / f"{model_name}_vectorizer.pkl"
                with open(vec_path, 'wb') as f:
                    pickle.dump(vec, f)
                print(f"✅ Saved TF-IDF vectorizer to {vec_path}")
        except Exception as e:
            print(f"⚠️  Warning: failed to save TF-IDF vectorizer: {e}")

def load_human_ai_dataset(data_path: str, n_rows: int = None, stratified: bool = False, random_state: int = 42) -> Tuple[List[str], np.ndarray]:
    """Load Human vs AI dataset, optionally with balanced class sampling.
    
    Args:
        data_path: Path to the CSV file
        n_rows: Maximum rows to load
        stratified: If True and n_rows specified, sample n_rows/2 from each class (0/1)
        random_state: Random seed for stratified sampling
    
    Returns:
        Tuple of (texts list, labels numpy array)
    """
    df = pd.read_csv(data_path)
    
    if stratified and n_rows:
        # Sample n_rows/2 from each class (0 and 1) for balanced 50/50
        n_per_class = n_rows // 2
        df_0 = df[df['generated'] == 0].sample(n=min(n_per_class, len(df[df['generated'] == 0])), random_state=random_state)
        df_1 = df[df['generated'] == 1].sample(n=min(n_per_class, len(df[df['generated'] == 1])), random_state=random_state)
        df = pd.concat([df_0, df_1], ignore_index=True)
        print(f"Loaded {len(df)} samples (stratified: {len(df_0)} real + {len(df_1)} fake)")
    elif n_rows:
        df = df.head(n_rows)
    
    texts = df['text'].tolist()
    labels = df['generated'].values
    return texts, labels


def _resolve_daigtv2_csv_path(data_path: str) -> Path:
    """Resolve a DAIGT v2 CSV path.

    Accepts either a direct CSV path or a directory containing the CSV.
    Prefers files like 'train_v2_*.csv' when multiple are present.
    """
    p = Path(data_path)
    if p.is_file() and p.suffix.lower() == ".csv":
        return p
    if p.is_dir():
        # Prefer train_v2_* pattern; otherwise any CSV
        candidates = sorted(p.glob("train_v2_*.csv"))
        if not candidates:
            candidates = sorted(p.glob("*.csv"))
        if candidates:
            return candidates[0]
    raise FileNotFoundError(f"Could not find DAIGT v2 CSV at {data_path}")


def load_daigtv2_dataset(data_path: str, n_rows: int = None, stratified: bool = False, random_state: int = 42) -> Tuple[List[str], np.ndarray]:
    """Load the DAIGT v2 dataset.

    Expected columns: 'text' (string), 'label' (0/1). Additional columns are ignored.
    The function accepts a direct CSV path or a directory containing the CSV.
    
    Args:
        data_path: Path to CSV or directory containing the CSV
        n_rows: Maximum rows to load
        stratified: If True and n_rows specified, sample n_rows/2 from each class (0/1)
        random_state: Random seed for stratified sampling
    
    Returns:
        Tuple of (texts list, labels numpy array)
    """
    csv_path = _resolve_daigtv2_csv_path(data_path)
    df = pd.read_csv(csv_path, nrows=n_rows)
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError(
            f"DAIGT v2 CSV must contain 'text' and 'label' columns. Found: {list(df.columns)}"
        )
    
    # Coerce label to int {0,1} early so we can stratify
    labels_col = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)
    df['_label_int'] = labels_col
    
    if stratified and n_rows:
        # Sample n_rows/2 from each class for balanced 50/50
        n_per_class = n_rows // 2
        df_0 = df[df['_label_int'] == 0].sample(n=min(n_per_class, len(df[df['_label_int'] == 0])), random_state=random_state)
        df_1 = df[df['_label_int'] == 1].sample(n=min(n_per_class, len(df[df['_label_int'] == 1])), random_state=random_state)
        df = pd.concat([df_0, df_1], ignore_index=True)
        print(f"Loaded {len(df)} samples (stratified: {len(df_0)} class-0 + {len(df_1)} class-1)")
    
    texts = sanitize_texts(df['text'].astype(str).tolist())
    # Some DAIGT releases may have label as bool/str; coerce to int {0,1}
    labels = df['_label_int'].to_numpy()
    return texts, labels


def train_detector(train_texts: List[str], train_labels: np.ndarray, args) -> Tuple[object, object]:
    """Extract features for training data and fit the BinaryDetector."""
    print("Extracting training features...")
    extractor = instantiate_extractor(args)
    train_features = get_features(train_texts, extractor, args)
    train_features = replace_nan_with_column_means(train_features)

    print(f"Training features shape: {train_features.shape}")
    # Dimensionality reduction strategy per analysis type
    if args.analysis_type == "embedding":
        n_components = 0.95  # PCA keep 95% variance
    elif args.analysis_type == "tfidf":
        n_components = getattr(args, 'svd_components', None)  # int or None
    else:
        n_components = None
    input_dim = train_features.shape[1]

    outlier_types = {"elliptic", "ocsvm", "iforest"}
    if args.classifier_type in outlier_types:
        # For TF-IDF + one-class, ensure dense features (StandardScaler/PCA expect dense)
        try:
            from scipy.sparse import issparse
            if args.analysis_type == "tfidf" and 'issparse' in globals() and issparse(train_features):
                train_features = train_features.toarray().astype(np.float32)
        except Exception:
            pass
        # Use OutlierDetections (one-class style) with its own PCA pipeline
        detector = OutlierDetections(
            detector_type=args.classifier_type,
            contamination=0.1,
            random_state=42,
            n_components=(0.95 if args.analysis_type == "embedding" else 0.95),
        )
        training_results = detector.fit(
            embeddings=train_features,
            labels=train_labels,
            validation_split=args.validation_split,
        )
    else:
        # Standard binary path
        detector = BinaryDetector(
            n_components=n_components,
            contamination=0.1,
            random_state=42,
            input_dim=None if n_components is not None else input_dim,
        )

        training_results = detector.fit(
            embeddings=train_features,
            labels=train_labels,
            validation_split=args.validation_split,
            classifier_type=args.classifier_type,
            pca=(n_components is not None),
        )

    print(f"Training completed. Validation accuracy: {training_results.get('val_accuracy', 'N/A')}")
    return detector, extractor


def run_training_pipeline(args):
    """Main training pipeline."""
    # Extract common parameters
    stratified = getattr(args, 'stratified_sample', False)
    random_state = getattr(args, 'random_state', 42)
    
    # Load training data
    if args.dataset_name == "human_ai":
        train_texts, train_labels = load_human_ai_dataset(
            args.train_data_path, 
            n_rows=getattr(args, 'n_rows', None),
            stratified=stratified,
            random_state=random_state
        )
    elif args.dataset_name in {"daigtv2", "daigt_v2", "daigt"}:
        # Dedicated loader with fixed columns
        train_texts, train_labels = load_daigtv2_dataset(
            args.train_data_path, 
            n_rows=getattr(args, 'n_rows', None),
            stratified=stratified,
            random_state=random_state
        )
    else:
        # For other datasets, load from CSV with specified columns
        read_kwargs = {}
        if getattr(args, 'n_rows', None):
            read_kwargs['nrows'] = args.n_rows
        train_df = pd.read_csv(args.train_data_path, **read_kwargs)
        # Allow dataset-specific smart defaults if user didn't override columns
        text_col = getattr(args, 'text_column', None)
        label_col = getattr(args, 'label_column', None)
        if text_col is None:
            # Try common names
            for cand in ("text", "content", "answer"):
                if cand in train_df.columns:
                    text_col = cand
                    break
        if label_col is None:
            for cand in ("label", "generated", "is_cheating", "target"):
                if cand in train_df.columns:
                    label_col = cand
                    break
        if text_col is None or label_col is None:
            raise ValueError(
                f"Could not infer text/label columns. Available columns: {list(train_df.columns)}.\n"
                f"Pass --text_column and --label_column explicitly."
            )
        
        # Apply stratified sampling if requested (balanced 50/50 per class)
        if stratified and getattr(args, 'n_rows', None):
            n_rows = getattr(args, 'n_rows', None)
            n_per_class = n_rows // 2
            label_col_int = pd.to_numeric(train_df[label_col], errors='coerce').fillna(0).astype(int)
            df_0 = train_df[label_col_int == 0].sample(n=min(n_per_class, len(train_df[label_col_int == 0])), random_state=random_state)
            df_1 = train_df[label_col_int == 1].sample(n=min(n_per_class, len(train_df[label_col_int == 1])), random_state=random_state)
            train_df = pd.concat([df_0, df_1], ignore_index=True)
            print(f"Loaded {len(train_df)} samples (stratified: {len(df_0)} class-0 + {len(df_1)} class-1)")
        
        train_texts = sanitize_texts(train_df[text_col].astype(str).tolist())
        train_labels = pd.to_numeric(train_df[label_col], errors='coerce').fillna(0).astype(int).to_numpy()

    # Optional sampling for faster experiments
    if getattr(args, 'sample_frac', None):
        frac = float(args.sample_frac)
        if not (0 < frac <= 1):
            raise ValueError("--sample_frac must be in (0,1]")
        print(f"Sampling {frac:.2%} of the training data with random_state={getattr(args, 'random_state', 42)}")
        # Rebuild a DataFrame to sample in a label-aware manner
        df_tmp = pd.DataFrame({
            'text': train_texts,
            'label': train_labels
        })
        # Stratified sample per label when possible
        sampled = df_tmp.groupby('label', group_keys=False).apply(
            lambda g: g.sample(frac=frac, random_state=getattr(args, 'random_state', 42))
        ) if len(np.unique(train_labels)) > 1 else df_tmp.sample(frac=frac, random_state=getattr(args, 'random_state', 42))

        train_texts = sampled['text'].tolist()
        train_labels = sampled['label'].astype(int).to_numpy()

    # Ensure labels are int64 for np.bincount
    train_labels = train_labels.astype(int) if train_labels.dtype != np.int64 else train_labels
    
    print(f"Training on {len(train_texts)} samples from {args.dataset_name} dataset")
    print(f"Label distribution: {np.bincount(train_labels)} (0=real, 1=fake)")

    # Train detector
    detector, extractor = train_detector(train_texts, train_labels, args)

    # Generate model name and save
    model_name = generate_model_name(args, args.dataset_name)
    save_dir = Path("saved_models")
    save_detector_and_metadata(detector, args, save_dir, model_name, extractor=extractor)

    # Clean up extractor to free memory (only after saving TF-IDF vectorizer if applicable)
    if extractor:
        del extractor
    if args.analysis_type == "embedding":
        clear_gpu_memory()

    print(f"✅ Training pipeline completed. Model saved as: {model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train detector on dataset and save for cross-dataset evaluation")

    # Model and feature extraction
    parser.add_argument("--model_name", type=str, required=False, default="sentence-transformers/all-distilroberta-v1",
                        help="HF model to use for feature extraction (ignored when --analysis_type=tfidf)")
    parser.add_argument("--analysis_type", type=str, default="embedding", choices=["embedding", "perplexity", "phd", "tfidf"],
                        help="Feature family to compute; use 'tfidf' for classical TF-IDF vectors")
    parser.add_argument(
        "--classifier_type",
        type=str,
        default="svm",
        choices=["neural", "svm", "lr", "xgb", "elliptic", "ocsvm", "iforest"],
        help="Classifier head: binary (svm/lr/xgb/neural) or outlier (elliptic/ocsvm/iforest)"
    )

    # Dataset configuration
    parser.add_argument("--dataset_name", type=str, default="human_ai", help="Dataset identifier for naming")
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to training data CSV")
    parser.add_argument("--text_column", type=str, default="text", help="Name of the text column")
    parser.add_argument("--label_column", type=str, default="generated", help="Name of the label column")

    # Feature extraction parameters
    parser.add_argument("--layer", type=int, default=22, help="Layer index for embedding/phd analysis")
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean",
        choices=["mean", "max", "last", "attn_mean", "mean_std", "statistical", "covariance"],
        help="Pooling strategy when analysis_type=embedding (supports: mean, max, last, attn_mean, mean_std, statistical/covariance)"
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Forward batch size for HF extractor")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum token length for feature extractors")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device to run on")
    parser.add_argument("--validation_split", type=float, default=0.2, help="Validation split ratio for BinaryDetector")
    parser.add_argument("--memory_efficient", action="store_true", help="Use memory-efficient single-layer pooled extraction for embeddings")
    parser.add_argument("--normalize", action="store_true", help="L2-normalize token embeddings before pooling (embedding analysis only)")
    parser.add_argument("--log_memory", action="store_true", help="Print memory usage during embedding extraction")
    parser.add_argument("--memory_log_interval", type=int, default=1, help="Batches between memory log prints")

    # Data subsampling options for large CSVs
    parser.add_argument("--n_rows", type=int, default=None, help="Read only the first N rows from CSV (for quick tests)")
    parser.add_argument("--sample_frac", type=float, default=None, help="Optionally sample a fraction of rows after loading (0<frac<=1)")
    parser.add_argument("--stratified_sample", action="store_true", help="When loading training data with --n_rows, sample balanced classes (50/50 real/fake)")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for sampling")

    # TF-IDF specific parameters
    parser.add_argument("--tfidf_max_features", type=int, default=20000, help="Max vocabulary size for TF-IDF")
    parser.add_argument("--tfidf_ngram_min", type=int, default=1, help="Minimum n-gram for TF-IDF")
    parser.add_argument("--tfidf_ngram_max", type=int, default=2, help="Maximum n-gram for TF-IDF")
    parser.add_argument("--tfidf_min_df", type=float, default=1, help="Min document frequency for TF-IDF")
    parser.add_argument("--tfidf_max_df", type=float, default=1.0, help="Max document frequency for TF-IDF")
    parser.add_argument("--tfidf_stop_words", type=str, default=None, help="Stop words for TF-IDF (e.g., 'english')")
    parser.add_argument("--tfidf_dense", action="store_true", help="Return dense arrays instead of sparse CSR for TF-IDF")
    parser.add_argument("--svd_components", type=int, default=None, help="Use TruncatedSVD with given components for TF-IDF")

    args = parser.parse_args()
    run_training_pipeline(args)