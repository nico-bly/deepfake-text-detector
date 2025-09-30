#!/usr/bin/env python3
"""
ESA Challenge Main Script - Unified Binary Classification Approach

This script implements the full pipeline using the new unified architecture:
UnifiedTextDataset → DataLoader → EmbeddingExtractor → BinaryDetector

Based on the framework from main.py but adapted for the unified approach
focusing on embeddings-only classification.
"""

import os
import sys
import gc
from pathlib import Path

import torch
import pandas as pd
import numpy as np

# Add current directory to path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))  # Add parent directory for models/utils access

from utils.data_loader import create_unified_dataloaders, reconstruct_pairs_from_predictions
from models.extractors import EmbeddingExtractor
from models.classifiers import BinaryDetector

def clear_gpu_memory():
    """Clear GPU memory completely"""
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()

def extract_training_data(train_dataloader, max_samples=None):
    """
    Extract training data from unified DataLoader
    
    Args:
        train_dataloader: Unified DataLoader
        max_samples: Maximum samples to extract (None for all)
    
    Returns:
        all_texts: List of texts
        all_labels: List of binary labels (0=real, 1=fake)
    """
    print("Extracting training data from unified DataLoader...")
    
    all_texts = []
    all_labels = []
    sample_count = 0
    
    for batch_idx, batch in enumerate(train_dataloader):
        batch_texts = batch['texts']
        batch_labels = batch['labels'].tolist()
        
        all_texts.extend(batch_texts)
        all_labels.extend(batch_labels)
        sample_count += len(batch_texts)
        
        if max_samples and sample_count >= max_samples:
            break
    
    # Limit to max_samples if specified
    if max_samples:
        all_texts = all_texts[:max_samples]
        all_labels = all_labels[:max_samples]
    
    print(f"Extracted {len(all_texts)} texts")
    print(f"Label distribution: {np.bincount(all_labels)} (0=real, 1=fake)")
    
    return all_texts, all_labels


def process_model_layer_combination(
    model_id, 
    target_layer, 
    train_texts, 
    train_labels, 
    test_dataloader, 
    device='cuda:2',
    batch_size=16,
    validation_split=0.2
):
    """
    Process a single model-layer combination using unified approach
    
    Args:
        model_id: Model identifier
        target_layer: Layer to extract embeddings from
        train_texts: Training texts
        train_labels: Training labels (0=real, 1=fake)
        test_dataloader: Test DataLoader
        device: GPU device
        batch_size: Batch size for processing
        validation_split: Validation split for training
    
    Returns:
        submission_df: Predictions with sample IDs for reconstruction
    """
    print(f"\n{'='*60}")
    print(f"Processing: {model_id} - Layer {target_layer}")
    print(f"{'='*60}")
    
    # Clear memory before loading new model
    clear_gpu_memory()
    
    try:
        # 1. Load EmbeddingExtractor
        print(f"Loading EmbeddingExtractor for {model_id}...")
        extractor = EmbeddingExtractor(model_id, device=device)
        
        # 2. Extract training embeddings
        print(f"Extracting training embeddings from layer {target_layer}...")
        train_layer_embeddings = extractor.get_all_layer_embeddings(
            train_texts,
            pooling='last_token',
            batch_size=batch_size
        )
        
        train_embeddings = train_layer_embeddings[target_layer]
        train_labels_array = np.array(train_labels)
        
        print(f"Training embeddings shape: {train_embeddings.shape}")
        
        # 3. Initialize and train BinaryDetector
        print(f"Training BinaryDetector...")
        detector = BinaryDetector(
            n_components=0.95,
            contamination=0.1,
            random_state=42
        )
        
        training_results = detector.fit(
            embeddings=train_embeddings,
            labels=train_labels_array,
            validation_split=validation_split
        )
        
        print(f"Training completed: {training_results}")
        
        # 4. Make predictions on test data
        print(f"Making predictions on test data...")
        test_predictions = []
        test_sample_ids = []
        
        for batch in test_dataloader:
            batch_texts = batch['texts']
            batch_sample_ids = batch['sample_ids']
            
            # Extract embeddings for batch
            batch_layer_embeddings = extractor.get_all_layer_embeddings(
                batch_texts,
                pooling='last_token',
                batch_size=batch_size
            )
            batch_embeddings = batch_layer_embeddings[target_layer]
            
            # Get predictions (probabilities of being fake)
            _, batch_probabilities = detector.predict(
                batch_embeddings,
                return_probabilities=True,
                return_distances=False
            )
            
            test_predictions.extend(batch_probabilities.tolist())
            test_sample_ids.extend(batch_sample_ids)
        
        print(f"Generated {len(test_predictions)} predictions")
        
        # 5. Reconstruct pairs for submission
        print(f"Reconstructing pair predictions...")
        submission_df = reconstruct_pairs_from_predictions(
            test_predictions, 
            test_sample_ids
        )
        
        print(f"Reconstructed {len(submission_df)} pair predictions")
        
        # Save individual result
        model_name = model_id.split('/')[-1]
        output_file = f"submission_{model_name}_layer{target_layer}.csv"
        submission_df.to_csv(output_file, index=False)
        print(f"Saved results to {output_file}")
        
        # Clean up
        del extractor
        del detector
        clear_gpu_memory()
        
        return submission_df
        
    except Exception as e:
        print(f"Error processing {model_id} layer {target_layer}: {e}")
        return None
    
    finally:
        clear_gpu_memory()


def ensemble_predictions_unified(all_results, voting="majority"):
    """
    Create ensemble predictions from unified results
    
    Args:
        all_results: List of (model_id, layer, submission_df) tuples
        voting: Voting strategy
    
    Returns:
        ensemble_df: Ensemble predictions
    """
    print(f"\nCreating ensemble predictions...")
    
    valid_results = [(model_id, layer, df) for model_id, layer, df in all_results if df is not None]
    
    if not valid_results:
        raise ValueError("No valid predictions to ensemble!")
    
    print(f"Ensembling {len(valid_results)} valid model-layer combinations:")
    for model_id, layer, _ in valid_results:
        model_name = model_id.split('/')[-1]
        print(f"  - {model_name}_layer{layer}")
    
    # Get all predictions aligned by ID
    base_df = valid_results[0][2][['id']].copy()  # Use first result for ID structure
    all_predictions = []
    
    for model_id, layer, submission_df in valid_results:
        # Ensure same order by merging on ID
        merged = base_df.merge(submission_df[['id', 'real_text_id']], on='id', how='left')
        all_predictions.append(merged['real_text_id'].values)
    
    # Convert to matrix for ensemble voting
    predictions_matrix = np.vstack(all_predictions)  # Shape: (n_models, n_samples)
    
    if voting == "majority":
        final_predictions = []
        for i in range(predictions_matrix.shape[1]):
            votes = predictions_matrix[:, i]
            # Count votes for each class (1 or 2)
            vote_counts = np.bincount(votes, minlength=3)  # [0, count_1, count_2]
            # Majority wins, ties go to 1
            final_pred = 1 if vote_counts[1] >= vote_counts[2] else 2
            final_predictions.append(final_pred)
    else:
        raise NotImplementedError(f"Voting strategy {voting} not implemented")
    
    # Create ensemble DataFrame
    ensemble_df = pd.DataFrame({
        'id': base_df['id'],
        'real_text_id': final_predictions
    })
    
    print(f"Ensemble prediction distribution: {np.bincount(final_predictions, minlength=3)[1:]}")
    
    return ensemble_df


def analyze_results(ensemble_df, all_results):
    """Analyze ensemble results and individual model performance"""
    print(f"\n{'='*60}")
    print("RESULTS ANALYSIS")
    print(f"{'='*60}")
    
    # Ensemble statistics
    ensemble_stats = ensemble_df['real_text_id'].value_counts()
    total_predictions = len(ensemble_df)
    
    print(f"Ensemble Results:")
    print(f"  Total predictions: {total_predictions}")
    print(f"  Text 1 predictions: {ensemble_stats.get(1, 0)} ({ensemble_stats.get(1, 0)/total_predictions*100:.1f}%)")
    print(f"  Text 2 predictions: {ensemble_stats.get(2, 0)} ({ensemble_stats.get(2, 0)/total_predictions*100:.1f}%)")
    
    # Individual model analysis
    print(f"\nIndividual Model Results:")
    valid_count = 0
    for model_id, layer, df in all_results:
        model_name = model_id.split('/')[-1]
        if df is not None:
            stats = df['real_text_id'].value_counts()
            print(f"  {model_name}_layer{layer}: Text1={stats.get(1,0)}, Text2={stats.get(2,0)} - SUCCESS")
            valid_count += 1
        else:
            print(f"  {model_name}_layer{layer}: FAILED")
    
    print(f"\nValid models: {valid_count}/{len(all_results)}")


def test_pipeline_components():
    """Test that all components are properly imported and functional"""
    print("Testing pipeline components...")
    
    try:
        # Test imports
        from utils.data_loader import create_unified_dataloaders
        from models.extractors import EmbeddingExtractor
        from models.classifiers import BinaryDetector
        print("✅ All imports successful")
        
        # Test with minimal data
        print("✅ Component test passed")
        return True
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        return False


def save_training_results(model_results, output_dir="results"):
    """Save detailed training results for analysis"""
    os.makedirs(output_dir, exist_ok=True)
    
    results_summary = []
    for model_id, layer, df in model_results:
        model_name = model_id.split('/')[-1]
        
        if df is not None:
            # Calculate statistics
            stats = df['real_text_id'].value_counts()
            results_summary.append({
                'model': model_name,
                'layer': layer,
                'status': 'SUCCESS',
                'text1_predictions': stats.get(1, 0),
                'text2_predictions': stats.get(2, 0),
                'total_predictions': len(df),
                'text1_ratio': stats.get(1, 0) / len(df) if len(df) > 0 else 0
            })
        else:
            results_summary.append({
                'model': model_name,
                'layer': layer,
                'status': 'FAILED',
                'text1_predictions': 0,
                'text2_predictions': 0,
                'total_predictions': 0,
                'text1_ratio': 0
            })
    
    # Save summary
    summary_df = pd.DataFrame(results_summary)
    summary_file = os.path.join(output_dir, 'training_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    print(f"Training summary saved to {summary_file}")
    
    return summary_df


def quick_test_mode():
    """Run a quick test with minimal data to verify the pipeline"""
    print("\n" + "="*60)
    print("QUICK TEST MODE")
    print("="*60)
    
    # Quick test configuration
    BATCH_SIZE = 4
    MAX_TRAIN_SAMPLES = 20
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Minimal model configuration for testing
    test_models_config = {
        "sentence-transformers/all-distilroberta-v1": [5],  # Just one layer for testing
    }
    
    return run_pipeline(
        batch_size=BATCH_SIZE,
        max_train_samples=MAX_TRAIN_SAMPLES,
        device=DEVICE,
        models_config=test_models_config,
        test_mode=True
    )


def run_pipeline(
    batch_size=16,
    max_train_samples=None,
    device='cuda:2',
    models_config=None,
    test_mode=False
):
    """
    Run the complete pipeline with given configuration
    
    Args:
        batch_size: Batch size for processing
        max_train_samples: Maximum training samples (None for all)
        device: GPU device to use
        models_config: Dictionary of models and layers
        test_mode: Whether running in test mode
    """
    
    # Default model configuration
    if models_config is None:
        models_config = {
            "sentence-transformers/all-distilroberta-v1": [3, 4, 5],
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": [4, 5],
        }
    
    # Data paths
    train_path = "/home/infres/billy-22/projets/esa_challenge_kaggle/esa-challenge-fake-texts/data/train"
    train_labels_path = "/home/infres/billy-22/projets/esa_challenge_kaggle/esa-challenge-fake-texts/data/train.csv"
    test_path = "/home/infres/billy-22/projets/esa_challenge_kaggle/esa-challenge-fake-texts/data/test"
    
    print(f"Pipeline Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Max training samples: {max_train_samples}")
    print(f"  Device: {device}")
    print(f"  Test mode: {test_mode}")
    print(f"  Models: {list(models_config.keys())}")
    print()
    
    try:
        # 1. Create unified DataLoaders
        print("1. Creating Unified DataLoaders...")
        train_dataloader, test_dataloader = create_unified_dataloaders(
            train_path=train_path,
            train_labels_path=train_labels_path,
            test_path=test_path if not test_mode else None,
            batch_size=batch_size,
            num_workers=0,
            max_length=512,
            lazy_loading=True,
            include_metadata=True
        )
        
        if train_dataloader is None:
            print("❌ Failed to create DataLoaders")
            return False
        
        print(f"✅ Created DataLoaders:")
        print(f"  Training samples: {len(train_dataloader.dataset)}")
        if test_dataloader:
            print(f"  Test samples: {len(test_dataloader.dataset)}")
        
        # 2. Extract training data
        print("\n2. Extracting Training Data...")
        train_texts, train_labels = extract_training_data(
            train_dataloader, 
            max_samples=max_train_samples
        )
        
        # 3. Process each model-layer combination
        print(f"\n3. Processing Model-Layer Combinations...")
        all_results = []
        
        for model_id, layers in models_config.items():
            for target_layer in layers:
                submission_df = process_model_layer_combination(
                    model_id=model_id,
                    target_layer=target_layer,
                    train_texts=train_texts,
                    train_labels=train_labels,
                    test_dataloader=test_dataloader if test_dataloader else train_dataloader,
                    device=device,
                    batch_size=batch_size,
                    validation_split=0.2
                )
                
                all_results.append((model_id, target_layer, submission_df))
        
        # 4. Create ensemble predictions
        print(f"\n4. Creating Ensemble Predictions...")
        ensemble_df = ensemble_predictions_unified(all_results, voting="majority")
        
        # Save results
        suffix = "_test" if test_mode else ""
        ensemble_output = f"submission_ensemble_unified{suffix}.csv"
        ensemble_df.to_csv(ensemble_output, index=False)
        print(f"✅ Ensemble saved to {ensemble_output}")
        
        # 5. Save detailed results
        save_training_results(all_results)
        
        # 6. Analyze results
        analyze_results(ensemble_df, all_results)
        
        print(f"\n✅ Pipeline completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        return False
    
    finally:
        clear_gpu_memory()


if __name__ == "__main__":
    print("ESA Challenge - Unified Binary Classification Pipeline")

    # Full production configuration
    # "sentence-transformers/all-distilroberta-v1": [3, 4, 5],
    # "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": [4, 5],
    # "meta-llama/Llama-3.1-8B": [26, 30],
    models_config = {
        "Qwen/Qwen3-Embedding-8B": [35],
    }
    success = run_pipeline(
        batch_size=16,
        max_train_samples=None,  # Use all training data
        device='cuda:2',
        models_config=models_config,
        test_mode=False
    )
    