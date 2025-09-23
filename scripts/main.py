import os
import torch
import gc

import pandas as pd
import numpy as np


from utils import extract_real_texts, read_texts_from_dir
from models import EmbeddingExtractor
from classifiers import OutlierDetections, TrajectoryClassifier

def clear_gpu_memory():
    """Clear GPU memory completely"""
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()

def process_trajectory_model(real_texts_train, fake_texts_train, df_test, extractor_model, layer=-2, classifier_type="svm"):
    """
    Process trajectory classifier for a specific layer
    
    Args:
        real_texts_train: Training real texts
        fake_texts_train: Training fake texts
        df_test: Test dataframe
        extractor_model: EmbeddingExtractor instance
        layer: layer to extract embeddings from
        classifier_type: type of classifier ("svm", "rf", "lr")
    
    Returns:
        submission_df or None if failed
    """
    print(f"\nProcessing trajectory classifier (layer {layer}, classifier: {classifier_type})...")
    
    try:
        # Create trajectory classifier
        traj_classifier = TrajectoryClassifier(
            classifier_type=classifier_type,
            random_state=42
        )
        
        # Train the classifier
        print(f"Training trajectory classifier...")
        cv_score = traj_classifier.train(
            real_texts_train,
            fake_texts_train,
            extractor_model,
            target_layer=layer,
            batch_size=8
        )
        
        print(f"Making trajectory predictions...")
        submission_df = predict_test_batch_trajectory(
            traj_classifier=traj_classifier,
            df_test=df_test,
            extractor_model=extractor_model,
            target_layer=layer,
            batch_size=8
        )
        
        return submission_df
        
    except Exception as e:
        print(f"Error processing trajectory classifier: {e}")
        return None

def predict_test_batch_trajectory(traj_classifier, df_test, extractor_model, target_layer=-2, batch_size=32):
    """
    Batch prediction for test set using trajectory classifier.
    Returns predictions with confidence scores.
    """
    # Collect all file_1 and file_2 texts
    texts1 = df_test['file_1'].tolist()
    texts2 = df_test['file_2'].tolist()
    
    # Get predictions and probabilities
    pred1, prob1 = traj_classifier.predict_batch(
        texts1, extractor_model, target_layer=target_layer, batch_size=batch_size
    )
    pred2, prob2 = traj_classifier.predict_batch(
        texts2, extractor_model, target_layer=target_layer, batch_size=batch_size
    )
    
    # Decide which file is real based on probabilities (higher = more likely real)
    real_text_ids = [1 if p1 > p2 else 2 for p1, p2 in zip(prob1, prob2)]
    
    # Calculate confidence as the difference between probabilities
    confidences = [abs(p1 - p2) for p1, p2 in zip(prob1, prob2)]
    
    # Print some stats
    print(f"Text1 average probability: {np.mean(prob1):.3f}")
    print(f"Text2 average probability: {np.mean(prob2):.3f}")
    print(f"Average confidence: {np.mean(confidences):.3f}")
    print(f"Text1 predicted as real: {sum(1 for id in real_text_ids if id == 1)} / {len(real_text_ids)}")
    
    # Build submission DataFrame with confidence
    submission_df = pd.DataFrame({
        'id': range(len(df_test)),
        'real_text_id': real_text_ids,
        'confidence': confidences,
        'text1_prob': prob1,
        'text2_prob': prob2
    })
    
    return submission_df

def process_model_sequentially(model_id, real_texts_train, fake_texts_train, df_test, layers_for_model, device='cuda:2', include_trajectory=False):
    """
    Process a single model with specified layers sequentially
    
    Args:
        model_id: Model identifier
        real_texts_train: Training real texts
        fake_texts_train: Training fake texts
        df_test: Test dataframe
        layers_for_model: List of layers to process for this specific model
        device: GPU device
        include_trajectory: Whether to include trajectory classifier
    
    Returns:
        dict: {layer: submission_df} for this model
    """
    print(f"\n{'='*60}")
    print(f"Processing model: {model_id}")
    print(f"Layers: {layers_for_model}")
    print(f"Include trajectory: {include_trajectory}")
    print(f"{'='*60}")
    
    # Clear memory before loading new model
    clear_gpu_memory()
    
    try:
        # Load the model
        print(f"Loading extractor for {model_id}...")
        extractor = EmbeddingExtractor(model_id, device=device)
        
        model_results = {}
        
        # Process each layer for this model
        for layer in layers_for_model:
            print(f"\nProcessing layer {layer} for {model_id}...")
            
            try:
                # Create and train detector for this layer
                detector = OutlierDetections(
                    detector_type="svm_binary",
                    random_state=42,
                    n_components=0.95
                )
                
                print(f"Training binary classifier for layer {layer}...")
                detector.learn_binary_classification(
                    real_texts_train,
                    fake_texts_train,
                    extractor_model=extractor,
                    target_layer=layer,
                    pooling='mean',
                    batch_size=8
                )
                
                print(f"Making predictions for layer {layer}...")
                submission_df = predict_test_batch_svm(
                    detector=detector,
                    df_test=df_test,
                    extractor_model=extractor,
                    target_layer=layer,
                    batch_size=8,
                    pooling='mean'
                )
                
                model_results[layer] = submission_df
                
                # Save individual result
                model_name = model_id.split('/')[-1]
                output_file = f"submission_{model_name}_layer{layer}.csv"
                submission_df.to_csv(output_file, index=False)
                print(f"Saved layer {layer} results to {output_file}")
                
                # Clear detector to free memory
                del detector
                clear_gpu_memory()
                
            except Exception as e:
                print(f"Error processing layer {layer} for {model_id}: {e}")
                model_results[layer] = None
        
        # Process trajectory classifier if requested
        if include_trajectory:
            print(f"\nProcessing trajectory classifier for {model_id}...")
            try:
                # Use the last layer for trajectory analysis
                trajectory_layer = layers_for_model[-1] if layers_for_model else -2
                
                trajectory_df = process_trajectory_model(
                    real_texts_train=real_texts_train,
                    fake_texts_train=fake_texts_train,
                    df_test=df_test,
                    extractor_model=extractor,
                    layer=trajectory_layer,
                    classifier_type="svm"
                )
                
                if trajectory_df is not None:
                    model_results[f"trajectory_{trajectory_layer}"] = trajectory_df
                    
                    # Save trajectory result
                    model_name = model_id.split('/')[-1]
                    output_file = f"submission_{model_name}_trajectory_layer{trajectory_layer}.csv"
                    trajectory_df.to_csv(output_file, index=False)
                    print(f"Saved trajectory results to {output_file}")
                else:
                    model_results[f"trajectory_{trajectory_layer}"] = None
                    
                clear_gpu_memory()
                
            except Exception as e:
                print(f"Error processing trajectory classifier for {model_id}: {e}")
                model_results[f"trajectory_{layers_for_model[-1] if layers_for_model else -2}"] = None
        
        print(f"Completed processing {model_id}")
        return model_results
        
    except Exception as e:
        print(f"Error loading model {model_id}: {e}")
        return {}
    
    finally:
        # Clean up the extractor
        if 'extractor' in locals():
            del extractor
        clear_gpu_memory()
        print(f"Model {model_id} unloaded and memory cleared")

def ensemble_predictions_sequential(all_model_results, df_test, voting="majority"):
    """
    Create ensemble predictions from sequential model results with confidence scores
    
    Args:
        all_model_results: dict {model_id: {layer: submission_df}}
        df_test: test DataFrame
        voting: voting strategy
    
    Returns:
        submission_df: ensemble predictions with confidence percentages
    """
    print(f"\nCreating ensemble predictions...")
    
    # Collect all valid predictions with their probabilities
    all_votes = []
    all_confidences = []
    all_text1_probs = []
    all_text2_probs = []
    valid_combinations = []
    
    for model_id, model_results in all_model_results.items():
        for layer, submission_df in model_results.items():
            if submission_df is not None:
                all_votes.append(submission_df['real_text_id'].values)
                all_confidences.append(submission_df['confidence'].values if 'confidence' in submission_df.columns else np.zeros(len(submission_df)))
                all_text1_probs.append(submission_df['text1_prob'].values if 'text1_prob' in submission_df.columns else np.zeros(len(submission_df)))
                all_text2_probs.append(submission_df['text2_prob'].values if 'text2_prob' in submission_df.columns else np.zeros(len(submission_df)))
                valid_combinations.append(f"{model_id}_layer{layer}")
    
    if not all_votes:
        raise ValueError("No valid predictions to ensemble!")
    
    print(f"Ensembling {len(all_votes)} valid model-layer combinations:")
    for combo in valid_combinations:
        print(f"  - {combo}")
    
    # Convert to matrices
    votes_matrix = np.vstack(all_votes)
    confidences_matrix = np.vstack(all_confidences)
    text1_probs_matrix = np.vstack(all_text1_probs)
    text2_probs_matrix = np.vstack(all_text2_probs)
    
    if voting == "majority":
        final_preds = []
        ensemble_confidences = []
        ensemble_text1_probs = []
        ensemble_text2_probs = []
        
        for i in range(votes_matrix.shape[1]):
            votes = votes_matrix[:, i]
            confidences = confidences_matrix[:, i]
            text1_probs = text1_probs_matrix[:, i]
            text2_probs = text2_probs_matrix[:, i]
            
            # Majority voting with ties going to 1
            vote_counts = np.bincount(votes, minlength=3)  # [0, count_1, count_2]
            final_pred = 1 if vote_counts[1] >= vote_counts[2] else 2
            final_preds.append(final_pred)
            
            # Average confidence scores
            avg_confidence = np.mean(confidences)
            ensemble_confidences.append(avg_confidence)
            
            # Average probabilities
            avg_text1_prob = np.mean(text1_probs)
            avg_text2_prob = np.mean(text2_probs)
            ensemble_text1_probs.append(avg_text1_prob)
            ensemble_text2_probs.append(avg_text2_prob)
    else:
        raise NotImplementedError(f"Voting strategy {voting} not implemented")
    
    # Convert confidence to percentage
    confidence_percentages = [conf * 100 for conf in ensemble_confidences]
    
    submission_df = pd.DataFrame({
        'id': range(len(df_test)),
        'real_text_id': final_preds,
        'confidence_percentage': confidence_percentages,
        'avg_text1_prob': ensemble_text1_probs,
        'avg_text2_prob': ensemble_text2_probs
    })
    
    print(f"Ensemble confidence statistics:")
    print(f"  Mean confidence: {np.mean(confidence_percentages):.1f}%")
    print(f"  Median confidence: {np.median(confidence_percentages):.1f}%")
    print(f"  Min confidence: {np.min(confidence_percentages):.1f}%")
    print(f"  Max confidence: {np.max(confidence_percentages):.1f}%")
    
    return submission_df


def analyze_prediction_ties(ensemble_df):
    """
    Analyze cases where models disagree or have low confidence
    """
    print(f"\n{'='*50}")
    print("PREDICTION ANALYSIS")
    print(f"{'='*50}")
    
    # Analyze confidence distribution
    confidence_stats = ensemble_df['confidence_percentage'].describe()
    print(f"\nConfidence Statistics:")
    for stat, value in confidence_stats.items():
        print(f"  {stat}: {value:.1f}%")
    
    # Find low confidence predictions
    low_confidence_threshold = 20.0  # 20%
    low_confidence_mask = ensemble_df['confidence_percentage'] < low_confidence_threshold
    low_confidence_count = low_confidence_mask.sum()
    
    print(f"\nLow Confidence Predictions (< {low_confidence_threshold}%):")
    print(f"  Count: {low_confidence_count} / {len(ensemble_df)} ({100*low_confidence_count/len(ensemble_df):.1f}%)")
    
    if low_confidence_count > 0:
        print(f"  Sample low confidence cases:")
        low_conf_samples = ensemble_df[low_confidence_mask].head(5)
        for idx, row in low_conf_samples.iterrows():
            print(f"    ID {row['id']}: prediction={row['real_text_id']}, confidence={row['confidence_percentage']:.1f}%")
    
    # Analyze cases where probabilities are very close (potential ties)
    prob_diff = abs(ensemble_df['avg_text1_prob'] - ensemble_df['avg_text2_prob'])
    close_probs_mask = prob_diff < 0.1  # Difference less than 0.1
    close_probs_count = close_probs_mask.sum()
    
    print(f"\nClose Probability Cases (prob difference < 0.1):")
    print(f"  Count: {close_probs_count} / {len(ensemble_df)} ({100*close_probs_count/len(ensemble_df):.1f}%)")
    
    if close_probs_count > 0:
        print(f"  Sample close probability cases:")
        close_samples = ensemble_df[close_probs_mask].head(5)
        for idx, row in close_samples.iterrows():
            print(f"    ID {row['id']}: text1_prob={row['avg_text1_prob']:.3f}, text2_prob={row['avg_text2_prob']:.3f}, diff={prob_diff.iloc[idx]:.3f}")
    
    print(f"\n{'='*50}")


def predict_test_batch_svm(detector, df_test, extractor_model, target_layer=5, batch_size=32, pooling="mean"):
    """
    Batch prediction for test set using SVM binary classifier.
    Returns predictions with confidence scores.
    """
    # Collect all file_1 and file_2 texts
    texts1 = df_test['file_1'].tolist()
    texts2 = df_test['file_2'].tolist()
    
    # For SVM, we get predictions, probabilities, and distances
    pred1, prob1, dist1 = detector.predict_texts_batch_svm(
        texts1, extractor_model, target_layer=target_layer, pooling=pooling, batch_size=batch_size
    )
    pred2, prob2, dist2 = detector.predict_texts_batch_svm(
        texts2, extractor_model, target_layer=target_layer, pooling=pooling, batch_size=batch_size
    )
    
    # Decide which file is real based on probabilities (higher = more likely real)
    real_text_ids = [1 if p1 > p2 else 2 for p1, p2 in zip(prob1, prob2)]
    
    # Calculate confidence as the difference between probabilities
    # Higher difference = more confident decision
    confidences = [abs(p1 - p2) for p1, p2 in zip(prob1, prob2)]
    
    # Print some stats
    print(f"Text1 average probability: {np.mean(prob1):.3f}")
    print(f"Text2 average probability: {np.mean(prob2):.3f}")
    print(f"Average confidence: {np.mean(confidences):.3f}")
    print(f"Text1 predicted as real: {sum(1 for id in real_text_ids if id == 1)} / {len(real_text_ids)}")
    
    # Build submission DataFrame with confidence
    submission_df = pd.DataFrame({
        'id': range(len(df_test)),
        'real_text_id': real_text_ids,
        'confidence': confidences,
        'text1_prob': prob1,
        'text2_prob': prob2
    })
    
    return submission_df


if __name__ == "__main__":
    # --------------- load data -----------------
    train_path="/home/infres/billy-22/projets/esa_challenge_kaggle/esa-challenge-fake-texts/data/train"
    df_train=read_texts_from_dir(train_path)
    test_path="/home/infres/billy-22/projets/esa_challenge_kaggle/esa-challenge-fake-texts/data/test"
    df_test=read_texts_from_dir(test_path)

    df_train_gt=pd.read_csv("/home/infres/billy-22/projets/esa_challenge_kaggle/esa-challenge-fake-texts/data/train.csv")


    # Extract real texts
    real_texts = extract_real_texts(df_train, df_train_gt)

    model_id = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
    model_id = "mistralai/Mistral-7B-v0.3"
    model_id = "meta-llama/Llama-3.1-8B"
    model_id = "sentence-transformers/all-distilroberta-v1"
    model_id = "Qwen/Qwen2-1.5B"
    model_id = "sentence-transformers/all-distilroberta-v1"
    # --------------- load extractor embedding -----------------


    extractor = EmbeddingExtractor(model_id, device='cuda:2')

    df_merged = df_train.merge(df_train_gt, on='id')

    real_texts_train = [
        row['file_1'] if row['real_text_id'] == 1 else row['file_2']
        for idx, row in df_merged.iterrows()
    ]

    fake_texts_train = [
        row['file_2'] if row['real_text_id'] == 1 else row['file_1']
        for idx, row in df_merged.iterrows()
    ]

    print(f"Training with {len(real_texts_train)} real and {len(fake_texts_train)} fake texts")

    # Define models and their specific layers to process
    # "mistralai/Mistral-7B-v0.3": [16, 24, 26, 28,30],
    # "meta-llama/Llama-3.1-8B": [30],
    models_config = {
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": [4,5],
        "sentence-transformers/all-distilroberta-v1": [3,4,5],
        "meta-llama/Llama-3.1-8B": [26,30],
        "Qwen/Qwen3-8B": [33],
    }

    # Enable trajectory analysis for specific models
    trajectory_models = {
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": False,
        "sentence-transformers/all-distilroberta-v1": False,
        "meta-llama/Llama-3.1-8B": False,
        "Qwen/Qwen3-8B": False,
    }

    device = 'cuda:2'

    # Process each model sequentially with its specific layers
    all_model_results = {}
    
    for model_id, layers_for_model in models_config.items():
        include_trajectory = trajectory_models.get(model_id, False)
        
        model_results = process_model_sequentially(
            model_id=model_id,
            real_texts_train=real_texts_train,
            fake_texts_train=fake_texts_train,
            df_test=df_test,
            layers_for_model=layers_for_model,
            device=device,
            include_trajectory=include_trajectory
        )
        all_model_results[model_id] = model_results
    
    # Create ensemble prediction
    try:
        ensemble_df = ensemble_predictions_sequential(
            all_model_results=all_model_results,
            df_test=df_test,
            voting="majority"
        )
        
        # Save full ensemble with confidence
        ensemble_df.to_csv('submission_ensemble_with_confidence.csv', index=False)
        print(f"\nEnsemble prediction with confidence saved to submission_ensemble_with_confidence.csv")
        
        # Analyze prediction patterns
        analyze_prediction_ties(ensemble_df)
        
        # Create simplified submission for competition (only required columns)
        submission_simple = pd.DataFrame({
            'id': ensemble_df['id'],
            'real_text_id': ensemble_df['real_text_id']
        })
        submission_simple.to_csv('submission_ensemble.csv', index=False)
        print(f"Competition submission saved to submission_ensemble.csv")
        
        # Print summary
        print(f"\nResults summary:")
        total_valid = 0
        for model_id, model_results in all_model_results.items():
            model_name = model_id.split('/')[-1]
            for layer, result in model_results.items():
                if result is not None:
                    print(f"  {model_name} layer {layer}: SUCCESS")
                    total_valid += 1
                else:
                    print(f"  {model_name} layer {layer}: FAILED")
        
        print(f"\nTotal valid predictions: {total_valid}")
        print(f"Ensemble distribution: {ensemble_df['real_text_id'].value_counts().to_dict()}")
        
    except Exception as e:
        print(f"Ensemble creation failed: {e}")
        print("Individual model results are still saved.")

    '''
    # 2. CREATE AND TRAIN SVM BINARY DETECTOR
    detector = OutlierDetections(
        detector_type="svm_binary", 
        random_state=42,
        n_components=0.95
    )

    # Train the binary classifier
    detector.learn_binary_classification(
        real_texts_train, 
        fake_texts_train, 
        extractor_model=extractor,
        target_layer=28, 
        pooling='mean',
        batch_size=8
    )


    submission_df = predict_test_batch_svm(
        detector=detector,
        df_test=df_test,
        extractor_model=extractor,
        target_layer=28, 
        batch_size=8,
        pooling='mean'
    )
    '''