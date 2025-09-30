import os
import torch
import gc
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def find_best_layer(df_train, df_train_gt):
    df_merged = df_train.merge(df_train_gt, on='id')

    # Split
    train_df, test_df = train_test_split(df_merged, test_size=0.2, random_state=42)

    # Extract real texts for training manifold
    real_texts_train = [
        row['file_1'] if row['real_text_id'] == 1 else row['file_2']
        for idx, row in train_df.iterrows()
    ]

    true_labels = test_df['real_text_id'].values
    for layer in layers_to_test:
        torch.cuda.empty_cache()
        gc.collect()
            
        detector = OutlierDetections(detector_type=detector_type, contamination=0.005,n_components=0.95,use_trajectory=False)
        print(f"\nEvaluating layer {layer}...")

        # Learn manifold on training real texts
        detector.learn_real_manifold(real_texts_train, pooling=pooling, batch_size=8,extractor_model=extractor, target_layer=layer)

        # Evaluate on test split
        predictions, _ = detector.evaluate_pairs(
            test_df, 
            test_df[['real_text_id']].set_index(test_df.index),  # labels_df indexed by test_df
            extractor_model=extractor, 
            target_layer=layer
        )

        # Compute accuracy
        accuracy = accuracy_score(true_labels, predictions)
        print(f"Layer {layer}: Accuracy = {accuracy:.4f}")
        layer_scores.append((layer, accuracy))
        del detector

    # Find the best layer
    best_layer, best_acc = max(layer_scores, key=lambda x: x[1])
    print(f"\nBest layer: {best_layer} with accuracy {best_acc:.4f}")