import os
import torch
import gc
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def read_texts_from_dir(dir_path):
  """
  Reads the texts from a given directory and saves them in the pd.DataFrame with columns ['id', 'file_1', 'file_2'].

  Params:
    dir_path (str): path to the directory with data
  """
  # Count number of directories in the provided path
  dir_count = sum(os.path.isdir(os.path.join(root, d)) for root, dirs, _ in os.walk(dir_path) for d in dirs)
  data=[0 for _ in range(dir_count)]
  print(f"Number of directories: {dir_count}")

  # For each directory, read both file_1.txt and file_2.txt and save results to the list
  i=0
  for folder_name in sorted(os.listdir(dir_path)):
    folder_path = os.path.join(dir_path, folder_name)
    if os.path.isdir(folder_path):
      try:
        with open(os.path.join(folder_path, 'file_1.txt'), 'r', encoding='utf-8') as f1:
          text1 = f1.read().strip()
        with open(os.path.join(folder_path, 'file_2.txt'), 'r', encoding='utf-8') as f2:
          text2 = f2.read().strip()
        index = int(folder_name[-4:])
        data[i]=(index, text1, text2)
        i+=1
      except Exception as e:
        print(f"Error reading directory {folder_name}: {e}")

  # Change list with results into pandas DataFrame
  df = pd.DataFrame(data, columns=['id', 'file_1', 'file_2']).set_index('id')
  return df



def extract_real_texts(df, labels_df):
        """
        Extract only the real texts from pairs using labels

        Args:
            df: DataFrame with text pairs (indexed by id)
            labels_df: DataFrame with columns ['id', 'real_text_id'] where real_text_id is 1 or 2

        Returns:
            list: Real texts only
        """
        real_texts = []

        for idx, row in df.iterrows():
            if idx in labels_df.index:
                real_text_id = labels_df.loc[idx]['real_text_id']

                if real_text_id == 1:
                    real_texts.append(row['file_1'])
                elif real_text_id == 2:
                    real_texts.append(row['file_2'])
                else:
                    print(f"Warning: Invalid real_text_id {real_text_id} for index {idx}")
            else:
                print(f"Warning: No label found for index {idx}")

        print(f"Extracted {len(real_texts)} real texts")
        
        return real_texts



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