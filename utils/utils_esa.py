import os
import pandas as pd


### utils functions specific to the challenge from ESA
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



def extract_real_fake_texts(df, labels_df, real=True):
        """
        Extract only the real texts from pairs using labels
        
        Args:
            df: DataFrame with text pairs (indexed by id)
            labels_df: DataFrame with columns ['id', 'real_text_id'] where real_text_id is 1 or 2

        Returns:
            list: Real texts only
        """
        texts = []

        for idx, row in df.iterrows():
            if idx in labels_df.index:
                real_text_id = labels_df.loc[idx]['real_text_id']
                if real_text_id == 1:
                    if real: 
                      texts.append(row['file_1'])
                    else:
                      texts.append(row['file_2'])
                elif texts == 2:
                    if real:
                      texts.append(row['file_2'])
                    else:
                       texts.append(row['file_1'])
                
                else:
                    print(f"Warning: Invalid real_text_id {real_text_id} for index {idx}")
            else:
                print(f"Warning: No label found for index {idx}")

        print(f"Extracted {len(texts)} real texts")
        
        return texts
