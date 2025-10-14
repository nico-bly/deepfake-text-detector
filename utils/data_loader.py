"""
Unified PyTorch DataLoader for ESA Challenge Fake Text Detection

This module provides a unified approach where each text is treated as an individual
sample with a binary label: 0 = real text, 1 = AI-generated text.
This simplifies the classification task to standard binary classification.
"""

import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Dict, Any, Union
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        batch_texts = batch['text']
        batch_labels = batch['label'].tolist()
        
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

class UnifiedTextDataset(Dataset):
    """
    Unified Dataset where each text is a separate sample with binary label.
    
    Labels:
    - 0: Real (human-written) text
    - 1: Fake (AI-generated) text
    
    This approach converts text pairs into individual samples, making it easier
    to use standard binary classification approaches.
    """
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        dataframe: Optional[pd.DataFrame] = None,
        labels_df: Optional[pd.DataFrame] = None,
        max_length: int = 512,
        lazy_loading: bool = True,
        include_metadata: bool = True,
    ):
        """
        Initialize the UnifiedTextDataset.
        
        Args:
            data_path: Path to directory containing text files
            dataframe: Pre-loaded DataFrame with columns ['id', 'file_1', 'file_2']
            labels_df: DataFrame with ground truth labels ['id', 'real_text_id']
            max_length: Maximum text length for truncation (in words)
            lazy_loading: If True, load texts on demand
            include_metadata: If True, include source information in samples
        """
        self.max_length = max_length
        self.lazy_loading = lazy_loading
        self.include_metadata = include_metadata
        
        # Load data from path or use provided DataFrame
        if data_path is not None:
            self.raw_data_df = self._load_from_directory(data_path)
            self.data_path = Path(data_path)
        elif dataframe is not None:
            self.raw_data_df = dataframe.copy()
            self.data_path = None
        else:
            raise ValueError("Either data_path or dataframe must be provided")
        
        # Convert text pairs to individual samples with labels
        self.samples_df = self._create_unified_samples(labels_df)
        
        # Pre-load texts if not using lazy loading
        if not lazy_loading:
            self._preload_texts()
        
        logger.info(f"UnifiedTextDataset initialized:")
        logger.info(f"  Original pairs: {len(self.raw_data_df)}")
        logger.info(f"  Individual texts: {len(self.samples_df)}")
        
        if 'label' in self.samples_df.columns:
            label_counts = self.samples_df['label'].value_counts()
            logger.info(f"  Real texts (0): {label_counts.get(0, 0)}")
            logger.info(f"  Fake texts (1): {label_counts.get(1, 0)}")
    
    def _load_from_directory(self, dir_path: str) -> pd.DataFrame:
        """Load text pairs from directory structure."""
        data_list = []
        dir_path = Path(dir_path)
        
        # Find all article directories
        article_dirs = sorted([d for d in dir_path.iterdir() if d.is_dir()])
        
        for article_dir in article_dirs:
            try:
                # Extract ID from directory name (e.g., article_0001 -> 1)
                article_id = int(article_dir.name.split('_')[-1])
                
                # Read file_1.txt and file_2.txt
                file_1_path = article_dir / 'file_1.txt'
                file_2_path = article_dir / 'file_2.txt'
                
                if file_1_path.exists() and file_2_path.exists():
                    with open(file_1_path, 'r', encoding='utf-8') as f:
                        text_1 = f.read().strip()
                    with open(file_2_path, 'r', encoding='utf-8') as f:
                        text_2 = f.read().strip()
                    
                    data_list.append({
                        'id': article_id,
                        'file_1': text_1,
                        'file_2': text_2,
                        'file_1_path': str(file_1_path),
                        'file_2_path': str(file_2_path)
                    })
                else:
                    logger.warning(f"Missing files in {article_dir}")
                    
            except (ValueError, FileNotFoundError) as e:
                logger.warning(f"Error processing {article_dir}: {e}")
        
        df = pd.DataFrame(data_list)
        logger.info(f"Loaded {len(df)} text pairs from {dir_path}")
        return df
    
    def _create_unified_samples(self, labels_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """
        Convert text pairs into individual samples with binary labels.
        
        For each pair (text_1, text_2):
        - If labels are available: mark real text as 0, fake text as 1
        - If no labels: create samples without labels for inference
        """
        samples_list = []
        
        for _, row in self.raw_data_df.iterrows():
            pair_id = row['id']
            text_1 = row['file_1']
            text_2 = row['file_2']
            
            # Determine labels if available
            if labels_df is not None and pair_id in labels_df['id'].values:
                real_text_id = labels_df[labels_df['id'] == pair_id]['real_text_id'].iloc[0]
                
                if real_text_id == 1:
                    # Text 1 is real, Text 2 is fake
                    text_1_label, text_2_label = 0, 1
                elif real_text_id == 2:
                    # Text 2 is real, Text 1 is fake  
                    text_1_label, text_2_label = 1, 0
                else:
                    logger.warning(f"Invalid real_text_id {real_text_id} for pair {pair_id}")
                    continue
                
                # Create samples with labels
                samples_list.extend([
                    {
                        'sample_id': f"{pair_id}_1",
                        'pair_id': pair_id,
                        'text_position': 1,
                        'text': text_1,
                        'label': text_1_label,
                        'file_path': row.get('file_1_path', None),
                    },
                    {
                        'sample_id': f"{pair_id}_2", 
                        'pair_id': pair_id,
                        'text_position': 2,
                        'text': text_2,
                        'label': text_2_label,
                        'file_path': row.get('file_2_path', None),
                    }
                ])
            else:
                # Create samples without labels (for inference)
                samples_list.extend([
                    {
                        'sample_id': f"{pair_id}_1",
                        'pair_id': pair_id,
                        'text_position': 1, 
                        'text': text_1,
                        'file_path': row.get('file_1_path', None),
                    },
                    {
                        'sample_id': f"{pair_id}_2",
                        'pair_id': pair_id,
                        'text_position': 2,
                        'text': text_2,
                        'file_path': row.get('file_2_path', None),
                    }
                ])
        
        return pd.DataFrame(samples_list)
    
    def _preload_texts(self):
        """Pre-load all texts into memory."""
        logger.info("Pre-loading all texts into memory...")
        # Texts are already loaded in _create_unified_samples
        pass
    
    def _load_text_on_demand(self, idx: int) -> str:
        """Load text on demand for lazy loading."""
        if self.lazy_loading and 'file_path' in self.samples_df.columns:
            row = self.samples_df.iloc[idx]
            file_path = row.get('file_path')
            
            if file_path and os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return f.read().strip()
                except FileNotFoundError:
                    logger.warning(f"File not found: {file_path}, using cached text")
        
        # Fallback to cached text
        return self.samples_df.iloc[idx]['text']
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples_df)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample from the dataset.
        
        Returns:
            Dictionary containing:
            - sample_id: Unique sample identifier (e.g., "1_1" for pair 1, text 1)
            - text: The text content
            - label: Binary label (0=real, 1=fake) if available
            - pair_id: Original pair ID
            - text_position: Position in pair (1 or 2)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        row = self.samples_df.iloc[idx]
        
        # Load text
        text = self._load_text_on_demand(idx)
        
        # Truncate text if needed
        if len(text.split()) > self.max_length:
            text = ' '.join(text.split()[:self.max_length])
        
        sample = {
            'sample_id': row['sample_id'],
            'text': text,
            'pair_id': int(row['pair_id']),
            'text_position': int(row['text_position']),
        }
        
        # Add label if available
        if 'label' in row:
            sample['label'] = int(row['label'])
        
        # Add metadata if requested
        if self.include_metadata:
            sample['text_length'] = len(text.split())
            sample['char_length'] = len(text)
        
        return sample


def collate_unified_texts(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for unified text batches.
    
    stay general
    Args:
        batch: List of samples from UnifiedTextDataset
        
    Returns:
        Batched dictionary with lists/tensors of data
    """
    collated = {
        'text': [sample['text'] for sample in batch],
    }
    
    # Add labels if present in any sample
    if 'label' in batch[0]:
        collated['label'] = torch.tensor([sample['label'] for sample in batch], dtype=torch.long)
    
    # case of dataset from real or fake
    if 'sample_id' in batch[0]:
        collated['sample_id'] = [sample['sample_id'] for sample in batch]
        collated['pair_id'] = torch.tensor([sample['pair_id'] for sample in batch], dtype=torch.long)
        collated['text_position'] = torch.tensor([sample['text_position'] for sample in batch], dtype=torch.long)

    # Add metadata if present
    if 'text_length' in batch[0]:
        collated['text_length'] = torch.tensor([sample['text_length'] for sample in batch], dtype=torch.long)
        collated['char_length'] = torch.tensor([sample['char_length'] for sample in batch], dtype=torch.long)
    
    return collated

class LargeCSVDataset(Dataset):
    def __init__(self, csv_path, text_col="text", label_col="label", indices=None):
        self.csv_path = csv_path
        self.text_col = text_col
        self.label_col = label_col
        self.df = pd.read_csv(csv_path, usecols=[text_col, label_col])
        if indices is not None:
            self.df = self.df.iloc[indices].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            "text": row[self.text_col],
            "label": row[self.label_col]
        }

def create_unified_dataloaders(
    train_path: Optional[str] = None,
    test_path: Optional[str] = None,
    train_labels_path: Optional[str] = None,
    train_df: Optional[pd.DataFrame] = None,
    test_df: Optional[pd.DataFrame] = None,
    labels_df: Optional[pd.DataFrame] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle_train: bool = True,
    max_length: int = 512,
    lazy_loading: bool = True,
    include_metadata: bool = True,
) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
    """
    Create unified train and test DataLoaders for binary text classification.
    
    Args:
        train_path: Path to training data directory
        test_path: Path to test data directory
        train_labels_path: Path to training labels CSV
        train_df: Pre-loaded training DataFrame
        test_df: Pre-loaded test DataFrame
        labels_df: Pre-loaded labels DataFrame
        batch_size: Batch size for DataLoaders
        num_workers: Number of worker processes
        shuffle_train: Whether to shuffle training data
        max_length: Maximum text length (words)
        lazy_loading: Whether to use lazy loading
        include_metadata: Whether to include text metadata
        
    Returns:
        Tuple of (train_dataloader, test_dataloader)
    """
    train_dataloader = None
    test_dataloader = None
    
    # Load labels if path provided
    if train_labels_path is not None and labels_df is None:
        labels_df = pd.read_csv(train_labels_path)
        logger.info(f"Loaded labels from {train_labels_path}")
    
    # Create training DataLoader
    if train_path is not None or train_df is not None:
        train_dataset = UnifiedTextDataset(
            data_path=train_path,
            dataframe=train_df,
            labels_df=labels_df,
            max_length=max_length,
            lazy_loading=lazy_loading,
            include_metadata=include_metadata,
        )
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            collate_fn=collate_unified_texts,
            pin_memory=torch.cuda.is_available(),
        )
        
        logger.info(f"Created unified training DataLoader: {len(train_dataset)} samples")
    
    # Create test DataLoader  
    if test_path is not None or test_df is not None:
        test_dataset = UnifiedTextDataset(
            data_path=test_path,
            dataframe=test_df,
            labels_df=None,  # Test data typically doesn't have labels
            max_length=max_length,
            lazy_loading=lazy_loading,
            include_metadata=include_metadata,
        )
        
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,  # Don't shuffle test data
            num_workers=num_workers,
            collate_fn=collate_unified_texts,
            pin_memory=torch.cuda.is_available(),
        )
        
        logger.info(f"Created unified test DataLoader: {len(test_dataset)} samples")
    
    return train_dataloader, test_dataloader

def get_text_dataloader(
    data_source,
    dataset_type="csv",
    batch_size=32,
    shuffle=True,
    num_workers=4,
    text_col="text",
    label_col="label",
    indices=None,
    **kwargs
):
    """
    Returns a DataLoader for either a flat CSV or paired text directory.

    Args:
        data_source: Path to CSV file or directory (or DataFrame for paired)
        dataset_type: "csv" for flat CSV, "paired" for paired text directory
        batch_size: Batch size
        shuffle: Shuffle data
        num_workers: DataLoader workers
        text_col: Column name for text (CSV)
        label_col: Column name for label (CSV)
        indices: Optional indices for splitting (CSV)
        **kwargs: Other args for UnifiedTextDataset

    Returns:
        DataLoader
    """
    if dataset_type == "csv":
        dataset = LargeCSVDataset(
            csv_path=data_source,
            text_col=text_col,
            label_col=label_col,
            indices=indices
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
    elif dataset_type == "paired":
        dataset = UnifiedTextDataset(
            data_path=data_source if isinstance(data_source, str) else None,
            dataframe=data_source if isinstance(data_source, pd.DataFrame) else None,
            **kwargs
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_unified_texts,
            pin_memory=torch.cuda.is_available(),
        )
    else:
        raise ValueError("dataset_type must be 'csv' or 'paired'")
    return loader



def extract_real_fake_texts(
    dataloader: DataLoader
) -> Tuple[List[str], List[str]]:
    """
    Extract separate lists of real and fake texts from a unified DataLoader.
    
    Args:
        dataloader: DataLoader with labeled samples
        
    Returns:
        Tuple of (real_texts, fake_texts)
    """
    real_texts = []
    fake_texts = []
    
    for batch in dataloader:
        if 'labels' not in batch:
            logger.warning("No labels found in batch, skipping extraction")
            continue
        
        texts = batch['texts']
        labels = batch['labels']
        
        for text, label in zip(texts, labels):
            if label == 0:  # Real text
                real_texts.append(text)
            elif label == 1:  # Fake text
                fake_texts.append(text)
    
    logger.info(f"Extracted {len(real_texts)} real and {len(fake_texts)} fake texts")
    return real_texts, fake_texts


def reconstruct_pairs_from_predictions(
    predictions: List[float],
    sample_ids: List[str],
) -> pd.DataFrame:
    """
    Reconstruct pair-wise predictions from individual text predictions.
    
    Args:
        predictions: List of predictions for each text (0=real, 1=fake probability)
        sample_ids: List of sample IDs (e.g., ["1_1", "1_2", "2_1", "2_2", ...])
        
    Returns:
        DataFrame with columns ['id', 'real_text_id', 'confidence']
    """
    # Group predictions by pair
    pair_predictions = {}
    
    for pred, sample_id in zip(predictions, sample_ids):
        pair_id, text_pos = sample_id.split('_')
        pair_id = int(pair_id)
        text_pos = int(text_pos)
        
        if pair_id not in pair_predictions:
            pair_predictions[pair_id] = {}
        
        pair_predictions[pair_id][text_pos] = pred
    
    # Determine which text is real for each pair
    results = []
    
    for pair_id, preds in pair_predictions.items():
        if 1 in preds and 2 in preds:
            # Lower prediction score means more likely to be real (0=real, 1=fake)
            text_1_fake_prob = preds[1]
            text_2_fake_prob = preds[2]
            
            # Text with lower fake probability is more likely real
            if text_1_fake_prob < text_2_fake_prob:
                real_text_id = 1
                confidence = abs(text_2_fake_prob - text_1_fake_prob)
            else:
                real_text_id = 2  
                confidence = abs(text_1_fake_prob - text_2_fake_prob)
            
            results.append({
                'id': pair_id,
                'real_text_id': real_text_id,
                'confidence': confidence,
                'text1_fake_prob': text_1_fake_prob,
                'text2_fake_prob': text_2_fake_prob,
            })
        else:
            logger.warning(f"Incomplete predictions for pair {pair_id}")
    
    return pd.DataFrame(results).sort_values('id').reset_index(drop=True)


# Example usage and testing functions
def test_unified_dataloader():
    """Test the unified DataLoader implementation."""
    print("Testing Unified DataLoader implementation...")
    
    # Test with dummy data
    dummy_data = pd.DataFrame({
        'id': [0, 1, 2],
        'file_1': ['This is real human text.', 'Another human text.', 'Third human text.'],
        'file_2': ['This is AI generated text.', 'This is fake AI text.', 'Another AI text.']
    })
    
    dummy_labels = pd.DataFrame({
        'id': [0, 1, 2],
        'real_text_id': [1, 2, 1]  # text 1 real, text 2 real, text 1 real
    })
    
    # Create unified dataset
    dataset = UnifiedTextDataset(dataframe=dummy_data, labels_df=dummy_labels)
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_unified_texts,
    )
    
    # Test iteration
    for batch_idx, batch in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}:")
        print(f"  Sample IDs: {batch['sample_ids']}")
        print(f"  Labels: {batch['labels'].tolist()}")
        print(f"  Texts: {[text[:50] + '...' for text in batch['texts']]}")
        print()
    
    # Test extraction
    real_texts, fake_texts = extract_real_fake_texts(dataloader)
    print(f"Extracted {len(real_texts)} real and {len(fake_texts)} fake texts")
    
    # Test reconstruction
    dummy_predictions = [0.2, 0.8, 0.9, 0.1, 0.3, 0.7]  # Fake probabilities
    sample_ids = ['0_1', '0_2', '1_1', '1_2', '2_1', '2_2']
    
    reconstructed = reconstruct_pairs_from_predictions(dummy_predictions, sample_ids)
    print("Reconstructed pairs:")
    print(reconstructed)
    
    print("Unified DataLoader test completed successfully!")


if __name__ == "__main__":
    #test_unified_dataloader()

    data_human_ai='/home/infres/billy-22/projets/esa_challenge_kaggle/deepfake-text-detector/data_human/AI_Human.csv'

    from sklearn.model_selection import train_test_split

    # Load indices for train/val split (efficient, doesn't load all data at once)
    df = pd.read_csv(data_human_ai, usecols=["text", "generated"])
    train_idx, val_idx = train_test_split(df.index, test_size=0.2, stratify=df["generated"], random_state=42)

    # Create datasets
    train_loader = get_text_dataloader(
        data_source=data_human_ai,
        dataset_type="csv",
        batch_size=32,
        shuffle=True,
        indices=train_idx,
        text_col="text",
        label_col="generated"
    )
   