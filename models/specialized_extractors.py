"""
Specialized embedding extractors for models with recommended extraction methods.
These extractors use model-specific pooling and normalization strategies.
"""

import torch
import torch.nn.functional as F
from typing import List, Union
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


class EmbeddingSpecializedExtractor:
    """Base class for model-specific embeddings."""
    
    def extract(self, texts: List[str], batch_size: int = 8, max_length: int = 8192) -> np.ndarray:
        """Extract embeddings using model-specific method.
        
        Returns:
            (n_texts, embedding_dim) numpy array
        """
        raise NotImplementedError


class QwenEmbeddingExtractor(EmbeddingSpecializedExtractor):
    """Qwen3-Embedding models use last-token pooling + L2 normalization.
    
    Reference: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
    """
    
    def __init__(self, model_name: str, device: str = "cuda:0"):
        self.device = device
        self.model_name = model_name
        print(f"Loading {model_name} (Qwen embedding extractor)")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        self.model = AutoModel.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"✓ Model loaded. Hidden size: {self.model.config.hidden_size}")
    
    def _last_token_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Extract last non-padding token."""
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[
                torch.arange(batch_size, device=last_hidden_states.device),
                sequence_lengths
            ]
    
    def extract(self, texts: List[str], batch_size: int = 8, max_length: int = 8192) -> np.ndarray:
        """Extract Qwen embeddings: last-token pool + L2 normalize."""
        if isinstance(texts, str):
            texts = [texts]
        
        all_embeddings = []
        iterator = range(0, len(texts), batch_size)
        iterator = tqdm(iterator, desc="Extracting Qwen embeddings", total=(len(texts) + batch_size - 1) // batch_size)
        
        for i in iterator:
            batch = texts[i:i+batch_size]
            
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Last-token pooling
            embeddings = self._last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
            
            # L2 normalize
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            all_embeddings.append(embeddings.cpu().float().numpy())
        
        return np.vstack(all_embeddings).astype(np.float32)


class SentenceTransformerExtractor(EmbeddingSpecializedExtractor):
    """Sentence-Transformers models use mean pooling + L2 normalization.
    
    Reference: https://huggingface.co/sentence-transformers/all-mpnet-base-v2
    """
    
    def __init__(self, model_name: str, device: str = "cuda:0"):
        self.device = device
        self.model_name = model_name
        print(f"Loading {model_name} (Sentence-Transformer extractor)")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"✓ Model loaded. Hidden size: {self.model.config.hidden_size}")
    
    def _mean_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean pooling with attention mask."""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def extract(self, texts: List[str], batch_size: int = 8, max_length: int = 512) -> np.ndarray:
        """Extract Sentence-Transformer embeddings: mean pool + L2 normalize."""
        if isinstance(texts, str):
            texts = [texts]
        
        all_embeddings = []
        iterator = range(0, len(texts), batch_size)
        iterator = tqdm(iterator, desc="Extracting Sentence-Transformer embeddings", 
                       total=(len(texts) + batch_size - 1) // batch_size)
        
        for i in iterator:
            batch = texts[i:i+batch_size]
            
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Mean pooling
            embeddings = self._mean_pooling(outputs[0], inputs['attention_mask'])
            
            # L2 normalize
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            all_embeddings.append(embeddings.cpu().float().numpy())
        
        return np.vstack(all_embeddings).astype(np.float32)


class MLMExtractor(EmbeddingSpecializedExtractor):
    """Masked Language Model extractors (RoBERTa, XLM-RoBERTa, etc.) use CLS-token pooling + L2 norm.
    
    Reference: https://huggingface.co/xlm-roberta-base
    """
    
    def __init__(self, model_name: str, device: str = "cuda:0"):
        self.device = device
        self.model_name = model_name
        print(f"Loading {model_name} (MLM extractor - CLS pooling)")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"✓ Model loaded. Hidden size: {self.model.config.hidden_size}")
    
    def extract(self, texts: List[str], batch_size: int = 8, max_length: int = 512) -> np.ndarray:
        """Extract MLM embeddings: CLS token + L2 normalize."""
        if isinstance(texts, str):
            texts = [texts]
        
        all_embeddings = []
        iterator = range(0, len(texts), batch_size)
        iterator = tqdm(iterator, desc="Extracting MLM embeddings", 
                       total=(len(texts) + batch_size - 1) // batch_size)
        
        for i in iterator:
            batch = texts[i:i+batch_size]
            
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # CLS token (first token)
            embeddings = outputs[0][:, 0, :]  # (batch, hidden_size)
            
            # L2 normalize
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            all_embeddings.append(embeddings.cpu().float().numpy())
        
        return np.vstack(all_embeddings).astype(np.float32)


# Registry of specialized extractors
SPECIALIZED_EXTRACTORS = {
    # Qwen embedding models
    "Qwen/Qwen3-Embedding-0.6B": QwenEmbeddingExtractor,
    "Qwen/Qwen3-Embedding-1B": QwenEmbeddingExtractor,
    "Qwen/Qwen3-Embedding-4B": QwenEmbeddingExtractor,
    
    # Sentence-Transformers
    "sentence-transformers/all-mpnet-base-v2": SentenceTransformerExtractor,
    "sentence-transformers/all-distilroberta-v1": SentenceTransformerExtractor,
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": SentenceTransformerExtractor,
    
    # MLM models
    "xlm-roberta-base": MLMExtractor,
    "FacebookAI/xlm-roberta-base": MLMExtractor,
    "FacebookAI/roberta-base": MLMExtractor,

    # minillm
    "sentence-transformers/all-MiniLM-L6-v2": SentenceTransformerExtractor,
}


def get_specialized_extractor(model_name: str, device: str = "cuda:0"):
    """Get specialized extractor if available, else None."""
    if model_name in SPECIALIZED_EXTRACTORS:
        extractor_class = SPECIALIZED_EXTRACTORS[model_name]
        return extractor_class(model_name, device)
    return None
