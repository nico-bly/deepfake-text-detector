import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from tqdm import tqdm

### ------------- model to extract embedding from decoder or encoder language models
class EmbeddingExtractor(torch.nn.Module):
    """
    Extract hidden representation of LLMs, using only forward pass
    outputs is a a vector for each text
    """

    def __init__(self, model_name="distilbert-base-uncased", device=None):
        super().__init__()
        
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModel.from_pretrained(
                model_name,
                dtype=torch.float16,  
                device_map=None  
            )
        
        self.model = self.model.to(self.device)
        
        self.model.eval()

        # Add padding token if missing
        if self.tokenizer.pad_token is None:
            print('adding pasd token')
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Model loaded successfully. Hidden size: {self.model.config.hidden_size}")
        print(f"Number of layers: {self.model.config.num_hidden_layers}")

    def test_model(self, sample_text="This is a test sentence."):
        """
        Test if the model works with a simple text
        """
        print("Testing model with sample text...")
        try:
            embeddings = self.get_all_layer_embeddings(sample_text)
            print(f"Model test successful!")
            print(f"Number of layers: {len(embeddings)}")
            print(f"Embedding shape for layer -1: {embeddings[-1].shape}")
            return True
        except Exception as e:
            print(f" Model test failed: {e}")
            return False
        
    def get_all_layer_embeddings(self, texts, pooling='mean', batch_size=16):
        """
        Extract embeddings from ALL layers for one or more texts, in batches.
        Args:
            texts: list[str] or str
            pooling: 'mean', 'cls', 'max', or 'all'
            batch_size: number of texts per forward pass
        Returns:
            dict: {layer_idx: np.ndarray of shape (num_texts, hidden_size)} 
                OR for 'all' pooling: {layer_idx: np.ndarray of shape (total_tokens, hidden_size)}
        """
        if isinstance(texts, str):
            texts = [texts]

        layer_embeddings = {}  # Will store final results
        
        # Handle 'all' pooling separately as it has different output structure
        if pooling == 'all':
            return self._get_all_token_embeddings(texts, batch_size)

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]

            model_inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )

            # Move each tensor explicitly to device
            model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}
            input_ids = model_inputs['input_ids']
            attention_mask = model_inputs['attention_mask']

            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
            
            # Process each layer
            batch_layer_embeddings = {}
            for layer_idx, hidden_state in enumerate(outputs.hidden_states):
                # hidden_state shape: [batch_size, seq_len, hidden_size]
                # attention_mask shape: [batch_size, seq_len]
                if pooling == 'mean':
                    # Expand attention mask to match hidden state dimensions
                    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
                    # Apply mask
                    masked_embeddings = hidden_state * mask_expanded
                    
                    
                    # Sum over sequence length and divide by actual lengths
                    sum_embeddings = masked_embeddings.sum(dim=1)  # [batch_size, hidden_size]
                    seq_lengths = attention_mask.sum(dim=1).unsqueeze(-1).float()  # [batch_size, 1]

                    embedding = sum_embeddings / seq_lengths.clamp(min=1)
                    
                elif pooling == 'cls':
                    embedding = hidden_state[:, 0, :]  # [batch_size, hidden_size]
                    
                elif pooling == 'max':
                    # Apply mask before max pooling
                    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
                    masked_embeddings = hidden_state * mask_expanded + (1 - mask_expanded) * (-1e9)
                    embedding = torch.max(masked_embeddings, dim=1)[0]  # [batch_size, hidden_size]
                elif pooling == 'last_token':
                    # Get the last non-padding token for each sequence
                    batch_size = hidden_state.shape[0]
                    
                    # Check if we have left padding (rare) or right padding (common)
                    # For right padding, we want the last valid token before padding starts
                    sequence_lengths = attention_mask.sum(dim=1) - 1  # -1 because we want last valid index
                    
                    # Extract last valid token for each sequence in batch
                    batch_indices = torch.arange(batch_size, device=hidden_state.device)
                    embedding = hidden_state[batch_indices, sequence_lengths]  # [batch_size, hidden_size]
                else:
                    raise ValueError(f"Unsupported pooling method: {pooling}")

                batch_layer_embeddings[layer_idx] = embedding.cpu().numpy()

            # Concatenate with previous batches
            if not layer_embeddings:  # First batch
                layer_embeddings = batch_layer_embeddings
            else:
                for layer_idx in layer_embeddings.keys():
                    layer_embeddings[layer_idx] = np.vstack([
                        layer_embeddings[layer_idx], 
                        batch_layer_embeddings[layer_idx]
                    ])

        return layer_embeddings
    
    def _get_all_token_embeddings(self, texts, batch_size):
        """
        Separate method for 'all' pooling to handle different output structure
        """
        layer_embeddings = {}
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]

            model_inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )

            model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}
            attention_mask = model_inputs['attention_mask']

            with torch.no_grad():
                outputs = self.model(**model_inputs, output_hidden_states=True)

            batch_layer_embeddings = {}
            for layer_idx, hidden_state in enumerate(outputs.hidden_states):
                # Extract only valid (non-padding) tokens
                embedding_list = []
                mask = attention_mask.bool()  # [batch_size, seq_len]
                
                for b in range(hidden_state.size(0)):
                    # Ensure mask indices are proper integer types
                    valid_mask = mask[b].cpu()  # Move to CPU to avoid CUDA tensor type issues
                    valid_indices = torch.where(valid_mask)[0]  # Get indices of valid tokens
                    
                    if len(valid_indices) > 0:
                        # Use integer indexing instead of boolean masking on CUDA tensors
                        valid_tokens = hidden_state[b][valid_indices]  # [valid_seq_len, hidden_size]
                        embedding_list.append(valid_tokens)
                
                # Concatenate all valid tokens from this batch
                if embedding_list:
                    batch_embeddings = torch.cat(embedding_list, dim=0)  # [total_valid_tokens, hidden_size]
                    batch_layer_embeddings[layer_idx] = batch_embeddings.cpu().numpy()

            # Concatenate with previous batches
            if not layer_embeddings:  # First batch
                layer_embeddings = batch_layer_embeddings
            else:
                for layer_idx in layer_embeddings.keys():
                    if layer_idx in batch_layer_embeddings:
                        layer_embeddings[layer_idx] = np.vstack([
                            layer_embeddings[layer_idx], 
                            batch_layer_embeddings[layer_idx]
                        ])

        return layer_embeddings
