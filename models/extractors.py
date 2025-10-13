from typing import Dict, List, Union
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from tqdm import tqdm

### ------------- model to extract embedding from decoder or encoder language models
class EmbeddingExtractor(torch.nn.Module):
    """
    Extract hidden representation of LLMs, using only forward pass
    outputs is a a vector for each text
    """

    def __init__(self, model_name="distilbert-base-uncased", device=None, use_flash_attention=False, is_qwen3=None):
        super().__init__()
        
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        if is_qwen3 is None:
            self.is_qwen3 = 'qwen3' in model_name.lower() and 'embedding' in model_name.lower()
        else:
            self.is_qwen3 = is_qwen3

        padding_side = 'left' if self.is_qwen3 else 'right'

        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side=padding_side)

        if use_flash_attention and self.device == 'cuda':
            print("Loading with Flash Attention 2...")
            self.model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                attn_implementation="flash_attention_2",
                device_map=None
            )
        else:
            self.model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
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
        
    def get_all_layer_embeddings(
        self, 
        texts: Union[str, List[str]], 
        batch_size: int = 16,
        max_length: int = 8192,  # Support long context
        show_progress=True
    ) -> List[Dict[int, np.ndarray]]:
        """
        Extract embeddings from ALL layers for one or more texts, in batches.
        we want as output only vectors that wont be batched anymore so this must also remove the padding made by tokenizer
        
        Args:
            texts: list[str] or str
            batch_size: number of texts per forward pass
            max_length: maximum sequence length
            
        Returns:
        all_texts_embeddings
                [
            {0: emb_layer0, 1: emb_layer1, ..., N: emb_layerN},   # embeddings for text 1
            {0: emb_layer0, 1: emb_layer1, ..., N: emb_layerN},   # embeddings for text 2
            ...
            ]

        """
        if isinstance(texts, str):
            texts = [texts]
        
        all_texts_embeddings = []


        iterator = range(0, len(texts), batch_size)
        if show_progress and len(texts) > batch_size:
            iterator = tqdm(iterator, desc="Extracting embeddings")
        for i in iterator:
            batch = texts[i:i+batch_size]
            model_inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
                
            )
            model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}
            with torch.no_grad():
                outputs = self.model(**model_inputs, output_hidden_states=True)

            attention_mask = model_inputs['attention_mask']
            
            # For each text in the batch
            for b in range(len(batch)):
                    text_layer_embeddings = {}
                    valid_indices = torch.where(attention_mask[b].bool())[0]
                    
                    if len(valid_indices) == 0:
                        print("attenton no valid indices")
                        print(outputs.hidden_states[b].shape)
                        print(f"Text: {repr(batch[b])}")
                        print(attention_mask[b].bool())

                    for layer_idx, hidden_state in enumerate(outputs.hidden_states):
                        
                        # Keep full sequence, no pooling
                        valid_tokens = hidden_state[b][valid_indices]
                        text_layer_embeddings[layer_idx] = valid_tokens.cpu().numpy()  # (seq, hidden)
                    
                    all_texts_embeddings.append(text_layer_embeddings)

        assert len(all_texts_embeddings) == len(texts)
        return all_texts_embeddings

#### --------------------------------------------- from embeddings generate features for classification --------------------------------------------

def extract_embed_at_layer(list_embeds, layer_idx):
    """
    Args:
        list_embeds: obtained from EmbeddingExtractor
            [
            {0: emb_layer0, 1: emb_layer1, ..., N: emb_layerN},   # embeddings for text 1
            {0: emb_layer0, 1: emb_layer1, ..., N: emb_layerN},   # embeddings for text 2
            ...
            ]
        layer_idx: int for the layer
    Returns:
        List of numpy arrays: all hidden states per text at a specific layer  
            emb_layerk_textk has shape shape (seq,hidden_size)
                (here for each text seq will be different)
            [ emb_layerk_text0, ... , emb_layerk_textk, ..., emb_layerk_lasttext ]
    """

    if layer_idx not in list_embeds[0]:
        raise ValueError(f"Layer {layer_idx} not found. Available: {list(list_embeds[0].keys())}")
    
    selected_embeds_at_choosen_layer = [list_embeds_per_text[layer_idx] for list_embeds_per_text in list_embeds]

    return selected_embeds_at_choosen_layer

def pool_embeds_from_layer(selected_embeds_at_choosen_layer, pooling='mean'):
    """
    Args:
        selected_embeds_at_choosen_layer: obtained from EmbeddingExtractor
            [ emb_layerk_text0, ... , emb_layerk_textk, ..., emb_layerk_lasttext]
        pooling: method to pool vectors

    Returns:
        numpy array of shape (num_texts, hidden_size)

    """
    list_pooled = []
    for idx, emb in enumerate(selected_embeds_at_choosen_layer):   # emb shape: (seq, hidden)
        if emb.size == 0:
            print(f"⚠️  Warning: Empty embedding at index {idx}, using zeros")
            print(emb)
            pooled_emb = np.zeros(emb.shape[1] if len(emb.shape) > 1 else 768)
        elif np.isnan(emb).all():
            print(f"⚠️  Warning: All-NaN embedding at index {idx}, using zeros")
            pooled_emb = np.zeros(emb.shape[1] if len(emb.shape) > 1 else 768)
        if pooling == 'mean':
            pooled_emb = emb.mean(axis=0)  # (hidden,)
        elif pooling == 'max':
            pooled_emb = emb.max(axis=0)  # (hidden,)
        elif pooling == 'first':
            pooled_emb = emb[0]  # (hidden,)
        elif pooling == 'last':
            pooled_emb = emb[-1]  # (hidden,)
        else:
            raise ValueError(f"Unsupported pooling: {pooling}")
        
        list_pooled.append(pooled_emb)
    
    return np.array(list_pooled)




if __name__ == "__main__":
    # Initialize
    #"Qwen/Qwen2.5-0.5B"
    #"meta-llama/Llama-3.1-8B"
    calc = PerplexityCalculator(model_name="Qwen/Qwen2.5-0.5B",device='cuda:2')
    
    # Sample texts
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "asdf jkl qwerty zxcv random nonsense words here",
        "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
    ]
    
    # Calculate perplexities
    perplexities = calc.calculate_batch_perplexity(texts)
    
    # Display results
    for text, ppl in zip(texts, perplexities):
        print(f"Text: {text[:50]}...")
        print(f"Perplexity: {ppl:.2f}\n")