from typing import Dict, List, Union, Optional
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from tqdm import tqdm
from scipy.spatial.distance import cdist
from threading import Thread

from transformers import AutoModelForCausalLM

#### --------------------------------------------- computes perplexity --------------------------------------------
class PerplexityCalculator:
    def __init__(self, model_name="gpt2", device=None):
        """Use actual causal LMs for perplexity"""
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading {model_name} on {self.device}...")
        
        # Ensure we're using a causal LM
        causal_models = ["gpt2", "Qwen/Qwen2.5-0.5B", "microsoft/DialoGPT-medium"]
        if not any(cm in model_name for cm in causal_models):
            print(f"Warning: {model_name} might not be a causal LM")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16,  # Use half precision
            device_map=self.device
        )
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def calculate_perplexity(self, text: str, max_length: int = 512) -> float:
        """Sanitize input and handle edge cases"""
        # Sanitize empty text (following your conventions)
        if not text or not text.strip():
            text = " "
        
        encodings = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=max_length
        )
        input_ids = encodings['input_ids'].to(self.device)
        
        # Skip if too short
        if input_ids.shape[1] < 2:
            return float('inf')
        
        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            loss = outputs.loss.item()
        return np.exp(loss)
        
    def calculate_perplexity_sliding_window(
        self, 
        text: str, 
        stride: int = 256, 
        max_length: int = 512
    ) -> float:
        """
        Calculate perplexity for long texts using sliding window
        Better for articles longer than max_length tokens
        Args:
            text: input text
            stride: sliding window stride
            max_length: window size
        Returns:
            average_perplexity: averaged across windows
        """
        encodings = self.tokenizer(text, return_tensors='pt')
        input_ids = encodings['input_ids'][0]
        
        # If text is short, use simple method
        if len(input_ids) <= max_length:
            return self.calculate_perplexity(text, max_length)
        
        # Sliding window approach
        nlls = []  # negative log-likelihoods
        total_length = 0
        
        prev_end_loc = 0
        for begin_loc in range(0, len(input_ids), stride):
            end_loc = min(begin_loc + max_length, len(input_ids))
            trg_len = end_loc - prev_end_loc
            
            input_ids_chunk = input_ids[begin_loc:end_loc].unsqueeze(0).to(self.device).long()
            target_ids = input_ids_chunk.clone()
            
            # Only calculate loss on the non-overlapping portion
            target_ids[:, :-trg_len] = -100
            
            with torch.no_grad():
                outputs = self.model(input_ids_chunk, labels=target_ids)
                nll = outputs.loss * trg_len
                nlls.append(nll)
                total_length += trg_len
            
            prev_end_loc = end_loc
            if end_loc == len(input_ids):
                break
        
        # Average perplexity across all windows
        avg_nll = torch.stack(nlls).sum() / total_length
        perplexity = torch.exp(avg_nll).item()
        return perplexity
    
    def calculate_batch_perplexity(
        self, 
        texts: List[str], 
        max_length: int = 512,
        use_sliding_window: bool = False,
        stride: int = 256
    ) -> List[float]:
        """
        Calculate perplexity for multiple texts efficiently
        Args:
            texts: list of text strings
            max_length: maximum token length
            use_sliding_window: whether to use sliding window for long texts
            stride: stride for sliding window
        Returns:
            perplexities: list of perplexity values
        """
        perplexities = []
        
        for i, text in enumerate(texts):
            if use_sliding_window:
                ppl = self.calculate_perplexity_sliding_window(text, stride, max_length)
            else:
                ppl = self.calculate_perplexity(text, max_length)
            perplexities.append(ppl)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(texts)} texts...")
        
        return perplexities

#### --------------------------------------------- computes perplexity --------------------------------------------

class TextIntrinsicDimensionCalculator:
    """
    Calculate intrinsic dimensionality of text token embeddings using PH-dim.
    Code based on "Intrinsic Dimension Estimation for Robust Detection of AI-Generated Texts"
    https://arxiv.org/abs/2306.04723

    
    """
    
    def __init__(self, model_name="bert-base-uncased", device=None, 
                 alpha=1.0, n_reruns=3, n_points=7, n_points_min=3,
                 min_subsample=40, intermediate_points=7, layer_idx: Optional[int] = None):
        """
        Initialize text intrinsic dimension calculator.
        
        Args:
            model_name: HuggingFace model for embeddings (e.g., bert-base-uncased, roberta-base)
            device: cuda or cpu
            alpha: PH-dim parameter (should be < ground-truth ID, 1.0 works well)
            n_reruns: Number of calculation restarts
            n_points: Number of subsamples for smaller clouds
            n_points_min: Number of subsamples for larger clouds
            min_subsample: Minimum number of tokens needed for PHD
            intermediate_points: Number of intermediate subsample sizes to test
            layer_idx: Optional layer index to draw token embeddings from
                       (default: None → final hidden layer)
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading {model_name} on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # PHD parameters
        self.alpha = alpha
        self.n_reruns = n_reruns
        self.n_points = n_points
        self.n_points_min = n_points_min
        self.min_subsample = min_subsample
        self.intermediate_points = intermediate_points
        self.metric = 'euclidean'
        self.layer_idx = layer_idx
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text (following the original implementation)."""
        return text.replace('\n', ' ').replace('  ', ' ')
    
    def _get_token_embeddings(self, text: str, max_length: int = 512, 
                             layer_idx: Optional[int] = None) -> np.ndarray:
        """Handle different model architectures"""
        # Sanitize input
        if not text or not text.strip():
            text = " "
            
        processed_text = self.preprocess_text(text)
        inputs = self.tokenizer(
            processed_text,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )

            target_layer_idx = self.layer_idx if layer_idx is None else layer_idx
            if target_layer_idx is None:
                token_embeddings = outputs.last_hidden_state[0].cpu().numpy()
            else:
                hidden_states = outputs.hidden_states
                layer_count = len(hidden_states)
                target_idx = target_layer_idx if target_layer_idx >= 0 else layer_count + target_layer_idx
                token_embeddings = hidden_states[target_idx][0].cpu().numpy()
        
        # Smart token filtering based on model type
        if 'bert' in self.tokenizer.__class__.__name__.lower():
            # BERT-style: remove CLS and SEP
            token_embeddings = token_embeddings[1:-1]
        elif 'gpt' in self.tokenizer.__class__.__name__.lower():
            # GPT-style: keep all tokens
            pass
        else:
            # Default: remove first token only (often CLS)
            token_embeddings = token_embeddings[1:]
        
        return token_embeddings
    
    def _prim_tree(self, adj_matrix: np.ndarray) -> float:
        """
        Compute Prim's MST with alpha parameter.
        
        Args:
            adj_matrix: Distance matrix
            
        Returns:
            Total alpha-weighted tree length
        """
        infty = np.max(adj_matrix) + 10
        
        dst = np.ones(adj_matrix.shape[0]) * infty
        visited = np.zeros(adj_matrix.shape[0], dtype=bool)
        ancestor = -np.ones(adj_matrix.shape[0], dtype=int)

        v, s = 0, 0.0
        for i in range(adj_matrix.shape[0] - 1):
            visited[v] = 1
            ancestor[dst > adj_matrix[v]] = v
            dst = np.minimum(dst, adj_matrix[v])
            dst[visited] = infty
            
            v = np.argmin(dst)
            s += (adj_matrix[v][ancestor[v]] ** self.alpha)
            
        return s.item()
    
    def _sample_embeddings(self, embeddings: np.ndarray, n_samples: int) -> np.ndarray:
        """Sample subset of embeddings."""
        n = embeddings.shape[0]
        random_indices = np.random.choice(n, size=n_samples, replace=False)
        return embeddings[random_indices]
    
    def _calc_ph_dim_single(self, embeddings: np.ndarray, test_n: range, 
                           outp: np.ndarray, thread_id: int):
        """
        Calculate PH-dim for a single thread.
        
        Args:
            embeddings: Point cloud of embeddings
            test_n: Range of subsample sizes
            outp: Output array for results
            thread_id: Thread identifier
        """
        lengths = []
        for n in test_n:
            if embeddings.shape[0] <= 2 * n:
                restarts = self.n_points_min
            else:
                restarts = self.n_points
               
            reruns = np.ones(restarts)
            for i in range(restarts):
                sample = self._sample_embeddings(embeddings, n)
                dist_matrix = cdist(sample, sample, metric=self.metric)
                reruns[i] = self._prim_tree(dist_matrix)

            lengths.append(np.median(reruns))
        
        lengths = np.array(lengths)
        x = np.log(np.array(list(test_n)))
        y = np.log(lengths)
        N = len(x)   
        outp[thread_id] = (N * (x * y).sum() - x.sum() * y.sum()) / (N * (x ** 2).sum() - x.sum() ** 2)
    
    def calculate_intrinsic_dimension(
        self, 
    text: str, 
    max_length: int = 512,
        min_points: Optional[int] = None,
        max_points: Optional[int] = None,
        point_jump: Optional[int] = None,
    token_embeddings = None,
    layer_idx: Optional[int] = None,


    ) -> float:
        """
        Calculate intrinsic dimensionality for a single text based on its token embeddings.
        
        Args:
            text: Text string
            max_length: Maximum token length for embeddings
            min_points: Minimum subsample size (default: self.min_subsample)
            max_points: Maximum subsample size (default: auto-calculated)
            point_jump: Step between subsamples (default: auto-calculated)
            
        Returns:
            intrinsic_dimension: Estimated intrinsic dimensionality, or np.nan if text too short
        """
        # Get token embeddings (excludes CLS and SEP)
        if token_embeddings is None:
            token_embeddings = self._get_token_embeddings(text, max_length, layer_idx)
        
        n_tokens = token_embeddings.shape[0]
        
        print(f"Text has {n_tokens} tokens (excluding special tokens)")
        
        # Check if we have enough tokens
        if n_tokens < self.min_subsample:
            print(f"Warning: Text has only {n_tokens} tokens, need at least {self.min_subsample}")
            return np.nan
        
        # Auto-calculate parameters following original implementation
        if min_points is None:
            min_points = self.min_subsample
        
        if max_points is None or point_jump is None:
            mx_points = n_tokens
            mn_points = min_points
            step = (mx_points - mn_points) // self.intermediate_points
            
            if max_points is None:
                max_points = mx_points - step
            if point_jump is None:
                point_jump = max(step, 1)
        
        print(f"PHD parameters: min={min_points}, max={max_points}, step={point_jump}")
        
        # Calculate PH-dim
        ms = np.zeros(self.n_reruns)
        test_n = range(min_points, max_points, point_jump)
        
        if len(list(test_n)) < 2:
            print("Warning: Not enough subsample sizes to estimate dimension")
            return np.nan
        
        threads = []
        for i in range(self.n_reruns):
            thread = Thread(target=self._calc_ph_dim_single, 
                          args=[token_embeddings, test_n, ms, i])
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        m = np.mean(ms)
        intrinsic_dim = 1 / (1 - m)
        
        print(f"Intrinsic dimension: {intrinsic_dim:.2f}")
        return intrinsic_dim
    
    def calculate_batch(
        self,
        texts: List[str],
        max_length: int = 512,
        **phd_kwargs
    ) -> List[float]:
        """
        Calculate intrinsic dimensionality for multiple texts.
        
        Args:
            texts: List of text strings
            max_length: Maximum token length for embeddings
            **phd_kwargs: Additional arguments for PH-dim calculation
            
        Returns:
            dimensions: List of intrinsic dimensionality values (np.nan for texts that are too short)
        """
        dimensions = []
        
        for i, text in enumerate(texts):
            print(f"\nProcessing text {i+1}/{len(texts)}...")
            try:
                dim = self.calculate_intrinsic_dimension(text, max_length, **phd_kwargs)
                dimensions.append(dim)
            except Exception as e:
                print(f"Error processing text {i+1}: {e}")
                dimensions.append(np.nan)
        
        return dimensions
    

if __name__ == "__main__":
    # Initialize
    #"Qwen/Qwen2.5-0.5B"
    #"meta-llama/Llama-3.1-8B"
    calc = TextIntrinsicDimensionCalculator(model_name="FacebookAI/roberta-base",device='cuda:2')
    
    # Sample texts
    sample_text = "Speaking of festivities, there is one day in China that stands unrivaled - the first day of the Lunar New Year, commonly referred to as the Spring Festival. Even if you're generally uninterested in celebratory events, it's hard to resist the allure of the family reunion dinner, a quintessential aspect of the Spring Festival. Throughout the meal, family members raise their glasses to toast one another, expressing wishes for happiness, peace, health, and prosperity in the upcoming year."
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "asdf jkl qwerty zxcv random nonsense words here",
        "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
    ]
    print("PHD estimation of the Intrinsic dimension of sample text is ", calc.calculate_intrinsic_dimension(sample_text))
    dim = calc.calculate_batch(texts)

'''  
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

'''  