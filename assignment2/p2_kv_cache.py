import torch
import torch.nn as nn
import math

class KVCacheAttention(nn.Module):
    """
    Problem 2: KV Cache Self-Attention [30 points]
    
    In autoregressive generation, the model predicts the next token one by one.
    To avoid recomputing Keys and Values for all past tokens at every step,
    we can cache them. This is called KV-Caching.
    
    This module implements a Single-Head Self-Attention specifically designed
    for step-by-step decoding.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # KV Cache stored as buffers so they move with the model device,
        # but are not trained parameters.
        self.register_buffer("k_cache", None)
        self.register_buffer("v_cache", None)

    def reset_cache(self):
        """Reset the KV cache. Should be called before generating a new sequence."""
        self.k_cache = None
        self.v_cache = None

    def forward(self, x: torch.Tensor, use_cache: bool = True) -> torch.Tensor:
        """
        Forward pass for a single token using KV Cache.
        
        Args:
            x: Input tensor for the CURRENT timestep. Shape: (batch_size, seq_len, d_model)
            use_cache: If True, uses and updates the cache. If False, operates as standard attention
                       (but assumes x contains the sequence up to current step).
                       
        Returns:
            Output tensor. Shape: (batch_size, 1, d_model)
        """
        batch_size, seq_len, _ = x.shape
        if use_cache:
            assert seq_len == 1, "With KV cache, input should be a single token (seq_len=1)"
            
        # 1. Project x to get Q, K, V
        q = self.q_proj(x)  # (batch_size, seq_len, d_model)
        k = self.k_proj(x)  # (batch_size, seq_len, d_model)
        v = self.v_proj(x)  # (batch_size, seq_len, d_model)
        
        ### TODO: Implement the KV Cache logic and Scaled Dot-Product Attention
        # Step 1: Update the cache if `use_cache` is True.
        #   - If self.k_cache / self.v_cache is None, initialize them with the current `k` and `v`.
        #   - Otherwise, concatenate the current `k` and `v` to the existing cache along the sequence dimension (dim=1).
        #   - Update `self.k_cache` and `self.v_cache`.
        #   - Let `k_full` and `v_full` be the tensors used for attention (from cache or current batch).
        
        # Step 2: Compute Scaled Dot-Product Attention using `q`, `k_full`, and `v_full`.
        #   - scores = (q @ k_full^T) / sqrt(d_model)
        #   - Apply softmax and multiply by v_full.
        
        # --- Your code starts here ---
        
        # step 1: Update the cache if use_cache is True
        if use_cache:
            # If the cache is empty, initialize it with the current k and v
            if self.k_cache is None or self.v_cache is None:
                self.k_cache = k
                self.v_cache = v
            else:
                # Concatenate the current k & v to the existing cache along the sequence dimension (dim=1)
                # self.k_cache shape: (batch_size, t - 1, d_model)
                # k shape: (batch_size, 1, d_model)
                # after concatenation, self.k_cache shape: (batch_size, t, d_model)
                # same for v_cache, concatenate along dim=1
                self.k_cache = torch.cat([self.k_cache, k], dim=1)
                self.v_cache = torch.cat([self.v_cache, v], dim=1)

            # Use the full cache for attention
            k_full = self.k_cache
            v_full = self.v_cache

        else:
            # if not using cache, we assume x contains the full sequence up to current step
            # so we use k and v directly without caching
            k_full = k
            v_full = v

        # step 2: compute scaled dot-product attention
        # q shape: (batch_size, q_len, d_model), where q_len is 1 in autoregressive decoding
        # k_full shape: (batch_size, kv_len, d_model), where kv_len is the length of the cached keys
        # k_full.transpose(-2, -1) shape: (batch_size, d_model, kv_len)
        # scores shape: (batch_size, q_len, kv_len)
        scores = torch.matmul(q, k_full.transpose(-2, -1)) / math.sqrt(self.d_model)
        # formula: scores = (q @ k_full^T) / sqrt(d_model)
        # scores shape: (batch_size, 1, kv_len) since q_len is 1 in autoregressive decoding
        # we divide by sqrt(d_model) for scaling to prevent large dot product values
        # which can lead to small gradients after softmax.

        # apply softmax to get attention weights
        # softmax is applied along the last dimension (kv_len) to get a distribution over the keys
        attn_weights = torch.softmax(scores, dim=-1) # shape: (batch_size, q_len, kv_len)

        output = torch.matmul(attn_weights, v_full)
        # (batch_size, q_len, kv_len) @ (batch_size, kv_len, d_model) -> (batch_size, q_len, d_model)

        return output

        # --- Your code ends here ---

if __name__ == "__main__":
    import time
    print("--- Problem 2: KV Cache Efficiency Benchmark ---")
    
    d_model = 256
    seq_len = 500
    model = KVCacheAttention(d_model)
    
    # Pre-fill some weights for stability
    nn.init.normal_(model.q_proj.weight, std=0.02)
    nn.init.normal_(model.k_proj.weight, std=0.02)
    nn.init.normal_(model.v_proj.weight, std=0.02)

    # 1. Benchmark WITHOUT KV Cache
    # We simulate autoregressive generation by passing increasing prefix lengths
    start_time = time.time()
    for i in range(1, seq_len + 1):
        x_seq = torch.randn(1, i, d_model)
        _ = model(x_seq, use_cache=False)
    no_cache_time = time.time() - start_time
    print(f"Time WITHOUT KV Cache: {no_cache_time:.4f}s")

    # 2. Benchmark WITH KV Cache
    # We reset the cache and pass tokens one by one
    model.reset_cache()
    start_time = time.time()
    for i in range(seq_len):
        x_tok = torch.randn(1, 1, d_model)
        _ = model(x_tok, use_cache=True)
    with_cache_time = time.time() - start_time
    print(f"Time WITH KV Cache:    {with_cache_time:.4f}s")
    
    print(f"Speedup: {no_cache_time / with_cache_time:.2f}x")
