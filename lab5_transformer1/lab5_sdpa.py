# lab5_sdpa.py
# Lab 5 Task 1: Single-Head Scaled Dot-Product Attention

import torch
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Manually implement single-head scaled dot-product attention.

    Args:
        Q: Query tensor, shape (batch_size, seq_len, dim)
        K: Key tensor, shape (batch_size, seq_len, dim)
        V: Value tensor, shape (batch_size, seq_len, dim)
        mask: Optional mask tensor, shape (batch_size, seq_len, seq_len) or (seq_len, seq_len)
              0 means masked out, 1 means keep

    Returns:
        output: Attention output, shape (batch_size, seq_len, dim)
        attn_weights: Attention weights, shape (batch_size, seq_len, seq_len)
    """
    d_k = Q.size(-1)

    ### TODO: 1. Compute dot product attention scores (Q @ K^T)

    # exchange the last two dimensions of K for the matrix multiplication
    scores = torch.matmul(Q, K.transpose(-2, -1))
    # Q shape (batch_size, seq_len, dim)
    # K shape (batch_size, seq_len, dim) -> K^T shape (batch_size, dim, seq_len)
    # scores shape (batch_size, seq_len, seq_len) after matmul

    ### TODO: 2. Scale the scores by dividing by sqrt(d_k)

    # prevent scores from growing too large
    scores = scores / math.sqrt(d_k)
    # softmax will cause vanishing & exploding gradients
    # which is not good for training
    # so we scale down the scores to keep them in a reasonable range
    
    ### TODO: 3. Apply the mask (if provided) by setting masked positions to -inf
    # Hint: use scores.masked_fill(mask == 0, float('-inf')) if mask is not None

    if mask is not None:
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)  # Add batch dimension if mask is (seq_len, seq_len)
        
        mask = mask.to(scores.device)  # Ensure mask is on the same device as scores

        # Set masked positions to -inf so that after softmax they become zero
        scores = scores.masked_fill(mask == 0, float('-inf'))
        # masked_fill: fill elements of scores with -inf where mask is 0
        # note that this step is done before softmax
    
    ### TODO: 4. Apply softmax to get the attention weights
    
    # softmax along the last dimension to get weights that sum to 1
    attn_weights = torch.softmax(scores, dim=-1)

    ### TODO: 5. Multiply weights with V to get the final output
    
    # --- Your code starts here ---
    
    # attention output is the weighted sum of values
    output = torch.matmul(attn_weights, V)
    # matmul: matrix multiplication
    # attn_weights shape (batch_size, seq_len, seq_len)
    # V shape (batch_size, seq_len, dim)

    return output, attn_weights

    # --- Your code ends here ---
    # return output, attn_weights

def create_causal_mask(seq_len):
    """
    Create a causal mask for autoregressive generation.
    The mask should be a lower triangular matrix where elements (i, j) with j <= i are 1, and 0 otherwise.
    """
    ### TODO: Create the causal mask (lower triangular matrix)
    # Hint: use torch.tril and torch.ones
    
    # --- Your code starts here ---
    
    # Create a lower triangular matrix of ones with shape (seq_len, seq_len)
    mask = torch.tril(torch.ones(seq_len, seq_len))
    # tril: triangle lower, ones: create a matrix of ones

    # --- Your code ends here ---
    # return mask

def main():
    print("Testing Task 1: Single-Head Scaled Dot-Product Attention")
    batch_size, seq_len, dim = 2, 4, 8
    Q = torch.randn(batch_size, seq_len, dim)
    K = torch.randn(batch_size, seq_len, dim)
    V = torch.randn(batch_size, seq_len, dim)

    # 1. Basic test
    output, attn_weights = scaled_dot_product_attention(Q, K, V)
    print(f"Basic output shape: {output.shape}")
    
    # 2. Mask test
    mask = create_causal_mask(seq_len)
    output_masked, attn_weights_masked = scaled_dot_product_attention(Q, K, V, mask)
    print(f"Masked weights (first row should only have first element non-zero):\n{attn_weights_masked[0, 0]}")

if __name__ == "__main__":
    main()
