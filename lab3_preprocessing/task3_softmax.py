# task3_softmax.py
import torch
import torch.nn.functional as F

# ==========================================
# Task 3: Output Layers (Logits, Softmax, Sampling)
# ==========================================

def manual_softmax(logits, temperature=1.0):
    """
    TODO: Implement Softmax manually with numerical stability.
    Formula: exp(x_i / T) / sum(exp(x_j / T))
    
    Stability Trick: Subtract the maximum value before exponentiating.
    x = (logits / temperature)
    x = x - max(x)
    probs = exp(x) / sum(exp(x))
    """
    # Step 1: Scale logits by temperature

    # First we scale the logits by the temperature
    # if T < 1, then the logits are amplified, making the distribution sharper
    # if T > 1, then the logits are shrunk, making the distribution flatter
    # logits size: (1, 7) in this case
    # x will also have size (1, 7) after scaling
    x = logits / temperature

    # Step 2: Subtract max for stability

    # dim=-1 means we are taking the max across the last dimension (the vocab dimension)
    # keepdim=True means we keep the same number of dimensions (1, 7)
    # this will give us a tensor of shape (1, 1) to subtract from each element in the logits
    # We need this step tp prevent overflow when we exponentiate the logits, especially when they are fairly large
    x = x - torch.max(x, dim=-1, keepdim=True).values

    # Step 3: Exponentiate and normalize

    # Now we are able to safely exponentiate the x values (which are scaled & stabilized)
    # x can be negative, but after exponentiation, we will get positive values
    exp_x = torch.exp(x)

    # Eventually we normalize by summing up
    probs = exp_x / torch.sum(exp_x, dim=-1, keepdim=True)

    return probs

def top_k_filtering(probs, k=3):
    """
    TODO: Keep only the top k probabilities and zero-out the rest. 
    Then re-normalize so they sum to 1.
    """
    # Hint: use torch.topk()

    # dim=-1 equals to the last dimension, which is the vocab dimension in this case
    vocab_size = probs.size(-1)
    k = min(k, vocab_size) # To ensure that k does not exceed the vocab size

    # torch.topk returns the top k values and their corresponding indices 
    # along the specified dimension (-1 in this case)
    topk_values, topk_idxs = torch.topk(probs, k=k, dim=-1)

    # We create a new tensor of zeros with the same shape as probs
    filtered_probs = torch.zeros_like(probs)

    # we scatter the top k values back into their original positions
    # and keep the rest as zero
    filtered_probs.scatter_(dim=-1, index=topk_idxs, src=topk_values)

    # Finally, we need to re-normalize the filtered probabilities so they sum to 1
    denominator = torch.sum(filtered_probs, dim=-1, keepdim=True)
    filtered_probs = filtered_probs / denominator

    return filtered_probs

def task3_softmax():
    print("--- Task 3: Logits, Softmax & Decoding ---")
    
    # 1. Input Logits (e.g., from a Language Model)
    logits = torch.tensor([[-2.0, 1.0, 5.0, -1.0, 2.0, 4.5, 0.5]])
    
    # 2. Test Manual Softmax
    probs = manual_softmax(logits, temperature=1.0)
    if probs is not None:
        print(f"Manual Softmax (T=1.0):\n{probs}")
        # Verify with PyTorch
        ref = F.softmax(logits, dim=-1)
        assert torch.allclose(probs, ref), "Softmax implementation mismatch!"
        print("Success: Manual Softmax matches PyTorch!")

    # 3. Effect of Temperature
    # TODO: Calculate and compare probs for T=0.1 (sharp) and T=5.0 (flat)
    # Observe which index becomes dominant as T -> 0.1

    probs_sharp = manual_softmax(logits, temperature=0.1)
    probs_flat = manual_softmax(logits, temperature=5.0)

    print("\n--- Temperature Effect ---")
    print(f"Softmax (T=0.1, sharp):\n{probs_sharp}")
    print(f"Softmax (T=5.0, flat):\n{probs_flat}")
    
    dominant_idx = torch.argmax(probs_sharp, dim=-1).item()
    print(f"Dominant index when T -> 0.1: {dominant_idx}")
    
    # 4. Top-K Filtering
    if probs is not None:
        print("\n--- Top-K Filtering (k=3) ---")
        filtered_probs = top_k_filtering(probs, k=3)
        if filtered_probs is not None:
            print(f"Top-K Probs:\n{filtered_probs}")
            print(f"Sum of Top-K Probs: {filtered_probs.sum().item():.2f}")

    # 5. Sampling
    # TODO: Implement a single-step sampling using torch.multinomial
    if probs is not None:  # 仅在已有有效概率分布时进行采样
        sampling_dist = filtered_probs if filtered_probs is not None else probs  # 若有top-k分布就从top-k采样，否则从原分布采样
        sampled_idx = torch.multinomial(sampling_dist, num_samples=1)  # 按概率随机采样1个索引（形状通常为[batch, 1]）
        sampled_token_id = sampled_idx.squeeze(-1).item()  # 去掉多余维度并转成Python整数，便于打印和后续使用
        sampled_prob = sampling_dist[0, sampled_token_id].item()  # 取出该采样索引对应的概率值，便于解释采样结果
        print("\n--- Sampling ---")
        print(f"Sampled token index: {sampled_token_id}")  # 输出被采样到的token索引
        print(f"Sampled token probability: {sampled_prob:.6f}")  # 输出该token在当前分布下的概率

if __name__ == "__main__":
    task3_softmax()
