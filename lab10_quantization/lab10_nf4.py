import torch

def get_nf4_bins():
    """
    Derives the 16 optimal bin values for NF4 based on a standard normal distribution N(0, 1).
    NF4 bins are equal-probability quantiles.
    Returns:
        A torch.tensor of 16 float values.
    """
    ### TODO: 1. Find the 16 values that divide N(0, 1) into equal-probability regions.
    # Hint: Use the Percent Point Function (Inverse CDF) of the Normal distribution.
    # You may use scipy.stats.norm.ppf (via scipy) or other methods.
    # Formula: q_i = Phi^-1((2i + 1) / 32) for i = 0 to 15.
    # Then normalize the resulting bins such that the max absolute value is 1.0.
    
    from scipy.stats import norm

    # 16 quantiles for NF4
    i = torch.arange(16) # 0 to 15
    probabilities = (2 * i + 1) / 32 # 1/32, 3/32, ..., 31/32

    # Get the corresponding bin values using the inverse CDF (ppf)
    bins = torch.tensor(norm.ppf(probabilities.numpy()), dtype=torch.float32)

    # Normalize the bins such that the max absolute value is 1.0
    bins = bins / torch.max(torch.abs(bins))

    return bins # torch.tensor of shape (16,)

def quantize_nf4(tensor):
    """
    Quantizes a standard normally distributed tensor into NF4.
    Returns the quantized indices and the dequantized values.
    """
    nf4_bins = get_nf4_bins().to(tensor.device)
    
    if torch.all(nf4_bins == 0):
        return None, None

    ### TODO: 2. Normalize the input tensor to the range [-1.0, 1.0]

    max_abs = tensor.abs().max()
    normalized_tensor = tensor / max_abs # normalize to [-1, 1]
    
    ### TODO: 3. Efficiently map each element in the normalized tensor to the closest bin in `nf4_bins`
    # Hint: Use broadcasting with torch.abs() and torch.argmin(), or torch.bucketize()

    distances = (normalized_tensor.unsqueeze(-1) - nf4_bins).abs()
    indices = distances.argmin(dim=-1)
    
    ### TODO: 4. Reconstruct the quantized values using the closest bins, and scale them back to original range

    dequantized = nf4_bins[indices] * max_abs
    
    return indices, dequantized


def uniform_int4_quantize(tensor):
    """
    Uniform INT4 Quantization for comparison.
    """
    qmin = -8
    qmax = 7
    max_abs = tensor.abs().max()
    scale = max_abs / 8
    
    q_tensor = torch.round(tensor / scale)
    q_tensor = torch.clamp(q_tensor, qmin, qmax)
    
    return q_tensor, q_tensor * scale

if __name__ == "__main__":
    print("=" * 60)
    print("Lab 10 Task 4: NF4 vs INT4 Quantization [Bonus]")
    print("=" * 60)

    # Test on a simulated normally distributed weight matrix
    torch.manual_seed(42)
    weights = torch.randn((1024, 1024))
    print(f"Generated normal weight matrix of shape {weights.shape}")
    
    try:
        print("\n--- Running Uniform INT4 Quantization ---")
        _, dequantized_int4 = uniform_int4_quantize(weights)
        mse_int4 = torch.nn.functional.mse_loss(weights, dequantized_int4).item()
        print(f"Standard INT4 MSE: {mse_int4:.6f}")
    except Exception as e:
        print(f"INT4 Error: {e}")
        mse_int4 = None
        
    try:
        print("\n--- Running Custom NF4 Quantization ---")
        indices, dequantized_nf4 = quantize_nf4(weights)
        
        if dequantized_nf4 is None:
            print("quantize_nf4 Not Implemented Yet!")
        else:
            mse_nf4 = torch.nn.functional.mse_loss(weights, dequantized_nf4).item()
            print(f"NF4 Quantization MSE: {mse_nf4:.6f}")
            
            if mse_int4 is not None:
                improvement = (mse_int4 - mse_nf4) / mse_int4 * 100
                print(f"Improvement: {improvement:.2f}%")
                
                if improvement > 15.0:
                    print("\nSuccess! Your NF4 mapping mathematically outperforms standard INT4 on normal distributions!")
    except Exception as e:
        print(f"NF4 Error: {e}")
