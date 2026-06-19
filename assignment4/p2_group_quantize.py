import torch

def group_quantize(tensor: torch.Tensor, group_size: int = 64):
    """
    Problem 2: Group-wise Quantization [30 points]
    
    Args:
        tensor: Weight matrix of shape (out_features, in_features)
        group_size: Number of elements per group (must divide in_features)
        
    Returns:
        q_tensor: Quantized tensor of shape (out_features, in_features // G, G), dtype=uint8
        scales: Scales of shape (out_features, in_features // G, 1)
        zero_points: Zero points of shape (out_features, in_features // G, 1)
    """
    out_features, in_features = tensor.shape
    assert in_features % group_size == 0, "in_features must be divisible by group_size"
    
    qmin = 0
    qmax = 255
    
    ### TODO:
    # Perform asymmetric affine quantization independently for each group of size `group_size`.
    # You must derive the `scale` and `zero_point` along the correct dimension.
    # Ensure numerical stability (e.g. handle division by zero) and properly clamp the values before casting to uint8.
    # --- Your code starts here ---
    
    reshaped = tensor.view(out_features, in_features // group_size, group_size)
    # Reshape the input tensor to group the in_features into groups of size group_size
    # tensor: (out_features, in_features)
    # The new shape is (out_features, num_groups, group_size) where num_groups = in_features // group_size

    min_vals, _ = reshaped.min(dim=2, keepdim=True)
    max_vals, _ = reshaped.max(dim=2, keepdim=True)
    # Compute the minimum and maximum values for each group along the last dimension (group_size)
    # dim=2 specifies that we are reducing along the group_size dimension
    # and keepdim=True keeps the reduced dimension for broadcasting
    # , _ is used to ignore the indices of the min and max values since we only need the values themselves
    # min_vals and max_vals will have shape (out_features, num_groups, 1) which allows for broadcasting in the next steps

    eps = 1e-8
    scales = (max_vals - min_vals) / float(qmax - qmin)
    scales = torch.clamp(scales, min=eps)
    # Calculate the scale for each group using the range of values and the quantization range (qmax - qmin)
    # Clamp the scales to a minimum value to avoid division by zero
    # scales will have shape (out_features, num_groups, 1)

    zero_points = qmin - min_vals / scales
    zero_points = torch.round(zero_points)
    zero_points = torch.clamp(zero_points, qmin, qmax)
    # Calculate the zero point for each group using the minimum value and the scale
    # Round the zero points to the nearest integer since they must be integers in quantization
    # zero_points will have shape (out_features, num_groups, 1)

    q_tensor = torch.round(reshaped / scales + zero_points)
    q_tensor = torch.clamp(q_tensor, qmin, qmax).to(torch.uint8)
    # Quantize the tensor by scaling and adding the zero point, then round to the nearest integer
    # Clamp the quantized values to the valid range [qmin, qmax] and convert to uint8

    return q_tensor, scales, zero_points

    # --- Your code ends here ---


def group_dequantize(q_tensor: torch.Tensor, scales: torch.Tensor, zero_points: torch.Tensor) -> torch.Tensor:
    """
    Dequantizes the group-quantized tensor back to float32 and restores original shape.
    """
    ### TODO:
    # Reconstruct the floating-point tensor using the scales and zero points.
    # Ensure the returned tensor shape matches the original unquantized weight matrix.
    
    # --- Your code starts here ---
    
    dequantized = scales * (q_tensor.float() - zero_points)
    # Dequantize the tensor by reversing the quantization formula

    out_features, num_groups, group_size = q_tensor.shape
    return dequantized.view(out_features, num_groups * group_size)
    # Reshape the dequantized tensor back to the original shape (out_features, in_features)

    # --- Your code ends here ---

if __name__ == "__main__":
    print("=" * 60)
    print("Task 2: Group-wise Quantization")
    print("=" * 60)

    torch.manual_seed(42)
    # 128 out_features, 256 in_features
    weights = torch.randn((128, 256))
    
    # Introduce an extreme outlier in one group
    weights[0, 5] = 100.0 

    try:
        q_tensor, scales, zps = group_quantize(weights, group_size=64)
        dq_tensor = group_dequantize(q_tensor, scales, zps)
        
        mse = torch.nn.functional.mse_loss(weights, dq_tensor).item()
        print(f"Group-wise MSE (G=64): {mse:.6f}")
        
        # Compare with per-tensor to see the benefit
        w_min, w_max = weights.min(), weights.max()
        scale_pt = (w_max - w_min) / 255
        zp_pt = torch.clamp(0 - torch.round(w_min / scale_pt), 0, 255)
        q_pt = torch.clamp(torch.round(weights / scale_pt + zp_pt), 0, 255)
        dq_pt = scale_pt * (q_pt - zp_pt)
        mse_pt = torch.nn.functional.mse_loss(weights, dq_pt).item()
        
        print(f"Per-tensor MSE: {mse_pt:.6f}")
        print(f"Group quantization improved MSE by a factor of {mse_pt / mse:.2f}x")
    except NotImplementedError:
        print("[Skip] Not implemented.")
