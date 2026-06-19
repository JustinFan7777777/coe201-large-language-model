import torch
import triton
import triton.language as tl

@triton.jit
def rmsnorm_kernel(
    x_ptr, y_ptr, w_ptr,
    stride_x_row,
    n_cols, eps,
    BLOCK_SIZE: tl.constexpr
):
    """
    Problem 3: Fused Triton RMSNorm Kernel [35 points]
    
    Args:
        x_ptr: Pointer to input tensor
        y_ptr: Pointer to output tensor
        w_ptr: Pointer to weight tensor
        stride_x_row: How much to advance the pointer to get to the next row
        n_cols: Number of columns (hidden dimension size)
        eps: Epsilon for numerical stability
        BLOCK_SIZE: Block size, should be next power of 2 greater than n_cols
    """
    row_idx = tl.program_id(0)
    
    ### TODO:
    # 1. Locate the pointers for the current row:
    #    row_start_ptr = x_ptr + row_idx * stride_x_row
    #    y_row_start_ptr = y_ptr + row_idx * stride_x_row
    #
    # 2. Create an array of column offsets up to BLOCK_SIZE.
    #
    # 3. Create a mask to ensure we don't read out of bounds (offsets < n_cols).
    #
    # 4. Load the row data `x` and the weights `w` (use mask, set `other=0.0`).
    #
    # 5. Compute the RMS:
    #    - square the row elements
    #    - sum them up using `tl.sum`
    #    - divide by n_cols, add eps, and take tl.sqrt
    #
    # 6. Normalize `x`, scale by `w`, and store into `y_row_start_ptr`.
    
    # --- Your code starts here ---
    
    row_start_ptr = x_ptr + row_idx * stride_x_row
    y_row_start_ptr = y_ptr + row_idx * stride_x_row
    # Calculate the starting pointers for the current row in the input and output tensors
    # row_start_ptr and y_row_start_ptr will point to the beginning of the current row

    offsets = tl.arange(0, BLOCK_SIZE)
    # Create an array of column offsets from 0 to BLOCK_SIZE-1
    # This will be used to index into the columns of the current row

    mask = offsets < n_cols
    # Create a mask to ensure we only access valid columns (those less than n_cols)
    # This prevents out-of-bounds memory access when BLOCK_SIZE > n_cols

    x = tl.load(row_start_ptr + offsets, mask=mask, other=0.0)
    w = tl.load(w_ptr + offsets, mask=mask, other=0.0)
    # Load the input row data and the weights using the offsets and mask
    # For positions where mask is False, the loaded value will be set to 0.0

    sum_sq = tl.sum(x * x, axis=0)
    # Compute the sum of squares of the elements in the row
    # This is the numerator for the RMS calculation

    rms = tl.sqrt(sum_sq / n_cols + eps)
    # Compute the RMS by dividing the sum of squares by n_cols
    # adding eps for stability, and taking the square root

    y = (x / rms) * w
    # Normalize the input row by dividing by the RMS
    # then scale by the weights, which is learnable

    tl.store(y_row_start_ptr + offsets, y, mask=mask)
    # Store the computed values back to the output tensor at the correct row
    # Use the same offsets and mask to ensure we only write to valid columns

    # --- Your code ends here ---


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    """
    PyTorch wrapper for Triton RMSNorm kernel.
    """
    M, N = x.shape
    y = torch.empty_like(x)
    
    # Needs to be a power of two for Triton block operations
    BLOCK_SIZE = triton.next_power_of_2(N)
    
    # Launch 1D grid, one program per row
    grid = lambda meta: (M,)
    
    rmsnorm_kernel[grid](
        x, y, weight,
        x.stride(0),
        N, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return y


def torch_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    return (x / rms) * weight

# --- Benchmark Block ---
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M"],  # x-axis is M (number of rows)
        x_vals=[128 * i for i in range(2, 33, 4)],  # Varying number of rows
        x_log=False,
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "Torch"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="GB/s",
        plot_name="rmsnorm-performance",
        args={"N": 4096},  # Fixed hidden dimension
    )
)
def benchmark(M, N, provider):
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    weight = torch.ones(N, device="cuda", dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_rmsnorm(x, weight), quantiles=quantiles)
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: rmsnorm(x, weight), quantiles=quantiles)
    
    # Memory ops: read x (M*N), read weight (N), write output y (M*N)
    gbps = lambda ms: (2 * M * N + N) * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    print("=" * 60)
    print("Task 3: Triton RMSNorm")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available. Triton kernel execution will likely fail.")
        print("You can verify logic, but local execution requires a GPU.")
    else:
        torch.manual_seed(42)
        M, N = 4096, 4096
        x = torch.randn(M, N, device="cuda")
        weight = torch.ones(N, device="cuda")
        
        y_torch = torch_rmsnorm(x, weight)
        y_triton = rmsnorm(x, weight)
        
        max_diff = torch.max(torch.abs(y_torch - y_triton)).item()
        print(f"Max difference between Torch and Triton: {max_diff:.6f}")
        
        if torch.allclose(y_torch, y_triton, atol=1e-5):
            print("✅ Correctness check PASSED!")
            print("\nRunning benchmark over sizes (this takes a moment)...")
            benchmark.run(print_data=True, show_plots=False)
        else:
            print("❌ Correctness check FAILED!")
