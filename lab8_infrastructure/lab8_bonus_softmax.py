import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel for fused softmax.
    Each program/block processes one row of the input tensor.
    """
    # --- Your code starts here ---
    # 1. Get the program ID for the 0th axis (which row this block is processing)
    # 2. Compute the start pointer for the current row
    # 3. Create an array of offsets within the row
    # 4. Load the row into SRAM, using a mask for bounds checking. 
    #    HINT: Use -float('inf') for out-of-bounds elements so they become 0 after exp()
    #    Example: tl.load(ptr, mask=mask, other=-float('inf'))
    # 5. Compute the numerically stable Softmax:
    #    HINT: Use tl.max() and tl.sum() to compute the maximum and sum of the row.
    #    5.1 Subtract the maximum value in the row
    #    5.2 Compute the exponential
    #    5.3 Compute the sum of the exponentials
    #    5.4 Divide to get the final softmax probabilities
    # 6. Store the result back to DRAM
    
    # ### TODO: Implement the Softmax kernel logic
    
    row_idx = tl.program_id(axis=0)
    # 1. Get the program ID for the 0th axis (which row this block is processing)

    row_in_ptr = input_ptr + row_idx * input_row_stride
    row_out_ptr = output_ptr + row_idx * output_row_stride
    # 2. Compute the start pointer for the current row
    # strides are used to compute the starting address of the row in the input and output tensors.

    col_offsets = tl.arange(0, BLOCK_SIZE)
    # 3. Create an array of offsets within the row: 0, 1, 2, ..., BLOCK_SIZE-1

    mask = col_offsets < n_cols
    # 4. Load the row into SRAM, using a mask for bounds checking.

    row = tl.load(row_in_ptr + col_offsets, mask=mask, other=-float('inf'))
    # Load the input row into SRAM. Out-of-bounds elements are set to -inf
    # so that they become 0 after exponentiation.

    row_max = tl.max(row, axis=0)
    row_minus_max = row - row_max
    # 5.1 Subtract the maximum value in the row for numerical stability.
    # softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))

    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    # 5.2 Compute the exponential and 5.3 Compute the sum of the exponentials.
    
    softmax = numerator / denominator
    # 5.4 Divide to get the final softmax probabilities.

    tl.store(row_out_ptr + col_offsets, softmax, mask=mask)
    # 6. Store the result back to DRAM, using the same mask for bounds checking.

    # --- Your code ends here ---

def softmax(x: torch.Tensor):
    """
    PyTorch wrapper for the triton softmax_kernel.
    """
    assert x.is_cuda and x.dim() == 2, "Input must be a 2D CUDA tensor"
    n_rows, n_cols = x.shape
    
    # 1. Pre-allocate the output tensor
    output = torch.empty_like(x)
    
    # --- Your code starts here ---
    # 2. Compute BLOCK_SIZE: the smallest power of 2 greater than or equal to n_cols
    # 3. Define the grid and launch the kernel. Grid should be 1D, with one block per row.
    #    HINT: You may set num_warps dynamically based on BLOCK_SIZE for better occupancy.
    #          (e.g., 4 warps normally, 8 if BLOCK_SIZE >= 2048, 16 if BLOCK_SIZE >= 4096)
    
    # ### TODO: Compute BLOCK_SIZE, grid, and launch softmax_kernel
    
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    # 2. Compute BLOCK_SIZE: the smallest power of 2 greater than or equal to n_cols

    if BLOCK_SIZE >= 4096:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    else:
        num_warps = 4
    # Dynamically set num_warps based on BLOCK_SIZE for better occupancy.

    grid = (n_rows,)
    # 3. Define the grid: one block per row

    softmax_kernel[grid](
        output_ptr=output,
        input_ptr=x,
        input_row_stride=x.stride(0),
        output_row_stride=output.stride(0),
        n_cols = n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )
    # Launch the kernel with the appropriate arguments.

    # --- Your code ends here ---
    
    return output

def naive_softmax(x: torch.Tensor):
    """
    Compute softmax using a sequence of native PyTorch basic operations.
    This demonstrates the 'Memory Wall' because each operation reads/writes to global memory.
    """
    # keepdim=True is necessary for correct broadcasting
    x_max = x.max(dim=1, keepdim=True)[0]
    safe_x = x - x_max
    num = torch.exp(safe_x)
    denom = num.sum(dim=1, keepdim=True)
    return num / denom

# --- Benchmark Block ---
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # Argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(2, 100, 20)],  # Different row sums
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['torch', 'torch_naive', 'triton'],
        line_names=['Torch Native', 'Torch Naive', 'Triton'],
        styles=[('green', '-'), ('red', '--'), ('blue', '-')],
        ylabel='GB/s',  # Label name for the y-axis
        plot_name='softmax-performance',
        args={'M': 4096},  # Fixed number of rows
    )
)
def benchmark(M, N, provider):
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1), quantiles=quantiles)
    if provider == 'torch_naive':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: naive_softmax(x), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: softmax(x), quantiles=quantiles)
    
    # Softmax reads input once, and writes output once
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)

def main():
    # 1. Simple correctness test
    torch.manual_seed(0)
    # NOTE: Triton currently prioritizes CUDA targets. Executing cleanly needs a GPU.
    if not torch.cuda.is_available():
        print("CUDA is not available. Executing Triton kernels usually requires a GPU (CUDA/ROCm).")
        print("You can inspect the code structure, but execution might fail locally.")
        return

    x = torch.randn(1823, 781, device='cuda')
    
    try:
        output_torch = torch.softmax(x, axis=-1)
        output_triton = softmax(x)
        
        # Softmax can accumulate floating point errors, so we use a slightly larger atol
        is_close = torch.allclose(output_torch, output_triton, atol=1e-5, rtol=1e-5)
        print(f"Correctness check result: {is_close}")
        
        if is_close:
            # 2. Run benchmark if correctness passes
            print("\nRunning benchmark over sizes (this takes a moment)...")
            benchmark.run(print_data=True, show_plots=False)
        else:
            print("Output mismatch! Diff max:", torch.max(torch.abs(output_torch - output_triton)))
            
    except Exception as e:
        print(f"Execution failed (implement the kernel first): {e}")

if __name__ == "__main__":
    main()
