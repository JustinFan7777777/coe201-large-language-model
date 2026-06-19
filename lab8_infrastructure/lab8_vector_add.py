import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel for vector addition.
    """
    # --- Your code starts here ---
    # 1. Get the program ID for the 0th axis
    # 2. Compute the start offset for this block
    # 3. Create an array of offsets within the block
    # 4. Create a mask to guard against out-of-bounds memory accesses
    # 5. Load data from x_ptr and y_ptr
    # 6. Compute x + y
    # 7. Store the result into output_ptr
    
    # ### TODO: Implement the kernel logic
    
    pid = tl.program_id(axis=0)
    # 1. Get the program ID for the 0th axis

    block_start = pid * BLOCK_SIZE 
    # 2. Compute the start offset for this block

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # 3. Create an array of offsets within the block

    mask = offsets < n_elements
    # 4. Create a mask to guard against out-of-bounds memory accesses

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    # 5. Load data from x_ptr and y_ptr

    output = x + y
    # 6. Compute x + y

    tl.store(output_ptr + offsets, output, mask=mask)
    # 7. Store the result into output_ptr

    # --- Your code ends here ---

def add(x: torch.Tensor, y: torch.Tensor):
    """
    PyTorch wrapper for the triton add_kernel.
    """
    # 1. Pre-allocate the output tensor
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    
    n_elements = output.numel()
    
    # --- Your code starts here ---
    # 2. Define the grid and launch the kernel
    # ### TODO: Choose a BLOCK_SIZE, compute the grid, and launch add_kernel
    
    BLOCK_SIZE = 1024
    # BLOCK_SIZE is a the number of elements each block will process.
    # A common choice is 1024

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    # Compute the grid size. We need enough blocks to cover all elements.
    # triton.cdiv(a, b) computes ceil(a / b), which gives us the number of blocks needed.

    add_kernel[grid](
        x, y, output, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    # Launch the kernel with the computed grid and block size.
    # The arguments to the kernel are the input tensors, output tensor, number of elements,
    # and the block size as a compile-time constant.

    # --- Your code ends here ---
    return output

# --- Benchmark Block ---
# We use Triton's built-in testing utility to measure relative performance
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(12, 28, 1)],  # Different vector sizes
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['triton', 'torch'],  # Possible values for `line_arg`.
        line_names=['Triton', 'Torch'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-')],  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name='vector-add-performance',  # Name for the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    )
)
def benchmark(size, provider):
    x = torch.rand(size, device='cuda', dtype=torch.float32)
    y = torch.rand(size, device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)
    # 3 memory operations (read x, read y, write output), each element is 4 bytes
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)

def main():
    # 1. Simple correctness test
    torch.manual_seed(0)
    size = 98432
    # NOTE: Triton currently prioritizes CUDA targets. Executing cleanly needs a GPU.
    if not torch.cuda.is_available():
        print("CUDA is not available. Executing Triton kernels usually requires a GPU.")
        print("You can inspect the code structure, but execution might raise an exception here.")
        return

    x = torch.rand(size, device='cuda')
    y = torch.rand(size, device='cuda')
    
    try:
        output_torch = x + y
        output_triton = add(x, y)
        print(f"Correctness check result: {torch.allclose(output_torch, output_triton)}")
        
        # 2. Run benchmark if correctness passes
        print("\nRunning benchmark over sizes (this takes a moment)...")
        benchmark.run(print_data=True, show_plots=False)
        
    except Exception as e:
        print(f"Execution failed (implement the kernel first): {e}")

if __name__ == "__main__":
    main()
