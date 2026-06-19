import time
import numpy as np

def benchmark_matrix_multiplication(size=1000):
    print(f"--- Matrix Multiplication Benchmark ({size}x{size}) ---")
    
    print("Initializing matrices...")
    A_np = np.random.rand(size, size)
    B_np = np.random.rand(size, size)
    
    A_list = A_np.tolist()
    B_list = B_np.tolist()
    
    C_list = [[0.0 for _ in range(size)] for _ in range(size)]
    
    # --- Task 1: Python Native Triple For-Loop ---
    print("\n1. Running native Python nested loops...")
    start_time_python = time.time()
    
    # TODO: Implement matrix multiplication using three nested for-loops
    # C_list[i][j] = sum(A_list[i][k] * B_list[k][j] for all k)
    
    ### TODO: Write your code below (Be careful with indentation)
    
    for i in range(size):
        for j in range(size):
            for k in range(size):
                C_list[i][j] += A_list[i][k] * B_list[k][j]
    
    ### END TODO
    
    end_time_python = time.time()
    python_duration = end_time_python - start_time_python
    print(f"Native Python time : {python_duration:.4f} seconds")
    
    # --- Task 2: NumPy Vectorized Calculation ---
    print("\n2. Running NumPy vectorized multiplication...")
    start_time_numpy = time.time()
    
    # TODO: Use NumPy to calculate the dot product of A_np and B_np
    # Reference: np.dot() or the @ operator
    
    ### TODO: Write your code below (1 line)

    C_np = np.dot(A_np, B_np)
    
    ### END TODO
    
    end_time_numpy = time.time()
    numpy_duration = end_time_numpy - start_time_numpy
    print(f"NumPy vectorized time : {numpy_duration:.4f} seconds")
    
    if numpy_duration > 0:
        speedup = python_duration / numpy_duration
        print(f"\n🚀 NumPy is {speedup:.2f}x faster!")

if __name__ == "__main__":
    benchmark_matrix_multiplication(1000)
