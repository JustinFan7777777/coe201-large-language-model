import numpy as np

def softmax(logits):
    """
    Calculate Softmax probability distribution using NumPy vectorization.
    Do NOT use any `for` or `while` loops!
    
    Math formula:
    S(x_i) = e^(x_i) / sum( e^(x_j) )  for all j

    For numerical stability, we subtract the maximum value before exponentiation:
    S(x_i) = e^(x_i - max(x)) / sum( e^(x_j - max(x)) )
    """
    print(f"\n--- Calculating Vectorized Softmax ---")
    
    # Convert input list to NumPy array (if it isn't already)
    x = np.array(logits)
    print(f"Input Logits: {x}")
    
    # Task 1: Find the maximum value in the array for numerical stability
    # Hint: np.max()
    
    ### TODO: Write your code below (1 line)

    max_val = np.max(x)

    ### END TODO
    
    # Task 2: Subtract the maximum value from the input to prevent overflow
    # Use array broadcasting: x - max_val
    
    ### TODO: Write your code below (1 line)

    stable_x = x - max_val 

    ### END TODO
    print(f"Shifted Logits: {stable_x}")
    
    # Task 3: Calculate the exponential values for all elements
    # Hint: np.exp()
    
    ### TODO: Write your code below (1 line)

    exp_x = np.exp(stable_x)

    ### END TODO
    print(f"Exponential values: {exp_x}")
    
    # Task 4: Calculate the sum of all exponential values
    # Hint: np.sum()
    
    ### TODO: Write your code below (1 line)

    sum_exp_x = np.sum(exp_x)

    ### END TODO
    
    # Task 5: Compute final probabilities (division)
    # Hint: Use array broadcasting to divide the entire array by a scalar
    
    ### TODO: Write your code below (1 line)

    probabilities = exp_x / sum_exp_x

    ### END TODO
    
    return probabilities

if __name__ == "__main__":
    # --- Test Softmax (Normal case) ---
    print("Test 1: Normal Logits")
    scores = [2.0, 1.0, 0.1]
    probs = softmax(scores)
    print(f"Softmax Result: {probs}")
    
    if np.abs(np.sum(probs) - 1.0) < 1e-6 and probs[0] > probs[1]:
        print("✅ Test 1 Passed!")
    else:
        print("❌ Test 1 Failed.")

    # --- Test Softmax (Numerical Stability case) ---
    print("\nTest 2: Extremely Large Logits (Testing Numerical Stability)")
    # Using very large numbers. Without subtracting max_val, np.exp() will cause an OverflowError.
    large_scores = [1000.0, 1000.0, 1000.0]
    
    try:
        large_probs = softmax(large_scores)
        print(f"Softmax Result: {large_probs}")
        if np.abs(np.sum(large_probs) - 1.0) < 1e-6 and np.abs(large_probs[0] - 0.3333333) < 1e-5:
            print("✅ Test 2 Passed (Numerical Stability Works)!")
        else:
            print("❌ Test 2 Failed.")
    except RuntimeWarning as e:
        print(f"❌ Test 2 Failed with Exception: {e}")
        print("Hint: Did you forget to subtract the maximum value?")
