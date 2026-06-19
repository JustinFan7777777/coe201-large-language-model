"""
Assignment 1 - Problem 1: NumPy Vectorized Operations (30 points)

In this problem, you will implement several core numerical operations using NumPy.
These operations are fundamental to deep learning frameworks and are used extensively
in neural network implementations.

IMPORTANT CONSTRAINTS:
- You are forbidden from using any Python 'for' or 'while' loops.
- All operations must be vectorized using NumPy's built-in functions.
- Using loops will result in a 50% penalty for that specific part, even if correct.

Example:
    >>> import numpy as np
    >>> A = np.array([[1, 2], [3, 4]])
    >>> B = np.array([[5], [6]])
    >>> matrix_multiply(A, B)
    array([[17],
           [39]])
"""
import numpy as np

# ============================================================
# Part A: Basic Operations (10 pts)
# ============================================================

def matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Implement standard matrix multiplication A * B.

    Args:
        A: Left matrix of shape (m, n)
        B: Right matrix of shape (n, k)

    Returns:
        C: Result matrix of shape (m, k) where C = A @ B

    Hint:
        Use np.dot() or the @ operator for matrix multiplication.

    Example:
        >>> A = np.random.randn(5, 3)
        >>> B = np.random.randn(3, 4)
        >>> C = matrix_multiply(A, B)
        >>> C.shape
        (5, 4)
    """
    ### TODO: Implement matrix multiplication

    C = A @ B
    # @ or np.dot can be used for matrix multiplication
    return C
    
    ### END TODO


def normalize_rows(M: np.ndarray) -> np.ndarray:
    """
    Normalize each row of M to have unit L2 norm.

    For each row vector v, compute: v_normalized = v / ||v||_2
    If a row has zero norm (all zeros), leave it as zeros to avoid division by zero.

    Args:
        M: Input matrix of shape (m, n)

    Returns:
        M_normalized: Matrix of shape (m, n) where each row has unit L2 norm

    Example:
        >>> M = np.array([[3, 4], [0, 0], [1, 2, 3]])  # Note: last row would cause error
        >>> normalize_rows(np.array([[3, 4], [0, 0]]))
        array([[0.6, 0.8],
               [0. , 0. ]])
    """
    ### TODO: Write your code below (2-3 lines)

    row_norms = np.linalg.norm(M, axis=1, keepdims=True)
    # linalg means linear algebra, norm computes the L2 norm by default
    # L2 norm: ||v||_2 = sqrt(sum_i(v_i^2))
    # axis=1 means we compute the norm across columns for each row
    # keepdims=True keeps the dimensions for broadcasting

    safe_norms = np.where(row_norms > 0, row_norms, 1.0)
    # Replace zero norms with 1 to avoid division by zero
    # np.where(condition, x, y) returns an array
    # where elements are x if condition is True, else y
    # in this case, if row_norms > 0, we keep the norm
    # otherwise we set it to 1
    # This way, rows that are all zeros will have a norm of 1
    # and dividing by 1 will still keep them as zeros

    M_normalized = M / safe_norms
    # Normalize each row by its norm
    # this will broadcast the division across each row
    # for rows with all zeros (safe_norms = 1), this will keep them as zeros
    # avoiding division by zero issues

    return M_normalized

    ### END TODO


def create_one_hot(indices: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Convert a 1D array of class indices into a 2D one-hot matrix.

    A one-hot encoding is a binary vector where all elements are 0 except for the
    index corresponding to the class, which is 1.

    Args:
        indices: 1D array of shape (N,) containing class indices (0 to num_classes-1)
        num_classes: Total number of classes (K)

    Returns:
        one_hot: 2D array of shape (N, num_classes) where one_hot[i, indices[i]] = 1
    
    Example:
        >>> indices = np.array([0, 2, 1])
        >>> create_one_hot(indices, 3)
        array([[1., 0., 0.],
               [0., 0., 1.],
               [0., 1., 0.]])
    """
    ### TODO: Write your code below (2-3 lines)

    one_hot = np.eye(num_classes, dtype=float)[indices]
    # np.eye(num_classes) creates an identity matrix of size (num_classes, num_classes)
    # indexing this matrix with 'indices' will select the appropriate rows
    # for example, if indices = [0, 2, 1] and num_classes = 3, np.eye(3) is:
    # [[1., 0., 0.],
    #  [0., 1., 0.],
    #  [0., 0., 1.]]
    # then np.eye(3)[indices] will give:
    # [[1., 0., 0.],  # for index 0
    #  [0., 0., 1.],  # for index 2
    #  [0., 1., 0.]]  # for index 1

    return one_hot

    ### END TODO


# ============================================================
# Part B: Statistical Operations (10 pts)
# ============================================================

def column_standardize(X: np.ndarray) -> np.ndarray:
    """
    Standardize features (columns) to have zero mean and unit variance.

    For each feature j, compute: z_j = (x_j - mean_j) / std_j

    This is a common preprocessing step in machine learning that ensures all
    features are on the same scale.

    Args:
        X: Input data matrix of shape (N, D) where each column is a feature

    Returns:
        X_standardized: Matrix of shape (N, D) with zero mean and unit variance per column

    Example:
        >>> X = np.array([[1, 2], [2, 4], [3, 6]])
        >>> X_std = column_standardize(X)
        >>> np.allclose(X_std.mean(axis=0), 0)
        True
        >>> np.allclose(X_std.std(axis=0), 1)
        True
    """
    ### TODO: Write your code below (3-4 lines)

    mean = X.mean(axis=0, keepdims=True)
    # Compute the mean of each column (feature)
    # axis=0: compute mean across rows for each column
    # keepdims=True keeps the dimensions for broadcasting
    # This will give us a (1, D) array where each element is the mean of that column
    # that is we compress the N samples into a single mean value for each feature

    std = X.std(axis=0, keepdims=True)
    # the same, we compute the standard deviation for each column (feature)
    # this will give us a (1, D) array where each element is the std of that column

    safe_std = np.where(std > 0, std, 1.0)
    # we traverse through the std array and check if any value is zero
    # which would cause division by zero)
    # if std > 0, we keep this std value
    # otherwise, we set it to 1 to avoid division by zero
    # mention that std=0 means all values in that column are the same
    # in that case, when subtracting the mean, all values will become zero
    # and dividing by 1 will keep them as zeros

    X_standardized = (X - mean) / safe_std
    # standardize each column by subtracting the mean and dividing by the std
    # for each feature, this will ensure that the resulting column 
    # has zero mean and unit variance

    return X_standardized

    ### END TODO


def pairwise_l2_distance(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute the Euclidean distance between every pair of rows from A and B.

    The pairwise L2 distance formula can be expanded as:
    ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a^T * b

    This formulation allows for efficient vectorized computation without loops.

    Args:
        A: Matrix of shape (m, d) containing m data points
        B: Matrix of shape (n, d) containing n data points

    Returns:
        D: Distance matrix of shape (m, n) where D[i, j] = ||A[i] - B[j]||_2

    Example:
        >>> A = np.array([[0, 0], [1, 0]])
        >>> B = np.array([[0, 0], [0, 3]])
        >>> pairwise_l2_distance(A, B)
        array([[0., 3.],
               [1., 3.16...]])
    """
    ### TODO: Implement pairwise L2 distance

    A_squared = np.sum(A ** 2, axis=1, keepdims=True)
    # Compute the squared L2 norm of each row in A
    # this time we have axis=1 to sum across columns for each row
    # this will give us a (m, 1) array
    # where each element is the squared norm of that row in A

    B_squared = np.sum(B ** 2, axis=1, keepdims=True).T
    # Compute the squared L2 norm of each row in B
    # this will give us a (n, 1) array
    # we transpose it to get a (1, n) array
    # where each element is the squared norm of that row in B

    sq_distances = A_squared + B_squared - 2 * A @ B.T
    # Compute the pairwise squared distances using the expanded formula
    # A @ B.T computes the dot product between rows of A and rows of B
    # this will give us a (m, n) matrix where each element
    # is the dot product of a row from A and a row from B
    # we multiply by 2 and subtract from the sum of squared norms to get the squared distances

    sq_distances = np.maximum(sq_distances, 0.0)
    # Due to numerical precision issues, some distances might be slightly negative
    # We use np.maximum to set any negative distances to zero

    distances = np.sqrt(sq_distances)
    # Take the square root to get the actual L2 distances

    return distances

    ### END TODO


# ============================================================
# Part C: Softmax & Loss (10 pts)
# ============================================================

def softmax_batch(logits: np.ndarray) -> np.ndarray:
    """
    Implement numerically stable softmax for a batch of inputs.

    The softmax function converts logits (raw scores) into probabilities:
    softmax(x)_i = exp(x_i) / sum_j(exp(x_j))

    For numerical stability, we subtract the maximum value before exponentiating:
    softmax(x)_i = exp(x_i - max(x)) / sum_j(exp(x_j - max(x)))

    Args:
        logits: Input array of shape (N, K) containing raw scores for K classes

    Returns:
        probs: Probability array of shape (N, K) where each row sums to 1


    Example:
        >>> logits = np.array([[1, 2, 3], [3, 2, 1]])
        >>> probs = softmax_batch(logits)
        >>> probs
        array([[0.09..., 0.24..., 0.66...],
               [0.66..., 0.24..., 0.09...]])
        >>> np.allclose(probs.sum(axis=1), 1.0)
        True
    """
    ### TODO: Write your code below (3-4 lines)

    shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
    # for each row, we subtract the maximum value in that row from all elements in that row
    # this ensures numerical stability when we exponentiate, preventing overflow

    exp_logits = np.exp(shifted_logits)
    # for each element, we compute the exponential of the shifted logits

    sum_exp = np.sum(exp_logits, axis=1, keepdims=True)
    # for each row, we compute the sum of the exponentials
    # this will give us a (N, 1) array

    probs = exp_logits / sum_exp
    # we divide each element in exp_logits by the corresponding sum of exponentials for that row
    # this will give us the softmax probabilities, where each row sums to 1

    return probs

    ### END TODO


def cross_entropy_loss(probs: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute the mean cross-entropy loss.

    Cross-entropy loss measures the difference between predicted probabilities
    and ground truth labels:
    L = -1/N * sum_i(log(p_i))

    where p_i is the predicted probability for the true class of sample i.

    Args:
        probs: Probability matrix of shape (N, K) - output of softmax
        targets: 1D array of shape (N,) containing ground truth class indices

    Returns:
        loss: Scalar value representing the mean cross-entropy loss

    Hint:
        Use advanced indexing to extract the probability of the correct class
        for each sample, then compute -mean(log(p)).
        Add a small epsilon (1e-12) inside log to avoid log(0).

    Example:
        >>> probs = np.array([[0.1, 0.9], [0.8, 0.2]])
        >>> targets = np.array([1, 0])
        >>> cross_entropy_loss(probs, targets)
        0.1115...
    """
    ### TODO: Implement cross-entropy loss

    N = probs.shape[0]
    # probs: (N, K)
    # where N is the number of samples
    # and K is the number of classes
    # we need N to compute the mean loss

    correct_class_probs = probs[np.arange(N), targets]
    # Use advanced indexing to extract the predicted probability
    # for the correct class for each sample
    # np.arange(N) creates an array [0, 1, 2, ..., N-1]
    # targets contains the class indices for each sample
    # so probs[np.arange(N), targets] will give us an array of shape (N,)
    # where each element is the predicted probability for the correct class of that sample

    epsilon = 1e-12
    # A small constant to prevent log(0) which would result in -inf
    # This ensures numerical stability when computing the log of probabilities

    loss = -np.mean(np.log(correct_class_probs + epsilon))
    # Compute the mean cross-entropy loss
    # We take the log of the correct class probabilities, add epsilon for stability,
    # and then take the negative mean to get the final loss value

    return float(loss)

    ### END TODO
