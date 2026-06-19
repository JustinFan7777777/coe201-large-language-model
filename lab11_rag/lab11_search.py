import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Computes the cosine similarity between two 1D vectors.
    """
    ### TODO: Implement basic cosine similarity (Optional helper, but good practice)

    dot_product = np.dot(a, b) # dot product of a & b
    
    norm_a = np.linalg.norm(a) + 1e-10 # norm of a with epsilon to avoid division by zero
    norm_b = np.linalg.norm(b) + 1e-10 # norm of b with epsilon to avoid division by zero

    cosine_sim = dot_product / (norm_a * norm_b) # cosine similarity formula

    return cosine_sim


def vector_search(
    query_emb: np.ndarray, doc_embs: np.ndarray, k: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """
    Finds the top-K most similar documents to the query using cosine similarity.

    Args:
        query_emb (np.ndarray): 1D array of shape (D,) representing the query embedding.
        doc_embs (np.ndarray): 2D array of shape (N, D) representing document embeddings.
        k (int): The number of top documents to retrieve.

    Returns:
        tuple: (top_k_indices, top_k_scores)
    """
    ### TODO: Implement Vector Search

    # 1. Normalize the query vector and document vectors.
    #    Hint: Add a small epsilon (1e-10) to the norm to avoid division by zero.
    query_norm = np.linalg.norm(query_emb, keepdims=True)
    query_normalized = query_emb / (query_norm + 1e-10)

    doc_norms = np.linalg.norm(doc_embs, axis=1, keepdims=True)
    # axis=1 means we compute the norm for each document vector (row-wise)
    doc_normalized = doc_embs / (doc_norms + 1e-10)

    # 2. Compute similarity scores using dot product (since vectors are normalized).
    #    Hint: Use np.dot for matrix multiplication.
    similarity_scores = np.dot(doc_normalized, query_normalized) # dot product to get cosine similarity scores
    similarity_scores = similarity_scores.flatten() # flatten to 1D array

    # 3. Extract the indices of the top-K scores.
    #    Hint: np.argsort sorts in ascending order. You need the highest scores.
    sorted_indices = np.argsort(similarity_scores)[::-1] # sort indices in descending order
    top_k_indices = sorted_indices[:k] # get top K indices

    # 4. Return the indices and the corresponding scores.
    top_k_scores = similarity_scores[top_k_indices] # get corresponding scores for top K indices

    return top_k_indices, top_k_scores


if __name__ == "__main__":
    print("=" * 60)
    print("Lab 11: Task 2 - Simple Vector Search")
    print("=" * 60)

    # Generate some dummy embeddings
    np.random.seed(42)
    D = 128  # Embedding dimension
    N = 1000  # Number of documents

    doc_embeddings = np.random.randn(N, D)
    query_embedding = np.random.randn(D)

    # Inject a highly similar document manually
    target_idx = 42
    doc_embeddings[target_idx] = query_embedding * 0.9 + np.random.randn(D) * 0.1

    print(f"Searching across {N} documents for top 3 matches...")

    try:
        top_indices, top_scores = vector_search(query_embedding, doc_embeddings, k=3)

        print("\nResults:")
        for rank, (idx, score) in enumerate(zip(top_indices, top_scores), 1):
            print(f"Rank {rank}: Document {idx:03d} (Score: {score:.4f})")

        assert top_indices[0] == target_idx, (
            "Failed to retrieve the most similar document!"
        )
        print("\nVector search test passed!")
    except Exception as e:
        print(f"\nTask 2 Not Implemented or Error: {e}")
