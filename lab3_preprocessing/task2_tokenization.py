# task2_tokenization.py
import torch
import torch.nn as nn
from gensim.models import KeyedVectors

# ==========================================
# Task 2: Tokenization & The Embedding Matrix
# ==========================================

def task2_tokenization():
    print("--- Task 2: Tokenization & Embedding Matrix ---")
    
    # 1. Load Real-World Vocabulary (from GloVe)
    # NOTE: GloVe is a word-level model, so "tokenization" here is just splitting by space.
    print("Loading GloVe model vocabulary...")
    # This model was used in Task 1
    model = KeyedVectors.load("glove-50d.model")
    
    # gensim model.key_to_index is our "Real World Vocabulary"
    vocab = model.key_to_index 
    print(f"Vocabulary size: {len(vocab)}")
    
    sentence = "the quick brown fox jumps over the lazy dog"
    
    # get token_ids 
    token_ids = [vocab[word] for word in sentence.split() if word in vocab]
    print(f"Token IDs: {token_ids}\n")
    
    # Manual Embedding Lookup
    # In this task, the "embedding" is a simple lookup from a pre-defined matrix.
    # shape: (vocab_size, embedding_dim) -> (400000, 50)
    embedding_weight = torch.tensor(model.vectors)
    
    # TODO (Step 1): Manual Indexing. 
    # 1. Find the integer index of the word "fox" in the vocabulary.
    # 2. Extract its 50-dimensional vector from `embedding_weight` using basic Python/PyTorch indexing.

    # find the index of "fox" in the vocabulary that we loaded from GloVe
    fox_idx = vocab["fox"]

    # embedding_weight: a matrix of shape (vocab_size, embedding_dim) -> (400000, 50)
    # we want to extract the vector for "fox" using "fox_idx"
    # that is the "fox_idx"-th row of the matrix "embedding_weight"
    manual_vec = embedding_weight[fox_idx]

    print(f"Manual lookup vector for 'fox' (first 5 dims): {manual_vec[:5] if manual_vec is not None else None}")
    
    # 2. One-hot Equivalence
    # Prove that lookup is equivalent to One-hot vector multiplication!
    
    # TODO (Step 2): One-hot Encoding.
    # Create a vector of zeros with length equal to the vocabulary size.
    # Set the value at the index of "fox" to 1.0.

    # vocab_size: the total number of unique words in our GloVe vocabulary
    vocab_size = len(vocab)

    # create a one-hot vector of shape (1, vocab_size)
    # where all values are 0, except for the index "fox_idx" which is 1
    one_hot = torch.zeros(1, vocab_size)
    one_hot[0, fox_idx] = 1.0
    
    # TODO (Step 3): Matrix Multiplication.
    # Perform matrix multiplication: one_hot @ embedding_weight.
    # This proves that 'lookup' is just a fast way to do 'one_hot @ Weights'.

    # multiplication: (1, vocab_size) @ (vocab_size, embedding_dim) -> (1, embeding_dim)
    # one-hot vector has only a '1' at the ppostion of "fox"
    # so the result is directly the "fox_idx"-th row of the matrix "embedding_weight"
    result_matmul = one_hot @ embedding_weight

    print(f"Matrix multiplication result (first 5 dims): {result_matmul[0, :5] if result_matmul is not None else None}")
    
    # TODO (Step 4): PyTorch Layer.
    # 1. Initialize nn.Embedding(vocab_size, 50).
    # 2. Load the GloVe `embedding_weight` into its .weight.

    vocab_size = len(vocab) # number of unique words in our GloVe vocabulary
    embedding_dim = embedding_weight.shape[1]  # should be 50 for GloVe 50d
    
    # initialize the embedding layer with two arguments
    # which are: the size of the vocabulary & the dimension of the embedding vectors
    embed_layer = nn.Embedding(vocab_size, embedding_dim)
    
    if embed_layer is not None and token_ids:
        # Load the weight into the layer
        with torch.no_grad():
            embed_layer.weight.copy_(embedding_weight)
            
        # TODO (Step 4.1): Use the high-level layer to get the vector for "fox" (index 3 in sentence)

        # token_ids: the list of integer indices corresponding to the all words in our sentence
        # we want to get the vector for "fox", which is at index 3 in our sentence
        ids_tensor = torch.tensor(token_ids)

        # use the embedding layer to get the output vectors for all token_ids in our sentence
        # input: (sentence_length,) -> (9,)
        # output: (sentence_length, embedding_dim) -> (9, 50)
        # that is, we transform each token_id in our sentence into its corresponding embedding vector
        output = embed_layer(ids_tensor)
        
        # TODO (Step 4.2): Manually extract the vector for "fox" from embed_layer.weight
        # just to prove it's the same matrix!

        # embed_layer.weight: the matrix of shape (vocab_size, embedding_dim) that we loaded with GloVe vectors
        # then we extract the vector for "fox" using "fox_idx"
        layer_weight_vec = embed_layer.weight[fox_idx]

        # Verification
        # Index 3 in our sentence "the quick brown fox" is "fox"
        if manual_vec is not None and result_matmul is not None and layer_weight_vec is not None:
            assert torch.allclose(manual_vec, output[3]), "Manual lookup vs nn.Embedding mismatch!"
            assert torch.allclose(manual_vec, result_matmul.squeeze()), "Matmul logic mismatch!"
            assert torch.allclose(manual_vec, layer_weight_vec), "Weight extraction mismatch!"
            print("\nVerification Success!")
            print("1. Manual Lookup (raw weight) == One-hot Matrix Multiplication")
            print("2. One-hot Multiplication == high-level nn.Embedding(ids)")
            print("3. high-level nn.Embedding(ids) == Manual Weight Extraction")

if __name__ == "__main__":
    task2_tokenization()
