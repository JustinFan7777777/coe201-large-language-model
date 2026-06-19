# task1_embeddings.py
import torch
import numpy as np
from gensim.models import KeyedVectors

# ==========================================
# Task 1: Real Word Embeddings & Manual Similarity
# ==========================================

def cosine_similarity_manual(v1, v2):
    """
    TODO: Implement Cosine Similarity manually using torch or numpy.
    Formula: (A dot B) / (||A|| * ||B||)
    """
    # Hint: Use torch.dot() and torch.norm() or numpy equivalents

    # float: ensure we get a float result from the division
    # to avoid integer division issues
    v1 = v1.float()
    v2 = v2.float()

    # Compute the dot product and two norms
    dot_product = torch.dot(v1, v2)
    norm_v1 = torch.norm(v1)
    norm_v2 = torch.norm(v2)

    # Add a small epsilon to the denominator
    # to prevent division by zero in case of zero-length vectors
    eps = 1e-12
    cosine = dot_product / (norm_v1 * norm_v2 + eps)

    # item: Convert the single-value tensor to a Python float
    # to ensure we return a standard float instead of a tensor
    # maintain compatibility with other code that expects a float
    return cosine.item()

def task1_embeddings():
    print("--- Task 1: Embeddings ---")
    
    # Load local pre-trained GloVe model
    print("Loading local GloVe model (50d)...")
    # 50 dimensions is a common choice for small demos
    model = KeyedVectors.load("glove-50d.model")
    
    # We extract vectors for a few words
    words = ["king", "queen", "man", "woman", "apple", "banana"]
    # Create a dictionary of word to vector mappings
    # We convert the vectors to torch tensors during this step
    vectors = {word: torch.tensor(model[word]) for word in words}
    
    # 2. Test your manual implementation
    v_king = vectors["king"]
    v_queen = vectors["queen"]
    v_apple = vectors["apple"]
    
    # Calculate similarities using the manual function
    sim_king_queen = cosine_similarity_manual(v_king, v_queen)
    sim_king_apple = cosine_similarity_manual(v_king, v_apple)
    
    print(f"Similarity (king vs queen): {sim_king_queen}")
    print(f"Similarity (king vs apple): {sim_king_apple}")
    
    # 3. Bonus: Word Analogies
    # Test classic patterns using your manual similarity function!
    # Pattern: A - B + C ≈ D
    print("\n--- Bonus: Analogies ---")
    
    # TODO: Analogy 1 (Gender): Queen - Woman + Man ≈ King

    # compute the analogy vector by performing the vector arithmetic
    analogy1 = torch.tensor(model["queen"]) - torch.tensor(model["woman"]) + torch.tensor(model["man"])
    # compute the similarity between "analogy1" & "king" using the manual cosine similarity function
    sim_analogy1_king = cosine_similarity_manual(analogy1, torch.tensor(model["king"]))
    print(f"Analogy 1 -> sim((queen - woman + man), king): {sim_analogy1_king}")
    
    # TODO: Analogy 2 (Capital-Country): Paris - France + China ≈ Beijing

    # the same as above
    analogy2 = torch.tensor(model["paris"]) - torch.tensor(model["france"]) + torch.tensor(model["china"])
    sim_analogy2_beijing = cosine_similarity_manual(analogy2, torch.tensor(model["beijing"]))
    print(f"Analogy 2 -> sim((paris - france + china), beijing): {sim_analogy2_beijing}")
    
    # TODO: Find your own! (e.g., Verb Tense, Comparative, etc.)
    # my example: Verb Tense - Walk - Walking + Run ≈ Running
    # which tests the verb tense pattern in the embeddings
    analogy3 = torch.tensor(model["walk"]) - torch.tensor(model["walking"]) + torch.tensor(model["run"])
    sim_analogy3_running = cosine_similarity_manual(analogy3, torch.tensor(model["running"]))
    print(f"Analogy 3 -> sim((walk - walking + run), running): {sim_analogy3_running}")

if __name__ == "__main__":
    task1_embeddings()
