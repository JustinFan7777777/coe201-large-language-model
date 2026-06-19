# task4_beam_search.py
import numpy as np

# ==========================================
# Task 4 (Optional): Manual Beam Search Walkthrough
# ==========================================
def task4_beam_search():
    print("\n--- Task 4: (Optional) Beam Search Implementation ---")
    
    # Vocabulary: {0: "<PAD>", 1: "The", 2: "A", 3: "cat", 4: "dog"}
    idx_to_word = {0: "<PAD>", 1: "The", 2: "A", 3: "cat", 4: "dog"}
    
    # Step 1 log-probabilities (simulated from model)
    # log(0.6) = -0.5108, log(0.4) = -0.9163
    step1_log_probs = {1: np.log(0.6), 2: np.log(0.4)}
    
    # Step 2 log-probabilities given the previous word
    # If "The" (idx 1): {"cat": 0.1, "dog": 0.9}
    # If "A" (idx 2):   {"cat": 0.9, "dog": 0.1}
    step2_log_probs = {
        1: {3: np.log(0.1), 4: np.log(0.9)},
        2: {3: np.log(0.9), 4: np.log(0.1)}
    }
    
    beam_width = 2
    
    # TODO 1: Initialize candidates for Step 1
    # Each candidate should be a tuple: (cumulative_log_prob, sequence_list)
    candidates = [] # [(log_p, [id1]), (log_p, [id2])]

    for word_id, log_prob in step1_log_probs.items():
        # word_id: 1 is "The", 2 is "A"
        # log_prob: log(0.6) for "The", log(0.4) for "A"
        candidates.append((log_prob, [word_id]))
    # candidates now contains: [(-0.5108, [1]), (-0.9163, [2])]
    
    # TODO 2: Step 2 Expansion
    # 1. For each candidate, expand it by trying all possible next words in step2_log_probs
    # 2. Store all resulting expansions (score + next_log_prob, sequence + [next_word])
    all_expansions = []

    # We loop through each candidate from step 1
    for current_log_prob, current_seq in candidates:

        # we get the last word in the current sequence
        last_word_id = current_seq[-1]

        # we get the poosible next word probabilities based on the last word
        next_word_probs = step2_log_probs[last_word_id]

        # we loop through each possible next word and its log probability
        for next_word_id, next_log_prob in next_word_probs.items():

            # culculate by cumulative log probability
            # we add up the two log probabilities, as:
            # log(P1*P2) = log(P1) + log(P2)
            new_log_prob = current_log_prob + next_log_prob

            # create a new sequence by appending the next word id to the current sequence
            # for example: [1] + [3] = [1, 3], corresponding to ["The", "cat"]
            new_seq = current_seq + [next_word_id]

            # we store the new log probability and the new sequence as a tuple in all_expansions
            all_expansions.append((new_log_prob, new_seq))

    # At this point, all_expansions should contain:
    # [(-0.5108 + log(0.1), [1,3]),  # "The cat"
    #  (-0.5108 + log(0.9), [1,4]),  # "The dog"  
    #  (-0.9163 + log(0.9), [2,3]),  # "A cat"
    #  (-0.9163 + log(0.1), [2,4])]  # "A dog"

    # TODO 3: Beam Selection
    # 1. Sort all_expansions by their cumulative score (descending)
    # 2. Pick the top 'beam_width' candidates as the final result

    # sort all_expansions by log probability in descending order
    # lambda x: x[0]: sort by the first element of the tuple
    # which is the cumulative log probability (metric for ranking)
    all_expansions.sort(key=lambda x: x[0], reverse=True)

    # we pick up the top 'beam_width' candidates from the sorted list
    # which are the top 2 sequences with the highest cumulative log probabilities
    top_beams = all_expansions[:beam_width]
    
    print("Top 2 beams:")
    for score, seq in top_beams:
        sentence = " ".join([idx_to_word[i] for i in seq])
        print(f"Sequence: {sentence}, Prob: {np.exp(score):.4f}")

if __name__ == "__main__":
    task4_beam_search()
