# Lab 7 Task 3: LLM Evaluation Pipeline [30 points]

import torch
import torch.nn.functional as F
import json
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    print("Please install transformers")
    exit(1)


def calculate_perplexity(model, tokenizer, text: str) -> float:
    """
    Calculate perplexity of text under the model.

    Perplexity = exp(average negative log-likelihood)
    Lower perplexity means the model is less "surprised" by the text.

    Args:
        model: HuggingFace causal LM model
        tokenizer: HuggingFace tokenizer
        text: Text to evaluate

    Returns:
        Perplexity value (float)
    """
    device = next(model.parameters()).device

    ### TODO: Implement perplexity calculation (20 pts)
    # Step 1: Tokenize the text into input_ids
    # Step 2: Forward pass to get logits
    # Step 3: Compute log probabilities using F.log_softmax
    # Step 4: For each position, select the log_prob of the actual next token
    #         (Hint: use logits[:, :-1] to predict tokens[1:])
    # Step 5: Calculate average negative log-likelihood
    # Step 6: Return exp(average NLL)

    # --- Your code starts here ---
    
    # step 1: tokenize the text
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)

    with torch.no_grad():
        # step 2: forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        # logits shape: (batch_size, seq_len, vocab_size)

        # step 3: shift logits and input_ids to align predictions with targets
        # logits at position t predicts token at position (t + 1)
        # we want to align shift_logits with shift_labels
        shift_logits = logits[:, :-1, :] # shape: (batch_siza, seq_len - 1, vocab_size)
        # [:, :-1, :] because the last token has no next token to predict

        shift_labels = input_ids[:, 1:] # shape: (batch_size, seq_len - 1)
        # [:, 1:] because we want to predict tokens starting from the second token

        shift_mask = attention_mask[:, 1:] # shape: (batch_size, seq_len - 1)
        # [:, 1:] to align with shift_labels

        # step 4: compute log probabilities
        log_probs = F.log_softmax(shift_logits, dim=-1)
        # gather log probabilities of the actual next tokens
        # we need to gather the log_probs at the positions of shift_labels
        # log_probs shape: (batch_size, seq_len - 1, vocab_size)

        token_log_probs = log_probs.gather(
            dim=-1, # gather along vocab dimension
            index=shift_labels.unsqueeze(-1) # index shape: (batch_size, seq_len - 1, 1)
            # unsqueeze(-1) to make it broadcastable for gathering
        ).squeeze(-1) # squeeze to get shape: (batch_size, seq_len - 1)

        # step 5: calculate average negative log-likelihood
        # we only want to consider positions where attention_mask is 1 (not padding)
        # formula: average NLL = -sum(token_log_probs * shift_mask) / sum(shift_mask)

        valid_token_log_probs = token_log_probs * shift_mask
        # valid_token_log_probs means we only keep log_probs for valid tokens (not padding)

        token_count = shift_mask.sum() # total number of valid tokens

        avg_nll = -valid_token_log_probs.sum() / token_count
        # avg_nll is the average negative log-likelihood

        # step 6: return exp(average NLL) as perplexity
        perplexity = torch.exp(avg_nll).item() # item() to get scalar float value

    return float(perplexity)

    # --- Your code ends here ---


def compute_log_likelihood(model, tokenizer, prompt: str, continuation: str) -> float:
    """
    Compute log-likelihood of continuation given prompt.

    This is useful for multiple-choice evaluation: compare likelihoods
    of different answer choices given the same question prompt.

    Args:
        model: HuggingFace causal LM model
        tokenizer: HuggingFace tokenizer
        prompt: The context/question
        continuation: The text to evaluate (e.g., answer choice)

    Returns:
        Log-likelihood value (float, negative)
    """
    device = next(model.parameters()).device

    ### TODO: Implement log-likelihood computation (10 pts)
    # Step 1: Tokenize prompt and continuation separately to find boundary
    # Step 2: Concatenate them and do forward pass
    # Step 3: Extract logits only for the continuation tokens
    # Step 4: Sum log probabilities of continuation tokens

    # --- Your code starts here ---
    
    # step 1: tokenize prompt and continuation separately
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    # tokenizer is used to convert text to token ids. add_special_tokens=False to avoid adding [CLS], [SEP], etc.

    continuation_ids = tokenizer(continuation, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    # return_tensors="pt" to get PyTorch tensors, to(device) to move to the same device as the model

    if continuation_ids.shape[1] == 0:
        # If continuation is empty, return log-likelihood of 0 (log(1))
        return 0.0
    
    # step 2: concatenate prompt and continuation
    input_ids = torch.cat([prompt_ids, continuation_ids], dim=1) # shape: (1, prompt_len + continuation_len)
    # dim=1 to concatenate along the sequence length dimension
    # ids shape: (batch_size=1, seq_len)

    attention_mask = torch.ones_like(input_ids, device=device)
    # ones_like to create attention mask of the same shape as input_ids, since we have no padding
    # attention_mask shape: (1, seq_len)

    with torch.no_grad():
        # step 3: forward pass to get logits
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs.logits # shape: (1, seq_len, vocab_size)
        # for each position in the input, we have a distribution over the vocabulary
        
        log_probs = F.log_softmax(logits, dim=-1) # shape: (1, seq_len, vocab_size)
        # log_probs is the log probability distribution over the vocabulary for each position
        
        # step 4: align logits with continuation tokens
        # the continuation starts at index prompt_len, so we want log_probs[:, prompt_len -:, :]
        prompt_len = prompt_ids.shape[1] # length of the prompt in tokens
        continuation_len = continuation_ids.shape[1] # length of the continuation in tokens

        continuation_pred_log_probs = log_probs[:, prompt_len - 1: prompt_len -1 + continuation_len, :]
        # we use prompt_len - 1 because the token at position prompt_len - 1 predicts the token at position prompt_len (the first token of the continuation)
        # shape: (1, continuation_len, vocab_size)

        # step 5: gather log probabilities of the actual continuation tokens
        continuation_token_log_probs = continuation_pred_log_probs.gather(
            dim=-1,
            index=continuation_ids.unsqueeze(-1) # shape: (1, continuation_len, 1)
        ).squeeze(-1) # shape: (1, continuation_len)

        total_log_likelihood = continuation_token_log_probs.sum().item() # sum log probabilities of continuation tokens

    return float(total_log_likelihood)

    # --- Your code ends here ---


def evaluate_multiple_choice(model, tokenizer, questions_path: str) -> dict:
    """
    Evaluate model on multiple-choice questions.

    For each question, compute log-likelihood of each answer choice
    given the question prompt, and select the one with highest likelihood.

    Args:
        model: HuggingFace causal LM model
        tokenizer: HuggingFace tokenizer
        questions_path: Path to JSON file with questions

    Returns:
        Dictionary with accuracy and per-question results
    """
    # Load questions
    with open(questions_path, 'r') as f:
        questions = json.load(f)

    results = []
    correct = 0

    for q in questions:
        question_text = q['question']
        choices = q['choices']  # List of answer strings
        correct_idx = q['answer']  # Index of correct answer

        ### TODO: Implement multiple-choice evaluation
        # For each choice, compute log-likelihood given the question
        # Select the choice with highest likelihood as prediction
        # Compare with correct answer

        # --- Your code starts here ---
        # Placeholder: predict choice 0 for all questions

        prompt = f"Question: {question_text}\nAnswer:"
        # prompt template, let model know that this is a question and we want to evaluate the answer choices

        choice_scores = []
        for c in choices:
            score = compute_log_likelihood(model, tokenizer, prompt, " " + c)
            # we add a space before the choice to ensure proper tokenization (avoid merging with "Answer:")
            # compute_log_likelihood will give us the log-likelihood of the choice given the prompt, which we can use to compare different choices
            choice_scores.append(score)
            # we store the log-likelihood scores for each choice in a list

        predicted_idx = max(range(len(choice_scores)), key=lambda i: choice_scores[i])
        # we find the index of the choice with the highest log-likelihood score, which is our predicted answer
        # lambda function is used to get the score for each index, and max will return the index with the highest score

        # --- Your code ends here ---

        is_correct = predicted_idx == correct_idx
        if is_correct:
            correct += 1

        results.append({
            'question': question_text,
            'predicted': choices[predicted_idx],
            'correct': choices[correct_idx],
            'is_correct': is_correct
        })

    accuracy = correct / len(questions)
    return {
        'accuracy': accuracy,
        'total': len(questions),
        'correct': correct,
        'results': results
    }


def main():
    print("=" * 60)
    print("Lab 7 Task 3: LLM Evaluation Pipeline")
    print("=" * 60)

    # Load model
    model_name = "models/Qwen2.5-0.5B"  # Local path
    print(f"\nLoading {model_name}...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    except Exception as e:
        print(f"Failed to load model. Error: {e}")
        return

    model.eval()

    # --- Test Perplexity ---
    print("\n" + "-" * 40)
    print("Part 1: Perplexity Calculation")
    print("-" * 40)

    test_texts = [
        "The capital of France is Paris.",
        "Python is a programming language.",
        "asdfghjkl random text qwerty",  # Should have higher perplexity
    ]

    for text in test_texts:
        ppl = calculate_perplexity(model, tokenizer, text)
        print(f"Text: {text[:40]}...")
        print(f"Perplexity: {ppl:.2f}")
        print()

    # --- Test Multiple Choice Evaluation ---
    print("\n" + "-" * 40)
    print("Part 2: Multiple Choice Benchmark")
    print("-" * 40)

    benchmark_path = "mini_benchmark.json"

    try:
        eval_results = evaluate_multiple_choice(model, tokenizer, benchmark_path)
        print(f"\nAccuracy: {eval_results['accuracy']:.2%}")
        print(f"Correct: {eval_results['correct']}/{eval_results['total']}")

        # Show some results
        print("\nSample results:")
        for r in eval_results['results'][:3]:
            status = "Correct" if r['is_correct'] else "Wrong"
            print(f"  [{status}] Q: {r['question'][:50]}...")
            print(f"       Predicted: {r['predicted']}, Correct: {r['correct']}")
    except FileNotFoundError:
        print(f"Benchmark file not found: {benchmark_path}")
        print("Make sure mini_benchmark.json exists in the current directory.")

    print("\n" + "=" * 60)
    print("Task 3 execution finished!")
    print("=" * 60)


if __name__ == "__main__":
    main()