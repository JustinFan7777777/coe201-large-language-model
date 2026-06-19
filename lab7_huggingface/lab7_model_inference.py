# Lab 7 Task 1: Model Inference Pipeline [35 points]

import torch
import torch.nn.functional as F
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    print("Please install transformers: pip install transformers")
    exit(1)


def filter_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    Apply Top-P (nucleus) sampling filter to logits.

    This helper removes unlikely tokens whose cumulative probability exceeds top_p,
    then returns the filtered logits (with removed tokens set to -inf).

    Args:
        logits: shape (batch_size, vocab_size)
        top_p: cumulative probability threshold (e.g., 0.9)

    Returns:
        Filtered logits with the same shape.
    """
    if top_p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Keep tokens until cumulative prob exceeds top_p
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    # Scatter the mask back to original indices
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = -float('Inf')
    return logits


def generate_text(model, tokenizer, prompt, max_new_tokens=10, temperature=1.0, top_p=1.0):
    """
    Generate text manually token-by-token.

    Args:
        model: HuggingFace causal LM model
        tokenizer: HuggingFace tokenizer
        prompt (str): Input text
        max_new_tokens (int): Maximum number of tokens to generate
        temperature (float): Temperature for sampling
        top_p (float): Top-p (nucleus) sampling threshold

    Returns:
        str: The generated text (including the prompt)
    """
    device = next(model.parameters()).device

    # 1. Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    print(f"Prompt: {prompt}")
    print("Generating...", end="", flush=True)

    safe_temperature = max(temperature, 1e-6)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            ### TODO: Implement the generation loop
            # 1. Forward pass: get logits from the model
            # 2. Extract logits for the LAST token
            # 3. Apply temperature scaling to the logits
            # 4. Apply top-p filtering using the provided `filter_top_p` helper
            # 5. Convert filtered logits to probabilities and sample using torch.multinomial
            # 6. Append the predicted token to input_ids
            # 7. Check if the predicted token is EOS (tokenizer.eos_token_id), if so break

            # --- Your code starts here ---

            # step 1: forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # step 2: get logits for the last token
            next_token_logits = outputs.logits[:, -1, :]
            # [:, -1, :] means that we are taking the logits for the last token in the sequence for each batch item

            # step 3: apple temperature scaling
            next_token_logits = next_token_logits / safe_temperature
            # if temperature < 1, it makes the distribution sharper (more likely to pick high-prob tokens), more conservative
            # if temperature > 1, it makes the distribution softer (more likely to pick low-prob tokens), more creative

            # step 4: apply top-p filtering
            # we only consider the tokens that together make up the top_p cumulative probability mass, and set the rest to -inf
            filtered_logits = filter_top_p(next_token_logits.clone(), top_p)
            
            # step 5: convert to probabilities and sample
            probs = F.softmax(filtered_logits, dim=-1)
            # dim=-1 means we are applying softmax across the vocabulary dimension
            
            next_token = torch.multinomial(probs, num_samples=1)
            # shape (batch_size, 1)
            # for each item in the batch, we only sample one token, based on the probability distribution defined by probs
            # num_samples=1 means we want to sample one token for each item in the batch
            # multinomial will return the index of the sampled token in the vocabulary
            # based on the probabilities in probs, it will randomly select a token index for each batch item
            # with higher-prob tokens more likely to be selected
            
            # step 6: append the predicted token to input_ids
            # concatenate the new token to the existing input_ids to form the new input for the next iteration
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            # dim=-1 means we are concatenating along the sequence length dimension
            # so we are adding the new token to the end of the sequence

            # step 7: update the attention mask to include the new token
            next_mask = torch.ones(attention_mask.shape[0], 1, dtype=attention_mask.dtype, device=device)
            # create a mask of 1s for the new token, shape (batch_size, 1)
            # meaning that the new token should be attended to
            # shape[0] is the batch size, we keep it the same, 1 is the new token we are adding
            # we use the same dtype and device as the existing attention_mask to ensure compatibility
            # ones means the token is valid and should be attended to, zeros would mean it is padding and should not be attended to
            
            attention_mask = torch.cat([attention_mask, next_mask], dim=-1)
            # concatenate the new mask to the existing attention_mask along the sequence length dimension
            # so we are updating the attention mask to account for the new token

            # step 8: check for EOS token
            if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
                break

            # --- Your code ends here ---

        # Decode the complete sequence
        output_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return output_text


def main():
    print("Testing Task 1: Manual Inference Loop")
    # Using a small model for testing
    model_name = "models/Qwen2.5-0.5B"  # Local path
    print(f"Loading {model_name} (this may take a minute if downloading)...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    except Exception as e:
        print(f"Failed to load model. Do you have network access? Error: {e}")
        return

    model.eval()

    prompt = "The capital of France is"

    print("\n--- Greedy (Temp=0.01) ---")
    greedy = generate_text(model, tokenizer, prompt, max_new_tokens=10, temperature=0.01, top_p=1.0)
    print(f"\nResult: {greedy}")

    print("\n--- Sampling (Temp=0.8, Top-p=0.9) ---")
    sampled = generate_text(model, tokenizer, prompt, max_new_tokens=10, temperature=0.8, top_p=0.9)
    print(f"\nResult: {sampled}")

    print("\nTask 1 execution finished! (Check if the outputs make sense)")


if __name__ == "__main__":
    main()
