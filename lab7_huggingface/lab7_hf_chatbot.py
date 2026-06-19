import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    model_name = "models/Qwen2.5-0.5B"  # Local path (same as Task 1 & 3)
    print(f"Loading {model_name}...")
    
    # 1. Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "mps" else torch.float32,
        attn_implementation="eager"
    ).to(device)

    print("\nChatbot initialized! Type '/exit' to stop, or '/clear' to reset history.\n")
    
    # 2. Maintain conversational memory
    messages = [
        {"role": "system", "content": "You are a helpful and concise AI assistant."}
    ]
    
    while True:
        try:
            user_input = input("User> ")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break
            
        if user_input.lower() in ["quit", "exit", "/exit"]:
            break
            
        if user_input.lower() == "/clear":
            messages = [messages[0]] # Keep system prompt
            print("--- History Cleared ---\n")
            continue
            
        # 3. Add user input to messages
        messages.append({"role": "user", "content": user_input})
        
        # --- Your code starts here ---

        # 4. Apply ChatML template to the messages list
        # Hint: set add_generation_prompt=True to get the <|im_start|>assistant header
        # Make sure return_tensors="pt" to get a PyTorch tensor
        
        # ### TODO: Format the messages and get model_inputs

        # Note: The tokenizer's apply_chat_template will handle the conversion of the messages list
        # into the appropriate input format for the model, including adding special tokens and formatting according to the ChatML specification.
        # The resulting model_inputs will contain the input_ids and attention_mask tensors needed for generation.
        model_inputs = tokenizer.apply_chat_template(
            messages,
            # messages: List[Dict[str, str]] - A list of messages in the conversation, where each message is a dictionary with "role" and "content" keys.
            # we transform messages into a format that model can understand

            add_generation_prompt=True,
            # add_generation_prompt: bool - If True, the tokenizer will append a generation prompt
            # (e.g., <|im_start|>assistant) at the end of the input sequence to indicate where the model should start generating its response.
            # the model will know it's time to generate a response after the user input
            # but not consistently repeating user input in the output

            return_tensors="pt",
            # return_tensors: str - The format in which to return the tokenized inputs.
            # Setting this to "pt" (pyTorch) will return PyTorch tensors, which are required for input
            # so that we can directly feed the tokenized inputs into the model for generation

            return_dict=True
            # return_dict: bool - If True, the tokenizer will return a dictionary containing the tokenized inputs (e.g., input_ids, attention_mask) instead of a tuple.
            # This allows us to access the tokenized inputs using keys (e.g., model_inputs["input_ids"]) rather than relying on positional indexing.
        )

        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        
        # 5. Generate response using model.generate()
        # Hint: Pass **model_inputs, and use max_new_tokens=100, do_sample=True, temperature=0.7
        
        # ### TODO: Generate output IDs

        # generate() is a method provided by the model that takes the tokenized input
        # and generates output token IDs based on the model's learned patterns.
        outputs = model.generate(
            **model_inputs,
            # **model_inputs: This syntax unpacks the model_inputs dictionary and passes its contents as keyword arguments to the generate() method.
            # This allows us to pass the input_ids and attention_mask tensors directly to the generate() method without having to specify each one individually.

            max_new_tokens=20,
            # max_new_tokens: the maximum number of new tokens to generate in the response
            # each token corresponds to a word or subword, so this limits the length of the generated response to 100 tokens.

            do_sample=True,
            # do_sample: If True, the model will use sampling to generate responses
            # which can lead to more diverse and creative outputs compared to greedy decoding (only select the most likely token at each step)

            temperature=0.7,
            # temperature: A value that controls the randomness of the generated output.
            # A lower temperature (e.g., 0.7) will make the model more conservative and focused on high-probability tokens

            top_p=0.9,
            # top_p: A value for nucleus sampling, which limits the token selection to a subset of the most probable tokens whose cumulative probability exceeds the specified value

            top_k=50,
            # top_k: A value for top-k sampling, which limits the token selection to the top K most probable tokens at each step of generation

            eos_token_id=tokenizer.eos_token_id,
            # eos_token_id: The token ID that represents the end of a sequence.
            # Setting this ensures that the model will stop generating tokens once it produces the end-of-sequence token, preventing it from generating excessively long responses.

            pad_token_id=tokenizer.eos_token_id
            # pad_token_id: The token ID used for padding sequences to a uniform length.
            # Setting this to the end-of-sequence token ID ensures that any padding added to the input sequences will be treated as the end of the sequence
            # preventing the model from generating responses that include unintended padding tokens.
        )

        # --- Your code ends here ---
        
        # 6. Extract the generated text (ignoring the input prompt)
        if outputs is not None and model_inputs is not None:
            generated_ids = outputs[0][model_inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        else:
            response = "Output is None. Implement the generate logic to see the response!"
        
        print(f"Assistant> {response}\n")
        
        # 7. Add assistant response back to memory
        if outputs is not None:
            messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
