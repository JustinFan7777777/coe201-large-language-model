# Lab 9: Interactive Chatbot Comparison
# This script allows you to chat with either the Base model or your Fine-tuned LoRA model.

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

# Set Hugging Face mirror endpoint for stable access
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
MODEL_CACHE_DIR = "./model_cache"
ADAPTER_PATH = "./lab9_lora_adapter"

def format_prompt(query):
    return f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"

def main():
    print("=" * 60)
    print("Lab 9: SFT Model Chatbot")
    print("=" * 60)

    mode = input("Select mode (base / lora): ").strip().lower()
    if mode not in ["base", "lora"]:
        print("Invalid mode. Defaulting to base.")
        mode = "base"

    print(f"\nLoading tokenizer and {mode} model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE_DIR, trust_remote_code=True)
    # Use native EOS for consistency with Task 2
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.float16, 
        cache_dir=MODEL_CACHE_DIR
    ).to("cuda")

    if mode == "lora":
        if not os.path.exists(ADAPTER_PATH):
            print(f"Error: Adapter not found at {ADAPTER_PATH}. Did you run Task 2?")
            return
        
        # Load the LoRA adapter onto the base model
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)
        print("LoRA adapter loaded!")

    print("\nChatbot ready! (Type 'exit' or 'quit' to stop)")
    print("-" * 30)

    # Pre-calculate stop token IDs for speed
    stop_token_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")]

    while True:
        query = input("\nUser> ").strip()
        if query.lower() in ["exit", "quit"]:
            break
        
        prompt = format_prompt(query)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        print("\nAssistant: ", end="", flush=True)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                eos_token_id=stop_token_ids,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Decode only the new tokens
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(response.strip())
        print("\n" + "-" * 30)

if __name__ == "__main__":
    main()
