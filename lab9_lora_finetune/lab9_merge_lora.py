# Lab 9 Task 3: Manual LoRA Weight Merging [30 points]

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import load_file
import os

# Set Hugging Face mirror endpoint for stable access
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
MODEL_CACHE_DIR = "./model_cache"
ADAPTER_PATH = "./lab9_lora_adapter"

def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 128) -> str:
    """Generate a response from the model given a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Define stop tokens: Native EOS and ChatML <|im_end|>
    stop_token_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=stop_token_ids
        )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

def main():
    print("=" * 60)
    print("Task 3: Manual LoRA Weight Merging")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE_DIR, trust_remote_code=True)
    # Use native EOS for consistency with Task 2
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    
    print("\n1. Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="cpu", cache_dir=MODEL_CACHE_DIR
    )

    print("\n2. Loading adapter weights (safetensors)...")
    try:
        adapter_weights = load_file(os.path.join(ADAPTER_PATH, "adapter_model.safetensors"))
    except Exception as e:
        print(f"Error loading adapter: {e}")
        print("Make sure you ran Task 2 first to generate and save the adapter!")
        return

    print("\n3. Merging weights manually...")
    r = 8
    alpha = 16
    scaling = alpha / r

    merged_count = 0
    # Disable gradients since we are just modifying weights for inference
    with torch.no_grad():
        # ### TODO: Manually merge the LoRA weights into the base model
        # 1. Iterate through all modules using `base_model.named_modules()`
        # 2. Check if the module is an instance of `torch.nn.Linear`
        # 3. Construct the PEFT adapter keys. PEFT adds "base_model.model." as a prefix
        #    to the module name. The keys in safetensors will look like:
        #    A: f"base_model.model.{name}.lora_A.weight"
        #    B: f"base_model.model.{name}.lora_B.weight"
        # 4. If both keys exist in `adapter_weights`:
        #    a) Load A and B matrices and convert them to float32 (for precise math)
        #    b) Compute the delta: delta_W = (B @ A) * scaling
        #    c) Add delta_W to `module.weight.data` (cast delta_W back to module.weight.dtype)
        #    d) Increment `merged_count`

        # --- Your code starts here ---
        
        # trace through all modules in the base model
        for name, module in base_model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # construct the expected keys for A and B
                key_A = f"base_model.model.{name}.lora_A.weight"
                key_B = f"base_model.model.{name}.lora_B.weight"

                if key_A in adapter_weights and key_B in adapter_weights:
                    # load A and B, convert to float32 for precision
                    A = adapter_weights[key_A].to(torch.float32)
                    B = adapter_weights[key_B].to(torch.float32)

                    # compute delta_W = (B @ A) * scaling
                    delta_W = (B @ A) * scaling

                    # add delta_W to the original weight (cast delta_W to module's dtype)
                    module.weight.data += delta_W.to(
                        device=module.weight.data.device,
                        dtype=module.weight.data.dtype
                    )

                    merged_count += 1

        # --- Your code ends here ---

    print(f"Successfully merged LoRA weights into {merged_count} linear layers!")
    if merged_count == 0:
        print("Warning: No layers were merged! Check your key naming logic.")

    print("\n4. Testing generation (Should show fine-tuned behavior with zero overhead)...")
    instruction = "Summarize the following text."
    input_text = "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions."
    test_prompt = f"<|im_start|>user\n{instruction}\n{input_text}<|im_end|>\n<|im_start|>assistant\n"
    response = generate_response(base_model, tokenizer, test_prompt)
    print(f"Prompt: {test_prompt}")
    print(f"Response: {response}")

    print("\nTask 3 finished!")

if __name__ == "__main__":
    main()
