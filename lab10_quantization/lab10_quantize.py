import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import time

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
MODEL_CACHE_DIR = "./model_cache"
DATA_CACHE_DIR = "./data_cache"

def generate_response(model, tokenizer, prompt, max_new_tokens=128):
    """
    Generate a response from the model.
    """
    tokenizer.padding_side = "left"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    stop_token_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=stop_token_ids
        )
    tokenizer.padding_side = "right"
    generated_ids = outputs[0][len(inputs["input_ids"][0]):]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)

def run_qlora_finetuning():
    print("=" * 60)
    print("Lab 10 Task 3: QLoRA Fine-Tuning")
    print("=" * 60)

    bf16_ready = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    
    ### TODO: 1. Configure Quantization (BitsAndBytesConfig)
    # Enable 4-bit, use nf4, enable double quant
    # Hint: Use `torch.bfloat16 if bf16_ready else torch.float16` for compute_dtype
    bnb_config = None

    compute_dtype = torch.bfloat16 if bf16_ready else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type="nf4"
    )

    print("\nLoading 4-bit base model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE_DIR, trust_remote_code=True)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token

    ### TODO: 2. Load the 4-bit Base Model
    # Use AutoModelForCausalLM.from_pretrained with your bnb_config
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=MODEL_CACHE_DIR,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=compute_dtype
    )

    if model is not None:
        print(f"Memory footprint of 4-bit base model: {model.get_memory_footprint() / 1024 / 1024:.2f} MB")

    ### TODO: 3. Prepare model for k-bit training (prepare_model_for_kbit_training)
    
    model = prepare_model_for_kbit_training(model)

    ### TODO: 4. Configure and Inject LoRA Adapters (LoraConfig, get_peft_model)
    # Target modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
    # Set r=8, lora_alpha=16

    lora_config = LoraConfig(
        r = 8, # Rank of the LoRA update
        lora_alpha = 16, # Scaling factor for the LoRA update
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"], # Modules to apply LoRA to
        lora_dropout = 0.05, # Dropout for LoRA layers
        bias = "none", # No bias in LoRA layers
        task_type = "CAUSAL_LM", # Causal language modeling task
    )
    
    # 5. Load Dataset and use SFTTrainer formatting function
    print("\nLoading dataset...")
    dataset = load_dataset("tatsu-lab/alpaca", split="train[:3000]", cache_dir=DATA_CACHE_DIR)
    
    def format_chatml(example):
        if example['input'].strip():
            return f"<|im_start|>user\n{example['instruction']}\n{example['input']}<|im_end|>\n<|im_start|>assistant\n{example['output']}<|im_end|>"
        return f"<|im_start|>user\n{example['instruction']}<|im_end|>\n<|im_start|>assistant\n{example['output']}<|im_end|>"

    if model is None:
        print("Model loading not implemented. Exiting...")
        return

    # Test generation before training
    instruction = "Summarize the following text."
    input_text = "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions."
    test_prompt = f"<|im_start|>user\n{instruction}\n{input_text}<|im_end|>\n<|im_start|>assistant\n"
    
    print("\n--- Before Fine-Tuning ---")
    print(f"Prompt: {test_prompt}")
    print(f"Response: {generate_response(model, tokenizer, test_prompt)}")

    ### TODO: 6. Configure Trainer (SFTConfig, SFTTrainer)
    # Important SFTConfig/TrainingArguments:
    # optim="paged_adamw_32bit", bf16=bf16_ready, fp16=(not bf16_ready)
    # Pass `format_chatml` to `formatting_func` in SFTTrainer
    # Set `max_length=512` and `dataset_text_field="text"` in SFTConfig

    model = get_peft_model(model, lora_config)

    sft_config = SFTConfig(
        output_dir = "./qlora_finetuning_output", # Directory to save checkpoints and logs
        per_device_train_batch_size = 4, # Batch size per device during training
        gradient_accumulation_steps = 4, # Number of steps to accumulate gradients before updating
        optim = "paged_adamw_32bit", # Optimizer to use
        bf16 = bf16_ready, # Use bfloat16 if supported
        fp16 = not bf16_ready, # Use float16 if bfloat16 is not supported
        max_steps = 10, # Total number of training steps
        logging_steps = 100, # Log training metrics every X steps
        save_strategy = "no", # Disable automatic checkpoint saving
        report_to = "none", # Disable reporting to external services (e.g., WandB)
        max_length = 512, # Maximum length of the input sequence
        dataset_text_field = "text", # Field in the dataset containing the text
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        formatting_func=format_chatml,
        args=sft_config,
    )
    
    print("\nStarting QLoRA fine-tuning...")
    start_time = time.time()
    
    trainer.train()
    
    print(f"Training completed in {time.time() - start_time:.2f} seconds.")
    
    print("\n--- After Fine-Tuning ---")
    print(f"Response: {generate_response(model, tokenizer, test_prompt)}")
    
    # trainer.model.save_pretrained("./lab10_qlora_adapter")

if __name__ == "__main__":
    run_qlora_finetuning()
