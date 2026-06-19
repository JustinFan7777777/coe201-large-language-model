# Lab 9 Task 2: SFT Fine-Tuning with LoRA [30 points]

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
import os

# Set Hugging Face mirror endpoint for stable access
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
MODEL_CACHE_DIR = "./model_cache"
DATA_CACHE_DIR = "./data_cache"


def load_sft_dataset(tokenizer, max_length=512):
    """
    Load the Alpaca dataset and transform it into ChatML format for SFT.
    """
    # Load a subset of the Alpaca dataset for demonstration
    dataset = load_dataset("tatsu-lab/alpaca", split="train[:3000]", cache_dir=DATA_CACHE_DIR)

    def tokenize_fn(examples):
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        
        for instruction, input_text, output in zip(
            examples["instruction"], examples["input"], examples["output"]
        ):
            # 1. Build the prompt using the Qwen ChatML template
            # TODO: Convert the Alpaca instruction/input into the ChatML format:
            # "<|im_start|>user\n{instruction}\n{input_text}<|im_end|>\n<|im_start|>assistant\n"
            if input_text.strip():
                prompt = f"<|im_start|>user\n{instruction}\n{input_text}<|im_end|>\n<|im_start|>assistant\n"
            else:
                prompt = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
            
            # The 'output' is the target response from the assistant.
            # We append BOTH the ChatML end tag and the model's native EOS token (<|endoftext|>).
            response = output + "<|im_end|>" + tokenizer.eos_token
            
            # 2. Tokenize prompt and response separately to avoid boundary token merging
            # If we tokenize the concatenated string, BPE might merge characters at the join point,
            # causing a 1-token shift that misaligns our labels.
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            response_ids = tokenizer.encode(response, add_special_tokens=False)
            
            # 3. Concatenate and create labels
            # Use -100 for prompt tokens so the model doesn't calculate loss on the user instruction.
            # This is called 'Label Masking'.
            input_ids = prompt_ids + response_ids
            labels = [-100] * len(prompt_ids) + response_ids
            
            # 4. Manual truncation if the combined length exceeds max_length
            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                labels = labels[:max_length]
            
            # 5. Create attention mask (1 for real tokens, 0 for padding tokens)
            attention_mask = [1] * len(input_ids)
                
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)

        return {
            "input_ids": input_ids_list, 
            "attention_mask": attention_mask_list, 
            "labels": labels_list
        }

    # Apply the tokenization in batches and remove raw text columns
    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
    )
    return tokenized


def setup_lora_model(model_name: str, r: int = 16, alpha: int = 32, torch_dtype=torch.float16):
    """
    Load a pre-trained model and wrap it with LoRA using PEFT.
    """
    # Load base model in specified precision (explicitly to cuda)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch_dtype, 
        cache_dir=MODEL_CACHE_DIR
    ).to("cuda")

    ### TODO: Create LoRA config and wrap the model
    # 1. Create LoraConfig with correct arguments:
    #    - task_type
    #    - r, lora_alpha, lora_dropout
    #    - target_modules
    # 2. Use get_peft_model(model, config) to wrap
    # 3. Print trainable parameters using model.print_trainable_parameters()

    # --- Your code starts here ---
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # Specify that this is a causal language modeling task
        r=r,  # Rank of the LoRA decomposition
        lora_alpha=alpha,  # Scaling factor for the LoRA contribution
        lora_dropout=0.05, # Optional dropout for regularization
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # Target attention projection layers
            "gate_proj", "up_proj", "down_proj"  # Target feedforward layers
        ],
        bias="none"  # We do not apply LoRA to bias terms
    )

    model = get_peft_model(model, lora_config)
    # Wrap the base model with LoRA using the specified configuration

    model.print_trainable_parameters() 
    # Print the number of trainable parameters to verify that only LoRA parameters are being updated
    
    # --- Your code ends here ---

    return model


def generate_response(model, tokenizer, prompt, max_new_tokens=128):
    """
    Generate a response from the model.
    Note: We use left padding for inference to ensure that the last token 
    seen by the model is a real token, which is essential for batched inference.
    """
    tokenizer.padding_side = "left"
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
    
    # Restore padding to 'right' for subsequent training/collator steps
    tokenizer.padding_side = "right"
    
    # Decode only the newly generated tokens (excluding the prompt)
    generated_ids = outputs[0][len(inputs["input_ids"][0]):]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def main():
    print("=" * 60)
    print("Task 2: SFT Fine-Tuning with LoRA")
    print("=" * 60)

    # 0. Check hardware for BF16 support
    bf16_ready = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16_ready else torch.float16
    print(f"Using dtype: {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE_DIR, trust_remote_code=True)
    # Use native EOS (<|endoftext|>) to ensure the base model stops correctly
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    
    # --- DIAGNOSTIC CHECK (Optional) ---
    print(f"Native EOS ID: {tokenizer.eos_token_id} ({tokenizer.eos_token})")
    # ----------------------------------

    # 1. Load and tokenize dataset
    print("\nLoading dataset...")
    dataset = load_sft_dataset(tokenizer)
    print(f"Dataset size: {len(dataset)} examples")
    print(f"Sample input_ids length: {len(dataset[0]['input_ids'])}")

    # 2. Setup LoRA model
    print("\nSetting up LoRA model...")
    model = setup_lora_model(MODEL_NAME, r=8, alpha=16, torch_dtype=dtype)

    # 3. Test generation before fine-tuning
    instruction = "Summarize the following text."
    input_text = "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions."
    test_prompt = f"<|im_start|>user\n{instruction}\n{input_text}<|im_end|>\n<|im_start|>assistant\n"
    print("\n--- Before Fine-Tuning ---")
    response = generate_response(model, tokenizer, test_prompt)
    print(f"Prompt: {test_prompt}")
    print(f"Response: {response}")

    # 4. Train (simplified training loop using Hugging Face Trainer)
    from transformers import Trainer, DataCollatorForSeq2Seq

    training_args = TrainingArguments(
        output_dir="./lab9_output",
        num_train_epochs=5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        bf16=bf16_ready,
        fp16=not bf16_ready,
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",
        remove_unused_columns=False,
        weight_decay=0.01,
        warmup_ratio=0.03,
    )

    # Use input_ids as labels for causal LM (teacher forcing with label shift)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    print("\nStarting training...")
    trainer.train()

    # Save the trained adapter for Task 3
    print("\nSaving adapter to ./lab9_lora_adapter...")
    trainer.save_model("./lab9_lora_adapter")

    # 5. Test generation after fine-tuning
    print("\n--- After Fine-Tuning ---")
    response = generate_response(model, tokenizer, test_prompt)
    print(f"Prompt: {test_prompt}")
    print(f"Response: {response}")

    print("\nTask 2 finished!")


if __name__ == "__main__":
    main()
