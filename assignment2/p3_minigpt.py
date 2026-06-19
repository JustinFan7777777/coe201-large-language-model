import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """Provided Sinusoidal Positional Encoding from Lab 6."""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class MiniGPT(nn.Module):
    """
    Problem 3: MiniGPT Training and Generation [40 points]
    
    A MiniGPT is a Decoder-only Transformer. It predicts the probability of the
    next token given the previous tokens.
    """
    def __init__(self, vocab_size: int, d_model: int, nhead: int, num_layers: int, max_seq_len: int = 512):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_seq_len)
        
        # A GPT is essentially a Transformer Encoder with a causal (upper triangular) mask!
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        ### TODO: Implement the forward pass. (10 pts)
        
        1. Look up embeddings for `input_ids`.
        2. Multiply embeddings by sqrt(d_model) for scaling.
        3. Apply positional encoding.
        4. Generate a causal mask using `nn.Transformer.generate_square_subsequent_mask`
           and move it to the correct device.
        5. Pass the sequence and the causal mask through `self.transformer`.
           (Hint: pass the mask to the `mask` argument, not `src_key_padding_mask`).
        6. Project the output to vocab space using `self.lm_head`.
        
        Returns:
            logits: shape (batch_size, seq_len, vocab_size)
        """
        # --- Your code starts here ---
        
        batch_size, seq_len = input_ids.size()

        # step 1: embedding lookup
        # input_ids shape: (batch_size, seq_len)
        # embedding shape: (batch_size, seq_len, d_model)
        x = self.embedding(input_ids)

        # step 2: scale embeddings
        # we scale the embeddings by sqrt(d_model) to prevent the dot products
        # in the attention mechanism from growing too large, which can lead to vanishing gradients.
        x = x * math.sqrt(self.d_model)
        
        # step 3: add positional encoding
        # x shape: (batch_size, seq_len, d_model)
        x = self.pos_encoding(x)

        # step 4: generate causal mask
        # The causal mask is an upper triangular matrix that prevents the model from attending to future tokens.
        # mask shape: (seq_len, seq_len), where mask[i, j] = 0 if j <= i else -inf (or a large negative value) to mask out future tokens.
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len)
        # The generated mask is of shape (seq_len, seq_len) and is on the CPU by default.
        # We need to move it to the same device as the input tensor `x` to avoid device mismatch errors during the transformer pass.
        mask = mask.to(x.device)

        # step 5: pass through transformer
        # The transformer expects input of shape (batch_size, seq_len, d_model) and a mask of shape (seq_len, seq_len).
        x = self.transformer(x, mask=mask)

        # step 6: project to vocab space
        # x shape: (batch_size, seq_len, d_model)
        # logits shape: (batch_size, seq_len, vocab_size)
        logits = self.lm_head(x)
        # lm head is a linear layer that maps the d_model-dimensional output of the transformer to the vocab_size-dimensional output space
        # The output `logits` contains the unnormalized probabilities for each token in the vocabulary at each position in the sequence.
        # For example, logits[i, j, k] gives the logit for token k at position j in the i-th sequence in the batch.

        return logits

        # --- Your code ends here ---

    def compute_loss(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        ### TODO: Implement the Causal Language Modeling loss. (15 pts)
        
        For next-token prediction, the target for the token at position `t` is 
        the token at position `t+1`.
        
        1. Get logits from `self.forward(input_ids)`.
        2. Shift logits and targets so that they align.
           - Shifted logits should contain predictions for positions 1 to seq_len-1.
           - Shifted targets should contain the true tokens for positions 1 to seq_len-1.
        3. Flatten the batch and sequence dimensions and compute CrossEntropyLoss.
        
        Returns:
            Scalar loss tensor.
        """
        # --- Your code starts here ---
        
        # step 1: get logits from forward pass
        logits = self.forward(input_ids)
        # logits shape: (batch_size, seq_len, vocab_size)

        # step 2: align logits & targets
        # logits[t] should predict input_ids[t+1]
        # so we shift logits to exclude the last time step and targets to exclude the first time step
        shift_logits = logits[:, :-1, :] # shape: (batch_size, seq_len-1, vocab_size)
        # [:, :-1, :] means we take all batches, all tokens except the last one, and all vocab logits

        shift_labels = input_ids[:, 1:] # shape: (batch_size, seq_len-1)
        # [:, 1:] means we take all batches and all tokens except the first one
        # after shifting, shift_labels[t] is the true token that shift_logits[t] should predict

        # step 3: flatten & compute cross-entropy loss
        # CrossEntropyLoss expects input of shape:
        # logits: (N, C), where N is the number of samples and C is the number of classes
        # labels: (N,), where each value is the class index (0 to C-1)
        # We need to flatten the batch and sequence dimensions together
        shift_logits_flat = shift_logits.reshape(-1, self.vocab_size)
        # reshape(-1, vocab_size) collapses the batch and sequence dimensions into one dimension of size (batch_size * (seq_len-1))

        shift_labels_flat = shift_labels.reshape(-1)
        # reshape(-1) collapses the batch and sequence dimensions into one dimension of size (batch_size * (seq_len-1))

        loss = F.cross_entropy(shift_logits_flat, shift_labels_flat)
        # F.cross_entropy combines log_softmax and nll_loss in one function
        # It computes the loss by comparing the predicted logits with the true labels.
        # it returns a scalar loss value that we can backpropagate.

        return loss

        # --- Your code ends here ---

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        ### TODO: Implement autoregressive Greedy Generation. (15 pts)
        
        1. Loop `max_new_tokens` times.
        2. In each iteration, pass the current `input_ids` through the model to get logits.
        3. Extract the logits for the very last token in the sequence.
        4. Find the token ID with the maximum probability (greedy search) using `torch.argmax`.
        5. Append this new token ID to the `input_ids` tensor along the sequence dimension.
        6. Return the full generated sequence.
        
        Returns:
            Tensor of shape (batch_size, seq_len + max_new_tokens)
        """
        # --- Your code starts here ---
        
        # loop to generate max_new_tokens one at a time
        for _ in range(max_new_tokens):
            # step 1: forward pass to get logits
            logits = self.forward(input_ids)
            # current input_ids shape: (batch_size, current_seq_len)
            # logits shape: (batch_size, current_seq_len, vocab_size)
            # current input_ids can grow in length with each iteration as we append new tokens

            # step 2: extract logits for the last token in the sequence
            # we only care about the last time step's logits to predict the next token
            # as the past tokens are already known (can't change) and we want to predict the next token based on all previous tokens
            last_token_logits = logits[:, -1, :] # shape: (batch_size, vocab_size)

            # step 3: greedy search to find the token ID with the maximum probability
            next_token_id = torch.argmax(last_token_logits, dim=-1, keepdim=True) # shape: (batch_size, 1)
            # torch.argmax returns the index of the maximum logit for each batch, which corresponds to the predicted next token ID
            # dim=-1 means we take the argmax across the vocab dimension, and keepdim=True keeps the output shape as (batch_size, 1) for easy concatenation

            # step 4: append the new token ID to the input_ids tensor
            input_ids = torch.cat([input_ids, next_token_id], dim=1)
            # torch.cat concatenates the current input_ids with the new token ID along the sequence dimension
            # after concatenation, input_ids shape becomes (batch_size, current_seq_len + 1)

        return input_ids
        # we loop until we have generated max_new_tokens
        # so after the loop, input_ids will have shape (batch_size, original_seq_len + max_new_tokens)

        # --- Your code ends here ---

# =============================================================================
# Provided Code Below (Do not modify, but run it to train your model)
# =============================================================================

import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

def load_dataset(file_path="tinyshakespeare.txt"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file {file_path} not found! Please ensure it is in the same directory.")
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

class CharDataset(Dataset):
    def __init__(self, data: str, seq_len: int):
        chars = sorted(list(set(data)))
        self.vocab_size = len(chars)
        self.char2idx = {ch: i for i, ch in enumerate(chars)}
        self.idx2char = {i: ch for i, ch in enumerate(chars)}
        
        # Convert text to integers
        self.data = torch.tensor([self.char2idx[c] for c in data], dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        # Return seq_len + 1 tokens because the compute_loss slices input vs target
        chunk = self.data[idx:idx + self.seq_len + 1]
        return chunk

def train_minigpt(model, dataloader, epochs=1, lr=1e-3, device='cpu'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    accuracies = []
    
    print(f"Starting training on {device}...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_tokens = 0
        
        for batch_idx, batch in enumerate(dataloader):
            batch = batch.to(device)
            
            # ### TODO: Implement the training step (10 pts)
            # Step 1: Zero the gradients of the optimizer.
            # Step 2: Compute the loss using `model.compute_loss(batch)`.
            # (Note: we need the variable `loss` for tracking metrics later)
            # Step 3: Backpropagate the loss.
            # Step 4: Step the optimizer.
            
            # --- Your code starts here ---

            # step 1: zero gradients
            optimizer.zero_grad()
            # clear the gradients of all optimized parameters before the backward pass to prevent accumulation of gradients from multiple batches

            # step 2: forward pass & compute loss
            loss = model.compute_loss(batch)
            # compute_loss internally calls the forward method to get logits and then computes the cross-entropy
            # loss is a scalar tensor that we will backpropagate, which measures how well the model's predictions match the true next tokens in the sequence

            # step 3: backpropagate the loss
            loss.backward()
            # computes the gradient of the loss with respect to all model parameters and stores them in the .grad attribute of each parameter

            # step 4: update parameters
            optimizer.step()
            # updates the model parameters based on the computed gradients and the learning rate defined in the optimizer

            # --- Your code ends here ---
            
            # --- Track Metrics ---
            total_loss += loss.item() * batch.size(0)
            
            # Compute accuracy manually just for metrics
            with torch.no_grad():
                logits = model(batch)
                shift_logits = logits[:, :-1, :]
                shift_labels = batch[:, 1:]
                preds = torch.argmax(shift_logits, dim=-1)
                correct = (preds == shift_labels).sum().item()
                total_correct += correct
                total_tokens += shift_labels.numel()
                
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(dataloader.dataset)
        avg_acc = total_correct / total_tokens
        losses.append(avg_loss)
        accuracies.append(avg_acc)
        print(f"--- Epoch {epoch+1} Summary | Avg Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f} ---")
        
    # Plotting
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), losses, marker='o', color='red')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs+1), accuracies, marker='o', color='blue')
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    
    plt.tight_layout()
    plt.savefig("minigpt_training_metrics.png")
    print("Saved training metrics plot to 'minigpt_training_metrics.png'")
    
    return losses, accuracies

def main():
    # --- Provided Code Below (Feel free to modify code below to improve performance) ---
    # 1. Prepare Data
    text = load_dataset()
    seq_len = 64
    batch_size = 64
    
    dataset = CharDataset(text, seq_len=seq_len)
    # Using a larger subset (e.g., 50000 samples) to ensure better learning
    # while keeping lab training time reasonable.
    train_size = min(50000, len(dataset))
    subset_indices = torch.randperm(len(dataset))[:train_size]
    subset = torch.utils.data.Subset(dataset, subset_indices)
    dataloader = DataLoader(subset, batch_size=batch_size, shuffle=True)
    
    print(f"Vocabulary Size: {dataset.vocab_size} (Character-level)")
    
    # 2. Recommended Model Configuration
    vocab_size = dataset.vocab_size # Usually 65 for TinyShakespeare
    d_model = 128
    nhead = 4
    num_layers = 4
    max_seq_len = 256
    
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    
    model = MiniGPT(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        max_seq_len=max_seq_len
    )
    
    # 3. Train
    train_minigpt(model, dataloader, epochs=10, lr=2e-3, device=device)
    
    # 4. Generate some text
    model.eval()
    context = "O God, "
    input_ids = torch.tensor([dataset.char2idx[c] for c in context], dtype=torch.long).unsqueeze(0).to(device)
    generated_ids = model.generate(input_ids, max_new_tokens=100)
    generated_text = "".join([dataset.idx2char[i.item()] for i in generated_ids[0]])
    print("\n--- Generated Text ---")
    print(generated_text)
    print("----------------------")

if __name__ == "__main__":
    main()
