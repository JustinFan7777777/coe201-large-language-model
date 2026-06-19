import torch
# torch provides tensors and operations for deep learning
import torch.nn as nn
# nn provides modules and classes for building neural network
# including RNN, LSTM, GRU, etc.

# Task 1: Warm-up - Sequence Shapes
# Goal: Understand the difference between output and hn

# batch_size is the number of sequences we process in parallel
batch_size = 5

# seq_len is the length of each sequence (3 words in a sentence, for example)
# also the number of time steps the RNN processes
seq_len = 3

# input_size is the dimensionalty of the input features (e.g., word embedding size)
input_size = 10

# hidden_size is the dimensionality of the hidden state in the RNN.
# it determines how much information the RNN can store and process at each time step.
hidden_size = 20 

# 1. Initialize RNN
rnn = nn.RNN(input_size, hidden_size, batch_first=True)
# input_size: the number of expected features in the input, also known as the embedding size.
# hidden_size: the number of features in the hidden state -> determines the capacity of the RNN to capture information.
# batch_first=True: indicates that the input and output tensors will have the batch size as the first dimension (i.e., (batch, seq, feature)).
# If False, the expected shape would be (seq, batch, feature). This is a common setting for PyTorch RNNs to make it easier to work with batches of data.

# 2. Create dummy input
x = torch.randn(batch_size, seq_len, input_size)
# generate a random tensor according to normal distribution
# The shape of x is (5, 3, 10) which means:
# - 5 sequences in the batch (batch_size)
# - Each sequence has 3 time steps (seq_len)
# - Each time step has 10 features (input_size)

# 3. Forward pass
output, hn = rnn(x)
# we give the input x to the RNN, and it returns two things:
# 1. output: the hidden states for every time step in the sequence
# with shape (batch_size, seq_len, hidden_size) -> (5, 3, 20)
# that is, for each of the 5 sequences, for each of the 3 time steps in the sequence, we get a hidde state of size 30, which captures the information
# 2. hn: the hidden state for the last time-step for each sequence in the batch
# with shape (num_layers, batch_size, hidden_size) -> (1, 5, 20)
# Since we are using a single-layer RNN, num_layers is 1.

print(f"Input shape:  {x.shape}")
# (Batch, Seq, Feature) -> (5, 3, 10)

print(f"Output shape: {output.shape}") 
# (Batch, Seq, Hidden) -> (5, 3, 20): Hidden state for EVERY step

print(f"hn shape:     {hn.shape}")    
# (Layers, Batch, Hidden) -> (1, 5, 20): Hidden state for LAST step only

# Verification Task:
# 1. Extract the hidden state of the LAST time step from the 'output' tensor.
# 2. Check if it matches the 'hn' tensor.
# 3. Print the result.

### TODO: Your code here (1-2 lines)

# Extract the last time step's hidden state from output
last_step_output = output[:, -1, :]
# output has shape (batch_size, seq_len, hidden_size) -> (5, 3, 20)
# we only want the last-time step, which is at index -1 along the seq_len dimension
# as a result, seq_len dimension will be compressed, or removed
# the what we will get is: last_step_output with shape (batch_size, hidden_size) -> (5, 20)

# Compare with hn (first layer's hidden state)
are_equal = torch.allclose(last_step_output, hn[0])
# last_step_output has shape (5, 20)
# hn has shape (1, 5, 20)
# hn[0] has shape (5, 20)
# they should have the same shape and match
# since they both indicate the hidden state at the last time step
# for each sequence in the batch, the hidden state (information) should be the same

print(f"Are they equal? {are_equal}") # Should be True if they match!

### END TODO

print("\n" + "="*40)
print("Part 2: RNN vs LSTM Shapes")
print("="*40)

# 1. Initialize LSTM
lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
# create a LSTM layer:
# input_size = 10 (same as before)
# hidden_size = 20 (same as before)
# batch_first=True means the input and output tensors will have the batch size as the first dimension

# 2. Forward pass
# Note: LSTM returns (output, (hn, cn))
output_lstm, (hn_lstm, cn_lstm) = lstm(x)
# this is the key difference between RNN and LSTM:
# LSTM returns an additional tensor 'cn' which is the cell state, representing long-term memory
# 1. output_lstm: hidden states for every time step, shape (batch_size, seq_len, hidden_size) -> (5, 3, 20)
# 2. hn_lstm: hidden state only for the last time step, shape (num_layers, batch_size, hidden_size) -> (1, 5, 20)
# notice that we only have 1 layer, so num_layers is 1 in this case
# 3. cn_lstm: cell state for the last time step, shape (num_layers, batch_size, hidden_size) -> (1, 5, 20)
# the cell state (cn) is what allows LSTM to maintain long-term memory
# while the hidden state (hn) captures short-term information at the last time step

print(f"LSTM Output shape: {output_lstm.shape}") # (Batch, Seq, Hidden) -> (5, 3, 20) -> hidden states for EVERY step
print(f"LSTM hn shape:     {hn_lstm.shape}")     # (Layers, Batch, Hidden) -> (1, 5, 20) -> hidden state for LAST step only
print(f"LSTM cn shape:     {cn_lstm.shape}")     # (Layers, Batch, Hidden) -> (1, 5, 20) -> Cell state for LAST step only to maintain long-term memory

print("\nKey Takeaway: LSTM maintains an extra 'Cell State' (cn) for long-term memory.")