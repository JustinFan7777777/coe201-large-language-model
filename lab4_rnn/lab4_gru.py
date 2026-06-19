# lab4_gru.py

import torch
import torch.nn as nn
import torch.optim as optim
import os
import csv
import random

# ==========================================
# Task 3 [Bonus]: Extreme GRU Challenge
# ==========================================

class CustomGRUCell(nn.Module):
    """
    Implement a single GRU step from scratch.
    Reference the gate equations in the slides.
    """
    def __init__(self, input_size, hidden_size):
        super(CustomGRUCell, self).__init__()
        ### TODO: Define your parameters (wr, wz, wn, etc.)
        
        self.input_size = input_size
        self.hidden_size = hidden_size # Define the linear layers for the gates

        # Reset gate parameters：
        # r_t = sigmoid(w_r * x_t + u_r * h_{t-1} + b_r)
        # function: controls how much of the previous hidden state to forget
        # and how much of the new input to consider for the candidate hidden state: n_t
        # if r_t is close to 0, the model forgets the previous hidden state and relies more on the new input
        # if r_t is close to 1, the model retains more of the previous hidden state and considers less of the new input for n_t
        self.w_r = nn.Linear(input_size, hidden_size, bias=True) # reset gate weights, maps input to hidden
        self.u_r = nn.Linear(hidden_size, hidden_size, bias=True) # reset gate weights, maps hidden to hidden

        # Update gate parameters:
        # z_t = sigmoid(w_z * x_t + u_z * h_{t-1} + b_z)
        # function: controls how much of the previous hidden state to keep
        # and how much of the new candidate hidden state to use for the final hidden state: h_t
        # if z_t is close to 0, the model updates the hidden state mostly with the new candidate n_t
        # if z_t is close to 1, the model retains most of the previous hidden state and updates it less with n_t
        self.w_z = nn.Linear(input_size, hidden_size, bias=True) # update gate weights, maps input to hidden
        self.u_z = nn.Linear(hidden_size, hidden_size, bias=True) # update gate, maps hidden to hidden

        # Candidate hidden state parameters:
        # n_t = tanh(w_n * x_t + u_n * (r_t * h_{t-1}) + b_n)
        # function: computes the new candidate hidden state based on the current input and the previous hidden state modulated by the reset gate
        # if r_t is close to 0, the candidate hidden state n_t is computed mostly from the new input x_t, allowing the model to reset its memory
        # if r_t is close to 1, the candidate hidden state n_t incorporates more information from the previous hidden state, allowing the model to retain memory over time
        self.w_n = nn.Linear(input_size, hidden_size, bias=True) # new gate weights, maps input to hidden
        self.u_n = nn.Linear(hidden_size, hidden_size, bias=True) # new gate weights, maps hidden to hidden 

    def forward(self, x, h_prev):
        """
        x: (batch, input_size)
        h_prev: (batch, hidden_size)
        """
        ### TODO: Implement the GRU forward logic
        # return h_t

        # reset gate
        # w_r(x) extracts features from the current input x, while u_r(h_prev) extracts features from the previous hidden state h_prev.
        # the sigmoid activation squashes the combined features into a value between 0 and 1, which determines how much of the previous hidden state to reset (forget) when computing the candidate hidden state n_t.
        r_t = torch.sigmoid(self.w_r(x) + self.u_r(h_prev)) # reset gate, shape: (batch, hidden_size)

        # update gate
        # w_z(x) extracts features from the current input x, while u_z(h_prev) extracts features from the previous hidden state h_prev.
        # the sigmoid activation squashes the combined features into a value between 0 and 1, which determines how much of the previous hidden state to keep when computing the final hidden state h_t.
        z_t = torch.sigmoid(self.w_z(x) + self.u_z(h_prev)) # update gate, shape: (batch, hidden_size)

        # candidate hidden state
        # w_n(x) extracts features from the current input x, while u_n(r_t * h_prev) extracts features from the previous hidden state h_prev modulated by the reset gate r_t.
        # the reset gate r_t controls how much of the previous hidden state h_prev is considered when computing the candidate hidden state n_t.
        n_t = torch.tanh(self.w_n(x) + r_t * self.u_n(h_prev)) # new gate, shape: (batch, hidden_size)

        # final hidden state
        # the final hidden state h_t is a combination of the previous hidden state h_prev and the candidate hidden state n_t, weighted by the update gate z_t.
        # if z_t is close to 1, the model retains more of the previous hidden state h_prev and updates it less with the new candidate n_t.
        # if z_t is close to 0, the model updates the hidden state mostly with the new candidate n_t and retains less of the previous hidden state h_prev.
        h_t = (1 - z_t) * n_t + z_t * h_prev # final hidden state, shape: (batch, hidden_size)
        
        return h_t

class SentimentGRU(nn.Module):
    """
    A many-to-one Sentiment Classifier using your CustomGRUCell.
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(SentimentGRU, self).__init__()
        ### TODO: Initialize embedding, gru_cell, and fc layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # embedding layer, maps vocab indices to dense vectors
        # vocab_size: size of the vocabulary, determines the number of rows in the embedding matrix
        # embed_dim: dimensionality of the embedding vectors, determines the number of columns in the embedding matrix
        # padding_idx=0 means that the embedding for index 0 will be a vector of zeros
        # which is useful for handling padded sequences without introducing noise into the model's learning process

        self.gru_cell = CustomGRUCell(embed_dim, hidden_dim)
        # GRU cell, processes one time step at a time, takes embedded input and previous hidden state, outputs new hidden state
        # embed_dim: dimensionality of the input to the GRU cell, which is the same as the embedding dimension
        # hidden_dim: dimensionality of the hidden state in the GRU cell
        # determines the capacity of the model to capture information from the sequence

        self.fc = nn.Linear(hidden_dim, output_dim)
        # fully connected layer, maps final hidden state to output logits
        # hidden_dim: dimensionality of the input to the fully connected layer
        # which is the same as the hidden dimension of the GRU cell
        # output_dim: dimensionality of the output logits
        # which corresponds to the number of classes (e.g., 2 for binary classification)

    def forward(self, x):
        """
        x shape: (Batch_Size, Seq_Len)
        """
        ### TODO: Implement the sequence loop and return the final classification logits

        embedded = self.embedding(x)
        # x shape: (Batch_Size, Seq_Len) -> embedded shape: (Batch_Size, Seq_Len, Embed_Dim)
        # the embedding layer transforms the input sequence of token indices
        # into a sequence of dense vectors (embeddings) that capture semantic information about the tokens.

        batch_size, seq_len, embedded_dim = embedded.size()
        # batch_size: number of sequences in the batch
        # seq_len: length of each sequence (number of time steps)
        # embedded_dim: dimensionality of the embedding vectors

        h_t = torch.zeros(batch_size, self.gru_cell.hidden_size, device=embedded.device)
        # initialize hidden state, shape: (Batch_Size, Hidden_Dim), set to zeros

        # loop through each time step in the sequence
        for t in range(seq_len):
            x_t = embedded[:, t, :]
            # get the t-th time step input, shape: (Batch_Size, Embed_Dim)
            
            h_t = self.gru_cell(x_t, h_t)
            # update hidden state using GRU cell

        logits = self.fc(h_t)
        # pass the final hidden state through the fully connected layer to get logits
        # logits shape: (Batch_Size, Output_Dim)
        
        return logits


# ==========================================
# Challenge: Short vs. Long Sequence Analysis
# ==========================================

def subgroup_analysis_test():
    """
    [YOUR TASK]
    1. Load the SST-2 dataset (train.tsv).
    2. Train your Custom GRU on the mixed dataset.
    3. Evaluate its accuracy on two subgroups of the test set:
       - LONG sequences (tokens > 35)
       - SHORT sequences (tokens <= 15)
    
    Observe how the GRU handles different sequence lengths. 
    Does it maintain high accuracy even on the longer reviews?
    """

    # 直接复用 RNN 文件中已经写好的工具函数，避免重复造轮子
    from lab4_rnn import load_data_from_tsv, SimpleTokenizer, collate_batch

    # 0. 读取数据
    tsv_path = "train.tsv"
    full_data = load_data_from_tsv(tsv_path, limit=30000)

    if not full_data:
        print("❌ 数据加载失败，程序退出。")
        return

    # 1. 打乱并划分训练集 / 测试集
    random.seed(42)
    random.shuffle(full_data)

    split_idx = int(len(full_data) * 0.9)
    train_data = full_data[:split_idx]
    test_data = full_data[split_idx:]

    # 2. 构建 tokenizer
    # 只用训练集建词表，避免测试集信息泄露
    tokenizer = SimpleTokenizer(train_data)

    # 3. 初始化模型
    EMBED_DIM = 128
    HIDDEN_DIM = 256
    OUTPUT_DIM = 2
    BATCH_SIZE = 32
    EPOCHS = 10

    model = SentimentGRU(
        vocab_size=len(tokenizer.word2idx),
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
    )

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 4. 训练模型
    print("\n=== 开始训练 Custom GRU ===")
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0

        for i in range(0, len(train_data), BATCH_SIZE):
            batch = train_data[i:i + BATCH_SIZE]
            inputs, labels = collate_batch(batch, tokenizer)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = total_loss / (len(train_data) // BATCH_SIZE + 1)
        train_acc = 100 * correct / total
        print(f"Epoch {epoch + 1} | Loss: {avg_loss:.4f} | Accuracy: {train_acc:.2f}%")

    # 5. 定义一个评估子集准确率的小函数
    def evaluate_subset(data_subset, name):
        if len(data_subset) == 0:
            print(f"{name}: 没有样本，跳过评估。")
            return

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for i in range(0, len(data_subset), BATCH_SIZE):
                batch = data_subset[i:i + BATCH_SIZE]
                inputs, labels = collate_batch(batch, tokenizer)

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        print(f"{name} Accuracy: {acc:.2f}% | Samples: {total}")

    # 6. 按句子长度划分测试集
    long_test_data = []
    short_test_data = []

    for text, label in test_data:
        token_len = len(text.split())

        if token_len > 35:
            long_test_data.append((text, label))

        if token_len <= 15:
            short_test_data.append((text, label))

    # 7. 输出分组统计
    print("\n=== Subgroup Analysis ===")
    print(f"Total test samples : {len(test_data)}")
    print(f"LONG samples       : {len(long_test_data)}")
    print(f"SHORT samples      : {len(short_test_data)}")

    # 8. 分别评估长句和短句
    evaluate_subset(long_test_data, "LONG  (>35 tokens)")
    evaluate_subset(short_test_data, "SHORT (<=15 tokens)")

    print("🚀 Challenge: Implement the subgroup analysis test here!")

if __name__ == "__main__":
    subgroup_analysis_test()
