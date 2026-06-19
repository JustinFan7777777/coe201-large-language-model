# lab4_rnn.py

import torch
import torch.nn as nn
import torch.optim as optim
import csv
import os
import random

# ==========================================
# 1. 数据加载与预处理工具
# ==========================================

def load_data_from_tsv(filepath, limit=10000):
    """读取 SST-2 TSV 文件 (label, sentence)"""
    data = []
    if not os.path.exists(filepath):
        print(f"❌ 错误: 找不到文件 '{filepath}'，请确保数据集已放置在正确位置。")
        return []

    print(f"正在读取 {filepath} (限制 {limit} 条数据用于教学) ...")
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader) # label sentence
        
        for row in reader:
            if len(row) < 2:
                continue
            try:
                # SST-2 格式: label \t sentence
                label = int(row[0])
                text = row[1]
                # 过滤过短或过长的句子
                if 5 < len(text.split()) < 50:
                    data.append((text, label))
            except ValueError:
                continue
            
            if len(data) >= limit:
                break
                
    print(f"成功加载 {len(data)} 条数据。")
    return data


class SimpleTokenizer:
    def __init__(self, data, max_vocab_size=5000):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.build_vocab(data, max_vocab_size)

    def build_vocab(self, data, max_vocab_size):
        word_counts = {}
        for text, _ in data:
            for word in text.lower().split():
                word_counts[word] = word_counts.get(word, 0) + 1
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[
            :max_vocab_size
        ]
        for idx, (word, count) in enumerate(sorted_words, start=2):
            self.word2idx[word] = idx
        print(f"词表大小: {len(self.word2idx)}")

    def encode(self, text):
        return [self.word2idx.get(w, 1) for w in text.lower().split()]


def pad_sequences(sequences, padding_value=0):
    """
    对序列进行预填充 (Pre-padding)，使 Batch 内长度一致。

    [思考题] 请阅读上方 pad_sequences 函数的实现。
    为什么这里要从后往前填入数据（即 Pre-padding)?
    提示：回顾 RNN 的隐含状态 $h_n$ 是如何作为序列“总结”的.
    """

    # 步骤 1: 获取最大长度
    max_len = max(len(seq) for seq in sequences)
    
    # 步骤 2: 创建全 0 (PAD) 张量
    padded = torch.full((len(sequences), max_len), padding_value, dtype=torch.long)
    
    # 步骤 3: 将数据填入末尾 (实现从前往后填 0)
    for i, seq in enumerate(sequences):
        length = len(seq)
        padded[i, max_len - length:] = seq
        
    return padded


def collate_batch(batch_data, tokenizer):
    inputs, labels = [], []
    for text, label in batch_data:
        inputs.append(torch.tensor(tokenizer.encode(text), dtype=torch.long))
        labels.append(label)
    
    padded_inputs = pad_sequences(inputs, padding_value=0)
    
    return padded_inputs, torch.tensor(labels, dtype=torch.long)


# ==========================================
# 2. 模型搭建 (本节核心任务)
# ==========================================


class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        """
        Initialization of layers
        """
        super(SentimentRNN, self).__init__()

        # --- Task 2: Implement SentimentRNN (Layers + Forward) ---
        # 1. Embedding Layer: vocab_size -> embed_dim (padding_idx=0)
        # 2. RNN Layer: embed_dim -> hidden_dim (batch_first=True)
        # 3. Linear Layer: hidden_dim -> output_dim (2 classes)
        
        # [思考题] 为什么在 SentimentRNN 中使用 pre-padding (在句子开头填充 0) 
        # 而不是 post-padding (在句子结尾填充 0)？如果换成后者，hn 的结果会有什么问题？

        # answer:
        # In RNNs, the hidden state at the end of the sequence (hn)
        # is often used as a summary of the entire sequence.
        # If we use post-padding (padding at the end),
        # the actual data would be at the beginning of the sequence
        # and the padding would come after it.
        # This means that the RNN would process the real data first and then encounter the padding 
        # which could lead to the hidden state being influenced by the padding tokens
        # especially if the sequence is shorter than the maximum length.
        # By using pre-padding (padding at the beginning)
        # we ensure that the RNN processes the actual data at the end of the sequence,
        # and the hidden state (hn) will be more representative of the real data rather than being diluted by the padding tokens.
        
        ### TODO: Define your layers (3 lines)

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0) 
        # 1. Embedding Layer:
        # vocab_size: size of the vocabulary
        # embed_dim: dimension of the embedding vectors
        # padding_idx=0: ensures that the embedding for the padding token (index 0)
        # is a zero vector and is not updated during training
        # This layer will convert input token indices
        # into dense vectors of fixed size (embed_dim).

        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        # 2. RNN Layer:
        # embed_dim: input size to the RNN (the dimension of the embeddings)
        # hidden_dim: the dimension of the hidden state in the RNN
        # batch_first=True: indicates that the input and output tensors
        # are provided as (batch, seq_len, feature)
        # This layer will process the sequence of embeddings
        # and produce hidden states for each time step
        # as well as a final hidden state (hn) that summarizes the entire sequence.

        self.fc = nn.Linear(hidden_dim, output_dim)
        # 3. Linear Layer:
        # hidden_dim: the size of the input to the linear layer
        # output_dim: the size of the output (number of classes, which is 2 for SST-2)
        # This layer will take the final hidden state from the RNN
        # and produce logits for each class (positive/negative)

        ### END TODO

    def forward(self, x):
        """
        x shape: (Batch_Size, Seq_Len)
        """
        # --- Task 2: Implement SentimentRNN (Layers + Forward) ---
        # 1. Embed x
        # 2. Pass to RNN -> returns (output, hn)
        # 3. Pull out the LAST hidden state (hn[0]) 
        #    Note: This acts as the 'summary' of the entire sequence.
        # 4. Pass results to FC layer
        
        ### TODO: Implement the forward pass (4 lines)

        embedded = self.embedding(x) # (B, T) -> (B, T, E), where E is embed_dim
        # The embedding layer converts the input token indices into dense vectors.
        # The output shape is (Batch_Size, Seq_Len, Embed_Dim).

        output, hn = self.rnn(embedded) # hn shape: (1, B, H), where H is hidden_dim, 1 is num_layers
        # The RNN processes the sequence of embeddings
        # and produces an output for each time step
        # as well as a final hidden state (hn).
        # Since we are using a single-layer RNN, hn has a shape of (1, Batch_Size, Hidden_Dim).

        last_hidden = hn[0]
        # single layer RNN, so we take the first (and only) layer's hidden state: (B, H)
        # This hidden state serves as a summary of the entire sequence,
        # capturing the relevant information needed for classification.

        logits = self.fc(last_hidden) 
        # (B, H) -> (B, output_dim)
        # the linear layer takes the final hidden state (the summary of the sequence)
        # and produces logits for each class (positive/negative).

        # return logits
        return logits
        # now we return the logits, which will be used to compute the loss during training
    
        ### END TODO


# ==========================================
# 3. 训练流程
# ==========================================


def main():
    # 0. 准备工作
    tsv_path = "train.tsv"
    # 我们使用 SST-2 数据集的前 30000 条进行快速教学
    full_data = load_data_from_tsv(tsv_path, limit=30000)

    if not full_data:
        print("❌ 数据加载失败，程序退出。")
        return

    random.seed(42)
    random.shuffle(full_data)
    train_data = full_data[: int(len(full_data) * 0.9)]
    test_data = full_data[int(len(full_data) * 0.9) :]

    tokenizer = SimpleTokenizer(train_data)

    # 1. 实例化模型
    EMBED_DIM = 128
    HIDDEN_DIM = 256

    try:
        model = SentimentRNN(len(tokenizer.word2idx), EMBED_DIM, HIDDEN_DIM, 2)
        print("\n模型结构:", model)
    except Exception as e:
        print(f"\n❌ 模型初始化失败，请检查 Task 2: {e}")
        return

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 2. 训练
    print("\n=== 开始训练 (SST-2 Dataset) ===")
    
    try:
        test_batch = train_data[:8]
        inputs, _ = collate_batch(test_batch, tokenizer)
        output = model(inputs)
        if output is None:
            print("❌ Task 2 (Model Implementation) 未完成，无法开始训练。")
            return
        print("✅ 模型前向传播测试通过！开始 Epoch 循环...")
    except Exception as e:
        print(f"❌ 模型前向传播出错，请检查 Task 2: {e}")
        return

    # 正式训练循环
    BATCH_SIZE = 32
    EPOCHS = 10
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0
        for i in range(0, len(train_data), BATCH_SIZE):
            batch = train_data[i : i + BATCH_SIZE]
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

        print(
            f"Epoch {epoch + 1} | Loss: {total_loss / (len(train_data) // BATCH_SIZE + 1):.4f} | Accuracy: {100 * correct / total:.2f}%"
        )

    print("\n=== 测试模型 ===")
    user_samples = [
        "this movie is a masterpiece",
        "absolute waste of time, horrible acting",
        "it was okay, not great but not bad either"
    ]
    model.eval()
    with torch.no_grad():
        for sample in user_samples:
            ids = torch.tensor([tokenizer.encode(sample)])
            outputs = model(ids)
            pred = outputs.argmax(dim=1).item()
            print(f"输入: {sample}")
            print(f"预测: {'Positive 😄' if pred == 1 else 'Negative 😞'}")


if __name__ == "__main__":
    main()
