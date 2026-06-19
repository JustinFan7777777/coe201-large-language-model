import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def train_mnist_mlp():
    print("--- 训练 3 层 MLP 进行 MNIST 数字分类 (使用 nn.Sequential) ---")
    
    # 1. 准备数据集
    # 使用 torchvision 下载并加载 MNIST 数据集
    # transforms.ToTensor() 将像素值从 [0, 255] 转换到 [0.0, 1.0]，并在后面增加一个通道维
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # 标准化
    ])
    
    print("正在加载数据集...")
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    print(f"训练集大小: {len(train_dataset)}, 测试集大小: {len(test_dataset)}")

    # 2. 搭建 3 层 MLP 模型
    # MNIST 图片大小为 28x28，展开后为 784。输出为 10（0-9 的数字分类）
    # 隐藏层大小可自由设定，例如：128, 64
    
    ### TODO: 使用 nn.Sequential 构建一个 3 层的全连接神经网络
    # 提示: 需要使用 nn.Flatten() 将输入的 (Batch, 1, 28, 28) 图片展平为 (Batch, 784)
    # 然后依次添加 nn.Linear 和 nn.ReLU (最后一层不需要 ReLU)

    model = nn.Sequential(
        # inputs: (Batch, 1, 28, 28), 1 means grayscale channel
        # outputs: (Batch, 784)
        nn.Flatten(),

        # nn.Linear(in_features, out_features): fully connected layer
        # in_features: number of input features (flattened image size)
        # out_features: number of neurons in the hidden layer
        # Linear: execute maatrix multiplication and add bias: y = xW^T + b
        # W: (128, 784), b: (128,)
        nn.Linear(784, 128),
        
        # activation function: ReLU (Rectified Linear Unit)
        # ReLU(x) = max(0, x), introduces non-linearity to the model
        nn.ReLU(),

        # the second hidden layer
        # in_features: 128 (from previous layer)
        # out_features: 64 (number of neurons in the second hidden layer)
        # W: (64, 128), b: (64,)
        nn.Linear(128, 64),

        # another ReLU activation
        nn.ReLU(),

        # output layer:
        # in_features: 64 (from previous layer)
        # out_features: 10 (number of classes for MNIST)
        nn.Linear(64, 10)
    )

    ### END TODO
    
    print(f"模型结构:\n{model}")

    # 3. 定义损失函数与优化器
    ### TODO: 定义交叉熵损失函数和优化器 (例如 SGD，学习率为 0.01)

    # CrossEntropyLoss: softmax + negative log likehood loss, for multi-class classification
    # this is why the output layer does not need to have an activation function
    criterion = nn.CrossEntropyLoss()

    # SGD: Stochastic Gradient Descent optimizer, updates model parameters based on computed gradients
    # model.parameters(): returns all parameters (weights and biases) of the model that require gradients
    # lr: learning rate = step size for updating parameters
    # if too large, may overshoot the optimal parameters
    # if too small, training will be slow
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    ### END TODO

    epochs = 3
    print(f"\n开始训练, 总轮数: {epochs}")
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # 将数据展平 (由模型中的 nn.Flatten 负责，或者你可以在这里 view(-1, 28*28))
            
            # 步骤 1: 将优化器中的梯度清零
            ### TODO (1行)

            # clear to 0 before every backward pass 
            # otherwise gradients will accumulate across batches
            optimizer.zero_grad()

            ### END TODO
            
            # 步骤 2: 前向传播
            ### TODO (1行)
            
            # forward pass: input (data) -> output (predicted values)
            output = model(data)

            ### END TODO
            
            # 步骤 3: 计算损失
            ### TODO (1行)
            
            # criterion: computes the loss using "CrossEntropyLoss"
            # output: (Batch, 10) raw scores (predicted logits) for each class
            # target: (Batch,) true class labels (0-9)
            # loss: a single scalar value representing the average loss for the batch
            loss = criterion(output, target)

            ### END TODO
            
            # 步骤 4: 反向传播计算梯度
            ### TODO (1行)
        
            # backpropagation: computes the gradients of loss with respect to all model parameters (weights and biases)
            # mechanism: traverses the computational graph from the output back to the input, applying the chain rule to compute gradients
            # the computed gradients are stored in the ".grad" attribute of each parameter temsor
            loss.backward()

            ### END TODO
            
            # 步骤 5: 更新模型参数
            ### TODO (1行)
    
            # updates the model parameters based on the computed gradients
            # param' = param - learning_rate * param.grad
            optimizer.step()

            ### END TODO
            
            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                      f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        
        # 测试模型性能
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad(): # 测试时不需要计算梯度
            for data, target in test_loader:
                output = model(data)
                test_loss += criterion(output, target).item()  # 累加 batch loss
                pred = output.argmax(dim=1, keepdim=True)  # 获取最大概率的索引
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader)
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
              f'({accuracy:.2f}%)\n')

if __name__ == "__main__":
    train_mnist_mlp()
