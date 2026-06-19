import torch

def manual_gradient_descent():
    """
    任务：针对线性方程 y = wx + b, 使用手动梯度下降法去拟合数据。
    """
    print("--- 任务：手动梯度下降 (拟合 y = 2x + 1) ---")
    
    # 1. 准备合成训练数据
    # 真实的模型是 y = 2x + 1
    # 我们生成一些 x，并计算正确的 y。这就是模型的学习目标。
    x_train = torch.randn(100, 1)  # 100 个样本
    y_true = 2 * x_train + 1 + torch.randn(100, 1) * 0.1 # 加入少量的噪声
    
    # 2. 初始化参数 (模型权重)
    # 待学习的参数是 w 和 b。它们需要追踪梯度 (requires_grad=True)
    # 随机初始化
    w = torch.randn(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    print(f"初始化的权重: w={w.item():.4f}, b={b.item():.4f}")
    
    # 3. 设置超参数
    learning_rate = 0.1
    epochs = 40
    
    # 4. 开始训练循环
    for epoch in range(epochs):
        # 步骤 1: 前向传播 (Forward pass)
        # 根据当前的 w 和 b 计算预测值 y_pred
        
        ### TODO: Write your code below (1 line)
        
        # x_train: (100, 1)
        # w: (1,); b: (1,)
        # y_pred: (100, 1) = (100, 1) * (1,) + (1,)
        y_pred = x_train * w + b

        ### END TODO
        
        # 步骤 2: 计算损失 (Loss)
        # 使用均方误差 (Mean Squared Error): (y_pred - y_true) 的平方的平均值
        
        ### TODO: Write your code below (1 line)

        # y_pred: (100, 1); y_true: (100, 1)
        # **2: element-wise square
        # mean: calculates the average to get a single scalar loss value
        loss = torch.mean((y_pred - y_true) ** 2)

        ### END TODO
        
        # 步骤 3: 反向传播 (Backward pass)
        # PyTorch 会自动计算 loss 针对所有 requires_grad=True 的张量的梯度
        
        ### TODO: Write your code below (1 line)

        # backpropagatetion: computes the gradients of loss with respect to w and b
        # w.grad: will hold the gradient of loss with respect to w
        # b.grad: will hold the gradient of loss with respect to b
        loss.backward()

        ### END TODO
        
        # 步骤 4: 手动更新权重 (不使用 optimizer)
        # 公式: w = w - learning_rate * w.grad
        # 关键: 在更新权重时，不能让 PyTorch 追踪这次计算。应该使用 torch.no_grad()
        
        ### TODO: Write your code below

        # torch.no_grad(): tells Pytorch not to track gradients for the operations inside this block
        with torch.no_grad():
            # learning_rate: step size for the update
            w -= learning_rate * w.grad
            b -= learning_rate * b.grad

        ### END TODO
        
        # 步骤 5: 将梯度清零
        # 为什么需要清零？因为 PyTorch 默认会累加梯度。如果不清零，下一轮计算的反向传播会混入这一轮的梯度。
        
        ### TODO: Write your code below

        # clear to 0 otherwise gradients will accumulate across epochs
        # "_" means in-place operation: modifies the tensor without creating a new one
        w.grad.zero_()
        b.grad.zero_()

        ### END TODO
        
        # 打印进度
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:2d} | Loss: {loss.item():.4f} | w: {w.item():.4f}, b: {b.item():.4f}")
            
    # 验证最终结果
    print(f"\n训练结束, 最终预测值: w={w.item():.4f}, b={b.item():.4f} (真实值为 w=2.0000, b=1.0000)")
    if abs(w.item() - 2.0) < 0.15 and abs(b.item() - 1.0) < 0.15:
         print("✅ 手动梯度下降测试通过！")
    else:
         print("❌ 测试失败，请检查逻辑。")

if __name__ == "__main__":
    manual_gradient_descent()
