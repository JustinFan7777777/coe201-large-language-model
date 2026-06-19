import torch
import math
import matplotlib.pyplot as plt

def poly_regression():
    print("--- 线性多项式拟合 与 优化器对比 ---")
    
    # 1. 准备数据: y = sin(x) 带有少量噪声，区间 [-pi, pi]
    x = torch.linspace(-math.pi, math.pi, 2000).unsqueeze(-1)
    # noise: average 0, std(standard deviation) 0.1
    y_true = torch.sin(x) + torch.randn(2000, 1) * 0.1
    
    # 手动实现不同的优化算法 (GD, SGD, 或者 Mini-Batch SGD)。
    
    print(f"\n开始训练...")
    
    # 2. 初始化参数 
    # 你可以自己决定多项式的最高次数，例如使用 3 次多项式 a + b*x + c*x^2 + d*x^3
    ### TODO: 初始化多项式参数 (如 a, b, c, d), 记得设置 requires_grad=True

    # 1 means the bias term (constant term), 2 means the linear term
    # 3 means the quadratic term, and 4 means the cubic term.
    # requires_grad=ture: allows PyTorch to automatically compute gradients
    # for these parameters during backpropagation
    a = torch.randn(1, requires_grad=True)
    b = torch.randn(1, requires_grad=True)
    c = torch.randn(1, requires_grad=True)
    d = torch.randn(1, requires_grad=True)
    
    ### END TODO
    
    # 提示：SGD 学习率如果和 GD 一样大会导致梯度爆炸，可能需要动态调整 learning_rate 或 epochs
    # learning_rate should be relatively small to ensure stable training
    # especially for higher-degree polynomials
    # epochs should be large enough to allow convergence
    # but not too large to cause overfitting or excessive training time
    # we can adjust these hyperparameters(learning_rate, epochs and batch_size) 
    # based on the observed training process, such as the loss curve and the fit of the model to the data
    learning_rate = 1e-4
    epochs = 5000

    # if batch_size = 1, then is SGD
    # if batch_size = len(x), then is GD
    # faced with a trade-off between the stability of GD and the speed of SGD,
    # we use mini-batch SGD, which is a compromise between the two
    batch_size = 64
    
    losses = []
    for epoch in range(epochs):
        # len(x): the total number of samples in the dataset
        # indices: a random permutation of the indices of the dataset, used to shuffle the data
        indices = torch.randperm(len(x))
        # x_shuffled: the input data x shuffled according to the random indices
        x_shuffled = x[indices]
        # a randomly shuffled version of the true labels y_true, corresponding to the shuffled input data
        y_shuffled = y_true[indices]
        
        epoch_loss = 0.0
        
        # --- 核心训练逻辑 ---
        # 提示：你需要自己决定如何遍历数据 (一次送入全部、一次送入一个、还是一次送入一批)
        # 在这里执行 前向传播 -> 计算Loss -> 反向传播 -> 更新权重。
        
        ### TODO: 实现数据遍历与更新逻辑
        
        # use mini-batch SGD to train the model
        # each epoch, we shuffle the data and then iterate through it in batches
        # the range of start is: (0, batch_size, 2*batch_size, ..., len(x_suffled)-batch_size)
        # we divide the dataset into batches of size batch_size, and for each batch we perform the forward pass
        for start in range(0, len(x_shuffled), batch_size):
            end = start + batch_size
            # x_batch: a subset of the input data for the current batch, used for training
            # y_batch: the corresponding true labels for the current batch, used for calculating the loss
            x_batch = x_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            # forward pass: triple polynomial prediction
            y_pred = a + b * x_batch + c * x_batch**2 + d * x_batch**3

            # loss: mean squared error between the predicted values and the true labels
            loss = ((y_pred - y_batch) ** 2).mean()

            # backward pass: 
            # firstly we need to clear the gradients of the parameters
            # to prevent accumulation of gradients from existing gradients
            if a.grad is not None:
                a.grad.zero_()
                b.grad.zero_()
                c.grad.zero_()
                d.grad.zero_()

            # automatic differentiation: deployed in PyTorch already
            # computes the gradients of the loss with respect to the parameters a, b, c, d
            loss.backward()

            # no_grad: a context manager that temporarily sets all the requires_grad flags to false,
            # which means that the operations performed within this block will not be tracked for gradient computation
            # this is necessary when we want to update the parameters manually without tracking these updates in the computational graph
            with torch.no_grad():
                a -= learning_rate * a.grad
                b -= learning_rate * b.grad
                c -= learning_rate * c.grad
                d -= learning_rate * d.grad

            # loss.item(): returns the value of the loss
            # we multiply it by the number of samples in the batch, to get the total loss for the batch
            # we accumulate this total loss for each batch into epoch_loss, which will later be averaged over the entire dataset
            epoch_loss += loss.item() * len(x_batch)

        ### END TODO
        
        # 记录每个 Epoch 的平均 Loss
        if epoch_loss > 0:
            # epoch_loss is the total loss for the epoch, which we accumulated from each batch
            # we divide it by the total number of samples in the dataset (len(x)) to get the average loss for the epoch
            epoch_loss /= len(x)
            losses.append(epoch_loss)
            if epoch % max(1, epochs // 5) == 0:
                print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")
            
    # 绘图展示拟合结果
    try:
        with torch.no_grad():
            ### TODO: 用训练好的参数代入全集 x 求解 y_final_pred 绘图

            y_final_pred = a + b * x + c * x**2 + d * x**3

            ### END TODO
            
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Loss Curve')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.scatter(x.numpy(), y_true.numpy(), s=1, alpha=0.3, label='Data')
        if y_final_pred is not None:
            plt.plot(x.numpy(), y_final_pred.numpy(), color='red', linewidth=2, label=f'Fit')
        plt.title(f"Polynomial Fit")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig("poly_regression_result.png")
        print("\n=> 训练结果已保存为 poly_regression_result.png! 如果右侧只有点没有红线，请检查是否完成功能")
    except:
        pass

if __name__ == '__main__':
    poly_regression()
