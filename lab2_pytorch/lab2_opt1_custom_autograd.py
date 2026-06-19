import torch
from torch.autograd import Function

class SwishActivation(Function):
    """
    可选任务 1: 自定义 Autograd 函数
    
    任务描述：
    PyTorch 能够自动计算绝大多数操作的梯度，但有时出于性能或数值稳定性的考虑，
    我们需要自定义算子的前向传播 (forward) 和反向传播 (backward)。
    
    请继承 `torch.autograd.Function` 来实现 Swish 激活函数：
    Swish 公式: f(x) = x * sigmoid(x)
    """
    
    @staticmethod
    def forward(ctx, i):
        """
        前向传播
        参数:
        ctx: 上下文对象，用于在前向和反向传播之间存储信息
        i: 输入张量 x
        """
        
        ### TODO: Write your code below
        
        # 计算 Swish 激活函数: f(x) = x * sigmoid(x)
        # sigmoid: sigmoid(x) = 1 / (1 + exp(-x))
        sigmoid_i = torch.sigmoid(i)
        # Swish: f(x) = x * sigmoid(x)
        swish_i = i * sigmoid_i
        
        # 将反向传播需要的变量保存到 ctx 中
        # ctx = context, used to save information for backward pass
        ctx.save_for_backward(sigmoid_i, swish_i)

        ### END TODO
        
        return swish_i

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播
        参数:
        ctx: 包含 forward 中保存的信息的上下文对象
        grad_output: 上一层传回来的梯度 (dL/df)
        返回值: dL/dx
        """
        # 提示: 使用 ctx.saved_tensors 获取保存的变量
        
        ### TODO: Write your code below

        # acquire saved tensors from forward pass
        sigmoid_i, swish_i = ctx.saved_tensors
        
        # 计算 Swish 的导数 f'(x)
        # f(x) = x * sigmoid(x)
        # sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
        # f'(x) = sigmoid(x) + x * sigmoid'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        # grad_swish = f'(x)
        grad_swish = sigmoid_i + swish_i * (1 - sigmoid_i)
        
        # 链式法则: dL/dx = (dL/df) * f'(x)

        # L is the loss, f is the output of the Swish function, x is the input
        # we want to find dL/dx, we have dL/df (grad_output) and f'(x) (grad_swish)
        # we need to multiply them together to obtain dL/dx, by using the chain rule
        grad_input = grad_output * grad_swish

        ### END TODO
        
        return grad_input


def test_custom_autograd():
    print("--- 测试自定义 Autograd 函数 (Swish) ---")
    
    # 为了使用 gradcheck，输入需要是双精度浮点数 (float64) 并且 requires_grad=True
    # requires_grad=True 是为了告诉 PyTorch 需要计算这个张量的梯度
    x = torch.randn(5, 5, dtype=torch.float64, requires_grad=True)
    
    try:
        # 实例化自定义的 Swish 函数
        swish = SwishActivation.apply
        
        # `torch.autograd.gradcheck` 会用数值微分（微小的有限差分）来验证你的解析梯度是否正确！
        print("\n正在运行 PyTorch gradcheck 验证你的反向传播公式...")
        test_passed = torch.autograd.gradcheck(swish, (x,), eps=1e-6, atol=1e-4)
        
        if test_passed:
            print("✅ 挑战成功！你的的前向和反向传播实现完全正确，与数值微分结果一致。")
    except Exception as e:
        print(f"❌ 挑战失败，计算图抛出异常或梯度不匹配:\n{e}")

if __name__ == "__main__":
    test_custom_autograd()
