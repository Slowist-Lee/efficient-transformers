import torch
from cs336_basics.optim import SGD


weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
opt = SGD([weights], lr=1e1)

for t in range(10):
    opt.zero_grad() # 重置所有可学习参数的梯度。
    loss = (weights**2).mean() # 计算一个标量损失值。
    print(loss.cpu().item())
    loss.backward() # 运行反向传递，计算梯度。
    opt.step() # 运行优化器步骤。