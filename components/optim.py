from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # 获取学习率。
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # 获取与 p 关联的状态。
                t = state.get("t", 0) # 从状态中获取迭代次数，或使用初始值。
                grad = p.grad.data # 获取损失对 p 的梯度。
                p.data -= lr / math.sqrt(t + 1) * grad # 原地更新权重张量。
                state["t"] = t + 1 # 增加迭代次数。
        return loss

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon value: {eps}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # 获取学习率。
            weight_decay = group["weight_decay"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # 获取与 p 关联的状态。
                t = state.get("t", 0) + 1 # 从状态中获取迭代次数，或使用初始值。
                
                m=state.get("m",torch.zeros_like(p.data))
                v=state.get("v",torch.zeros_like(p.data))
                grad = p.grad.data # 获取损失对 p 的梯度。
                # 一阶矩估计
                m = beta1*m+(1-beta1)*grad
                # 二阶矩估计
                v = beta2*v+(1-beta2)*torch.pow(grad,2)
                # 存进来
                state["m"]=m
                state["v"]=v

                temp_alpha=lr*(math.sqrt(1-math.pow(beta2,t))/(1-math.pow(beta1,t)))
                p.data -= temp_alpha*(m/(torch.sqrt(v)+eps))
                p.data -= lr*weight_decay*p.data
                state["t"] = t # 增加迭代次数。
        return loss

# 余弦退火
def get_lr_cosine_schedule(t,alpha_max,alpha_min,tw,tc):
    if t<tw:
        alpha = t/tw*alpha_max
    elif t>=tw and t<=tc:
        alpha = alpha_min+0.5*(1+math.cos((t-tw)/(tc-tw)*math.pi))*(alpha_max-alpha_min)
    elif t>tc:
        alpha = alpha_min
    return alpha

# 梯度裁剪 (防止梯度爆炸)
def gradient_clipping(parameters: tuple, max_norm: float):
    grads = [p.grad.data for p in parameters if p.grad is not None]
    if not grads:
        return 0.0
    total_norm = torch.sqrt(sum(g.detach().pow(2).sum() for g in grads))
    if total_norm > max_norm:
        clip_coeff = max_norm / (total_norm + 1e-6)
        for g in grads:
            g.data*=clip_coeff
    return float(total_norm.item())