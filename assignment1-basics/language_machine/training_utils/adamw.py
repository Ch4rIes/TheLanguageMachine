import math
from collections.abc import Callable, Iterable
from typing import Optional

import torch


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        """
        eps is to improve numerical stability for extremely small v (second moment)
        weight_decay is the decoupled weight decay coefficient
        """
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                m = state.get("m", 0)
                v = state.get("v", 0)
                t = state.get("t", 1)
                eps = group["eps"]
                weight_decay = group["weight_decay"]
                beta1, beta2 = group["betas"]

                m = beta1 * m + (1 - beta1) * p.grad.data
                v = beta2 * v + (1 - beta2) * (p.grad.data * p.grad.data)
                adjusted_lr = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                # update parameter
                p.data -= adjusted_lr * m / (v.sqrt() + eps)
                # weight decay
                p.data -= lr * weight_decay * p.data

                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
        return loss
