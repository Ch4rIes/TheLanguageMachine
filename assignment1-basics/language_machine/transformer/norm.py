import torch


class RMSLNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, X) -> torch.Tensor:
        upscaled_X = X.to(torch.float32)
        rms = ((upscaled_X.pow(2)).mean(dim=-1, keepdim=True) + self.eps).sqrt()
        return (upscaled_X / rms * self.weight).to(X.dtype)
