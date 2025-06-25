import torch
import torch.nn as nn

class ReZeroResidual(nn.Module):
    """Trainable residual scaling (ReZero / sequential softmax residual).

    The module keeps a learnable scalar parameter *alpha* initialised to zero.
    Given residual input ``x`` and transform output ``h`` it returns
    ``y = x + alpha * h``.
    """

    def __init__(self):
        super().__init__()
        # scalar parameter; could also be per-channel but scalar is cheaper
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return x + self.alpha * h 