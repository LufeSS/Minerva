import torch
import torch.nn as nn

class HighwayResidual(nn.Module):
    """Highway residual connection used between sub-layers.

    Computes a learned gate *per channel* that interpolates between the transform
    output ``h`` and the residual input ``x``.

    Output: ``y = g * h + (1 - g) * x`` where ``g = sigmoid(W_g(h))``.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, hidden_dim, bias=True)
        # Initialize gate bias to favour carry behaviour at the beginning (i.e. g≈0)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -1.0)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Apply highway gating.

        Args:
            x: Residual input of shape ``(batch, seq, dim)``
            h: Transformed output of the same shape.
        Returns:
            Gated sum of ``x`` and ``h``.
        """
        g = torch.sigmoid(self.gate(h))
        return g * h + (1.0 - g) * x 