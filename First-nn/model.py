import torch  
from torch import nn

# Linear regression model
class LinearRegressionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(1), requires_grad=True)

    # Forward pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        breakpoint()
        return x * self.weight + self.bias