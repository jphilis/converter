import torch
from typing import Optional

__all__ = ["LeakyReLU", "ReLU", "Sigmoid"]

class LeakyReLU:
    """Simple standalone LeakyReLU-like activation wrapper."""
    def __init__(self, name: Optional[str] = None, negative_slope: float = 0.01):
        self.name = name
        self.negative_slope = float(negative_slope)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x >= 0, x, x * self.negative_slope)

    def __repr__(self) -> str:
        return f"LeakyReLU(name={self.name!r}, negative_slope={self.negative_slope})"

class ReLU:
    """Simple standalone ReLU-like activation wrapper."""
    def __init__(self, name: Optional[str] = None):
        self.name = name

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)

    def __repr__(self) -> str:
        return f"Relu(name={self.name!r})"

class Sigmoid:
    """Simple standalone Sigmoid activation wrapper."""
    def __init__(self, name: Optional[str] = None):
        self.name = name

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)

    def __repr__(self) -> str:
        return f"Sigmoid(name={self.name!r})"