import torch
from my_framework.modules import LeakyReLU, ReLU, Sigmoid

def convert_to_custom_module(module: torch.nn.Module, module_name: str):
    """Convert a PyTorch module into a corresponding custom module."""
    if isinstance(module, torch.nn.LeakyReLU):
        return LeakyReLU(name=module_name)
    if isinstance(module, torch.nn.ReLU):
        return ReLU(name=module_name)
    if isinstance(module, torch.nn.Sigmoid):
        return Sigmoid(name=module_name)
    raise NotImplementedError