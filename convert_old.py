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

'''
SOLID
S: Single Responsibility Principle - Is broken since the function both handles conversion logic and type checking.
O: Open/Closed Principle - Is broken because adding new module types requires modifying the existing function.
L: Liskov Substitution Principle - Is maintained as the function can accept any subclass of torch.nn.Module.
I: Interface Segregation Principle - Is not directly applicable here as there are no interfaces being defined.
D: Dependency Inversion Principle - Is broken since the function depends directly on concrete implementations of torch.nn.Module subclasses.
'''
