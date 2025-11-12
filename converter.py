import torch
from my_framework.modules import LeakyReLU, ReLU, Sigmoid

CONVERTER_REGISTER = {}

def add_converter(torch_module: torch.nn.Module):
    def decorator(converter_function):
        CONVERTER_REGISTER[torch_module] = converter_function
        return converter_function
    
    return decorator

@add_converter(torch.nn.ReLU)
def convert_relu(module, name):
    return ReLU(name=name)

@add_converter(torch.nn.LeakyReLU)
def convert_LeakyReLU(module, name):
    return LeakyReLU(name, module.negative_slope) # We need to use module to pass module-specific parameters

@add_converter(torch.nn.Sigmoid)
def convert_sigmoid(module, name):
    return Sigmoid(name=name)

def convert_to_custom_module(module: torch.nn.Module, module_name: str):
    """Convert a PyTorch module into a corresponding custom module."""

    for torch_type, converter in CONVERTER_REGISTER.items():
        if isinstance(module, torch_type):
            return converter(module, module_name)
    raise NotImplementedError(f"No converter registered for module.")    
    # raise ValueError("No converter registered for module.") # For testing purposes