import torch
from my_framework.modules import ReLU
from converter import convert_to_custom_module

def test_relu_conversion():
    torch_module = torch.nn.ReLU()
    custom_module = convert_to_custom_module(torch_module, "relu_test")

    assert isinstance(custom_module, ReLU)
    assert custom_module.name == "relu_test"
    

def test_unregistered_module():
    class Dummy(torch.nn.Module): pass

    try:
        custom_module = convert_to_custom_module(Dummy(), "dummy")
    except NotImplementedError as e:
        assert str(e) == "No converter registered for module."
    except Exception as e:
        assert False, f"Unexpected exception type: {type(e)}"

def main():
    test_relu_conversion()
    test_unregistered_module()
    print("All tests passed.")

if __name__ == "__main__":
    main()