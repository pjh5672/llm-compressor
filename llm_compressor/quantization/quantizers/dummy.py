import torch
import torch.nn as nn

if __package__:
    from .base import BaseQuantizer
else:
    from base import BaseQuantizer


class DummyQuantizer(nn.Module, BaseQuantizer):
    def __init__(
        self,
        format=None,
        group_size=None,
        axes=None,
        zero_point=False,
        device=torch.device("cpu"),
        **kwargs,
    ):
        """
        Dummy Quantizer for passing original tensor
        """
        super().__init__()

    def configure(self):
        pass

    def find_params(self):
        pass

    def forward(self, x, **kwargs):
        return x

    def fake_quantize(self):
        pass

    def extra_repr(self):
        s = "Disabled"
        return s


if __name__ == "__main__":
    torch.manual_seed(0)

    device = torch.device("cuda")
    x = torch.randn(4, 6).to(device=device)
    print(x)

    quantizer = DummyQuantizer()
    print(quantizer)
    x_dq = quantizer(x)
    print(x_dq)
    print(((x - x_dq) ** 2).mean())
