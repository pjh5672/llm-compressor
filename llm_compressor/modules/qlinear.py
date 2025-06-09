import sys
from pathlib import Path

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

PATH = Path(__file__).resolve().parents[1]
if str(PATH) not in sys.path:
    sys.path.append(str(PATH))

from quantization.quant import FakeQuantizer  # noqa: E402


class QLinear(nn.Linear):
    def __init__(
        self,
        linear: nn.Linear,
        bit_config,
    ):
        super().__init__(
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            linear.weight.device,
            linear.weight.dtype,
        )
        self.train(linear.training)

        with torch.no_grad():
            self.weight.copy_(linear.weight)
            if self.bias is not None:
                self.bias.copy_(linear.bias)

        self.bit_config = bit_config
        self.input_quantizer = FakeQuantizer.build(**self.bit_config.act_in)
        self.weight_quantizer = FakeQuantizer.build(**self.bit_config.weight)
        self.bias_quantizer = FakeQuantizer.build(**self.bit_config.weight)
        self.output_quantizer = FakeQuantizer.build(**self.bit_config.act_out)

        if self.bias_quantizer is not None and self.bias is not None:
            self.bias.data = self.bias_quantizer(self.bias.data)

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward with quantized weight if available."""
        print(qlinear.weight.data)
        return self.output_quantizer(
            F.linear(self.input_quantizer(inputs), self.weight, self.bias)
        )

    def extra_repr(self):
        s = f"in_features={self.in_features}, "
        s += f"out_features={self.out_features}, "
        s += f"bias={self.bias is not None}\n"
        s += f"axes={self.bit_config.act_in['axes']}(input), "
        s += f"axes={self.bit_config.weight['axes']}(weight), "
        s += f"axes={self.bit_config.act_out['axes']}(output)"
        return s


if __name__ == "__main__":
    from easydict import EasyDict

    bit_config = EasyDict({})
    bit_config.weight = {
        "type": "int",
        "format": "int4",
        "group_size": -1,
        "axes": -1,
        "zero_point": False,
    }
    bit_config.act_in = {
        "type": "int",
        "format": "int8",
        "group_size": -1,
        "axes": -1,
        "zero_point": False,
    }
    bit_config.act_out = {
        "type": "int",
        "format": "int8",
        "group_size": -1,
        "axes": -1,
        "zero_point": False,
    }
    linear = nn.Linear(6, 4, bias=False)
    qlinear = QLinear(linear=linear, bit_config=bit_config)
    print(qlinear)
    qlinear.weight.data = qlinear.weight_quantizer(qlinear.weight.data)
    x = torch.randn(4, 6)
    y = qlinear(x)
    print(y)
