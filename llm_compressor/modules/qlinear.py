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
        q_config=None,
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

        self.q_config = q_config.linear
        self.q_type = self.q_config.type  # {input:-, weight:-, output:-, bias:-}
        self.bit_config = (
            self.q_config.bit_config
        )  # {input:-, weight:-, output:-, bias:-}

        self.input_quantizer = FakeQuantizer.build(
            self.q_type.input, **self.bit_config.input
        )
        self.weight_quantizer = FakeQuantizer.build(
            self.q_type.weight, **self.bit_config.weight
        )
        self.bias_quantizer = FakeQuantizer.build(
            self.q_type.bias, **self.bit_config.bias
        )
        self.output_quantizer = FakeQuantizer.build(
            self.q_type.output, **self.bit_config.output
        )

        if self.weight_quantizer is not None:
            self.weight = self.weight_quantizer(self.weight)

        if self.bias_quantizer is not None and self.bias is not None:
            self.bias = self.bias_quantizer(self.bias)

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward with quantized weight if available."""
        return self.output_quantizer(F.linear(inputs, self.weight, self.bias))

    def extra_repr(self):
        s = self.__class__.__name__
        s += f"(in_features={self.in_features}, out_features={self.out_features}, "
        s += f"bias={self.bias is not None})\n"
        s += f"Input Quant: {self.input_quantizer}\n"
        s += f"Weight Quant: {self.weight_quantizer}\n"
        s += f"Bias Quant: {self.bias_quantizer}\n"
        s += f"Output Quant: {self.output_quantizer}"
        return s


if __name__ == "__main__":
    linear = nn.Linear(6, 4)
    qlinear = QLinear(linear=linear)
    print(qlinear)
