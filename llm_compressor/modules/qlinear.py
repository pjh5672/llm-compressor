import sys
from pathlib import Path
from copy import deepcopy

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
        quant_config,
        dtype,
    ):
        super().__init__(
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            linear.weight.device,
            dtype,
        )
        self.train(linear.training)

        with torch.no_grad():
            self.weight.copy_(linear.weight)
            if self.bias is not None:
                self.bias.copy_(linear.bias)

        self.quant_config = quant_config
        self.input_quantizer = FakeQuantizer.build(**self.quant_config.act_in)
        self.weight_quantizer = FakeQuantizer.build(**self.quant_config.weight)
        self.output_quantizer = FakeQuantizer.build(**self.quant_config.act_out)

        if self.bias is not None:
            self.quant_config.bias = deepcopy(self.quant_config.weight)
            self.quant_config.bias.update({"device": torch.device("cpu")})
            self.bias_quantizer = FakeQuantizer.build(**self.quant_config.bias)
            self.bias.data = self.bias_quantizer(self.bias.data)
            del self.bias_quantizer
            torch.cuda.empty_cache()

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward with quantized weight if available."""
        return self.output_quantizer(
            F.linear(self.input_quantizer(inputs), self.weight, self.bias)
        )

    def extra_repr(self):
        s = f"(in_features={self.in_features}, "
        s += f"out_features={self.out_features}, "
        s += f"bias={self.bias is not None})"
        return s


if __name__ == "__main__":
    import torch
    from easydict import EasyDict

    device = torch.device("cuda:0")
    quant_config = EasyDict({})
    quant_config.weight = {
        "type": "int",
        "format": "int4",
        "group_size": -1,
        "axes": -1,
        "zero_point": False,
        "device": device,
    }
    quant_config.act_in = {
        "type": "int",
        "format": "int8",
        "group_size": -1,
        "axes": -1,
        "zero_point": False,
        "device": device,
    }
    quant_config.act_out = {
        "type": "int",
        "format": "int8",
        "group_size": -1,
        "axes": -1,
        "zero_point": False,
        "device": device,
    }
    linear = nn.Linear(6, 4, bias=False).to(device)
    qlinear = QLinear(
        linear=linear, quant_config=quant_config, dtype=linear.weight.dtype
    )
    print(qlinear)
    x = torch.randn(4, 6).to(device)
    qlinear.weight.data = qlinear.weight_quantizer(qlinear.weight.data)
    y = qlinear(x)
    print(y)
