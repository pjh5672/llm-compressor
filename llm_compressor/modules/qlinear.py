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
        quant_config,
        dtype,
        **kwargs,
    ):
        super().__init__(
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            linear.weight.device,
            dtype,
        )
        op_name = kwargs.get("op_name", None)
        save_path = kwargs.get("save_path", "./")

        self.train(linear.training)

        with torch.no_grad():
            self.weight.copy_(linear.weight)
            if self.bias is not None:
                self.bias.copy_(linear.bias)

        self.input_quantizer = FakeQuantizer.build(
            quant_config.act_in,
            op_name=f"{op_name}.input",
            save_path=save_path,
        )
        self.weight_quantizer = FakeQuantizer.build(
            quant_config.weight,
            op_name=f"{op_name}.weight",
            save_path=save_path,
        )
        self.output_quantizer = FakeQuantizer.build(
            quant_config.act_out,
            op_name=f"{op_name}.output",
            save_path=save_path,
        )

    def forward(self, inputs: Tensor, **kwargs) -> Tensor:
        """Forward with quantized weight if available."""
        R1 = kwargs.get("R1", None)
        if R1 is not None:
            dtype = self.weight.dtype
            transpose = kwargs.get("transpose", False)
            if not transpose:
                weight = self.weight.to(torch.float64) @ R1.to(torch.float64)
            else:
                weight = R1.T.to(torch.float64) @ self.weight.to(torch.float64)

            R2 = kwargs.get("R2", None)
            if R2 is not None:
                had_dim = R2.shape[0]
                if transpose:
                    W_ = weight
                    init_shape = W_.shape
                    temp = W_.reshape(-1, init_shape[-1] // had_dim, had_dim)
                    temp = temp.to(torch.float64) @ R2.to(torch.float64)
                    weight = temp.reshape(init_shape)
                else:
                    W_ = weight.t()
                    transposed_shape = W_.shape
                    temp = W_.reshape(-1, transposed_shape[-1] // had_dim, had_dim)
                    temp = temp.to(torch.float64) @ R2.to(torch.float64)
                    weight = temp.reshape(transposed_shape).t()

            self.weight.data = self.weight_quantizer(weight.data.to(dtype))

        return self.output_quantizer(
            F.linear(self.input_quantizer(inputs.to(self.weight)), 
                     self.weight, self.bias)
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
        "is_profile": True,
    }
    quant_config.act_in = {
        "type": "int",
        "format": "int8",
        "group_size": -1,
        "axes": -1,
        "zero_point": False,
        "is_profile": True,
    }
    quant_config.act_out = {
        "type": "int",
        "format": "int8",
        "group_size": -1,
        "axes": -1,
        "zero_point": False,
        "is_profile": True,
    }
    linear = nn.Linear(6, 4, bias=False)
    qlinear = QLinear(
        linear=linear, quant_config=quant_config, dtype=linear.weight.dtype
    ).to(device)
    print(qlinear)
    x = torch.randn(4, 6).to(device)
    qlinear.weight.data = qlinear.weight_quantizer(qlinear.weight.data)
    y = qlinear(x)
    print(y)
