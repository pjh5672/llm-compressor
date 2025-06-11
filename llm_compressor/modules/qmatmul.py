import sys
from pathlib import Path
from copy import deepcopy

import torch
from torch import nn
from torch import Tensor

PATH = Path(__file__).resolve().parents[1]
if str(PATH) not in sys.path:
    sys.path.append(str(PATH))

from quantization.quant import FakeQuantizer  # noqa: E402


class QMatmul(nn.Module):
    def __init__(
        self,
        quant_config,
        axes=-1,
    ):
        """Attributes
        axes:
            - -1: row-wise quant.
            - -2: column-wise quant.
        """
        super().__init__()

        # Act. Matmul takes place on Q@K.T & S@V in Attention, for KVQuant
        self.quant_config = quant_config
        self.input1_quantizer = FakeQuantizer.build(**self.quant_config.act_in)
        self.quant_config.act_in2 = deepcopy(self.quant_config.act_in)
        self.quant_config.act_in2["axes"] = axes
        # row-wise setup
        if (axes == -1) and (self.quant_config.act_in["group_size"] == -2):
            self.quant_config.act_in2["group_size"] = -1
        # column-wise setup
        if (axes == -2) and (self.quant_config.act_in["group_size"] == -1):
            self.quant_config.act_in2["group_size"] = -2
        self.input2_quantizer = FakeQuantizer.build(**self.quant_config.act_in2)
        self.output_quantizer = FakeQuantizer.build(**self.quant_config.act_out)

    def forward(self, inputs1: Tensor, inputs2: Tensor) -> Tensor:
        """Matrix multiplication with quantized activations if available."""
        return self.output_quantizer(
            torch.matmul(self.input1_quantizer(inputs1), self.input2_quantizer(inputs2))
        )


if __name__ == "__main__":
    import torch
    from easydict import EasyDict

    device = torch.device("cuda:0")
    quant_config = EasyDict({})
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

    qmatmul = QMatmul(quant_config=quant_config, axes=-2)
    print(qmatmul)
    x1 = torch.randn(4, 6).to(device)
    x2 = torch.randn(6, 4).to(device)
    y = qmatmul(x1, x2)
