import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F


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

        self.input_quantizer = None
        self.output_quantizer = None
        self.weight_quantizer = None
        self.bias_quantizer = None

        if self.weight_quantizer is not None:
            self.weight = self.weight_quantizer(self.weight)

        if self.bias_quantizer is not None and self.bias is not None:
            self.bias = self.bias_quantizer(self.bias)

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward with quantized weight if available."""
        return self.output_quantizer(F.linear(inputs, self.weight, self.bias))
