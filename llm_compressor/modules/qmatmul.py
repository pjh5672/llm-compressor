from torch import nn
from torch import Tensor


class QMatmul(nn.Module):
    def __init__(
        self,
        q_config=None,
    ):
        super().__init__()
        self.q_config = q_config.matmul
        self.q_type = self.q_config.type
        self.bit_config = self.q_config.bit_config

        self.input_quantizer = FakeQuantizer.build(
            self.q_type.input, **self.bit_config.input
        )
        self.output_quantizer = FakeQuantizer.build(
            self.q_type.output, **self.bit_config.output
        )

    def forward(self, inputs1: Tensor, inputs2: Tensor) -> Tensor:
        """Matrix multiplication with quantized activations if available."""
        return self.output_quantizer(
            self.input_quantizer(inputs1) @ self.input_quantizer(inputs2)
        )

    def extra_repr(self):
        s = self.__class__.__name__
        s += f"Input Quant: {self.input_quantizer}\n"
        s += f"Output Quant: {self.output_quantizer}"
        return s


if __name__ == "__main__":
    qmatmul = QMatmul()
    print(qmatmul)
