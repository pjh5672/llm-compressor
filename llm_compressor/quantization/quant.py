import torch.nn as nn

from quantizers.int_quant import INTQuantizer
from quantizers.fp_quant import FPQuantizer
from quantizers.mx_quant import MXQuantizer


class FakeQuantizer(nn.Module):
    @staticmethod
    def build(quant_type, **kwargs):
        if quant_type == "int":
            return INTQuantizer(**kwargs)
        
        elif quant_type == "fp":
            return FPQuantizer(**kwargs)
        
        elif quant_type == "mx":
            return MXQuantizer(**kwargs)


if __name__ == "__main__":
    from quantizers.formats import ElemFormat
    q_config = {"fmt": ElemFormat.int4, "group_size": -1, "axes": -1, "zero_point": False}
    quantizer = FakeQuantizer.build(quant_type="int", **q_config)
    print(quantizer)
    quantizer = FakeQuantizer.build(quant_type="mx", **q_config)
    print(quantizer)
    q_config = {"fmt": ElemFormat.fp4_e2m1, "group_size": -1, "axes": -1, "zero_point": False}
    quantizer = FakeQuantizer.build(quant_type="fp", **q_config)
    print(quantizer)
