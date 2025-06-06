import torch.nn as nn

if __package__:
    from .quantizers.int_quant import INTQuantizer
    from .quantizers.fp_quant import FPQuantizer
    from .quantizers.mx_quant import MXQuantizer
else:
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
        
        else:
            raise RuntimeError(f"Unknown Quant type. got {quant_type}")


if __name__ == "__main__":
    from quantizers.formats import ElemFormat
    bit_config = {"fmt": ElemFormat.int4, "group_size": -1, "axes": -1, "zero_point": False}
    quantizer = FakeQuantizer.build(quant_type="int", **bit_config)
    print(quantizer)
    quantizer = FakeQuantizer.build(quant_type="mx", **bit_config)
    print(quantizer)
    bit_config = {"fmt": ElemFormat.fp4_e2m1, "group_size": -1, "axes": -1, "zero_point": False}
    quantizer = FakeQuantizer.build(quant_type="fp", **bit_config)
    print(quantizer)
