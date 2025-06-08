import torch.nn as nn

if __package__:
    from .quantizers.formats import ElemFormat
    from .quantizers.int_quant import INTQuantizer
    from .quantizers.fp_quant import FPQuantizer
    from .quantizers.mx_quant import MXQuantizer
else:
    from quantizers.formats import ElemFormat
    from quantizers.int_quant import INTQuantizer
    from quantizers.fp_quant import FPQuantizer
    from quantizers.mx_quant import MXQuantizer


def create_fmt_ctx(fmt):
    if fmt == "int4":
        return ElemFormat.int4
    elif fmt == "int8":
        return ElemFormat.int8
    elif fmt == "fp4_e2m1":
        return ElemFormat.fp4_e2m1
    elif fmt == "fp8_e4m3":
        return ElemFormat.fp8_e4m3
    elif fmt == "fp8_e5m2":
        return ElemFormat.fp8_e5m2
    elif fmt == "int32":
        return ElemFormat.int32
    else:
        raise RuntimeError(f"Invalid format, got {fmt}")


class FakeQuantizer(nn.Module):
    @staticmethod
    def build(**bit_config):
        quant_type = bit_config.pop("type")
        bit_config["format"] = create_fmt_ctx(bit_config["format"])

        if quant_type == "int":
            return INTQuantizer(**bit_config)
        elif quant_type == "fp":
            return FPQuantizer(**bit_config)
        elif quant_type == "mx":
            return MXQuantizer(**bit_config)
        else:
            raise RuntimeError(f"Unknown Quant type. got {quant_type}")


if __name__ == "__main__":
    bit_config = {
        "type": "mx",
        "format": "fp4_e2m1",
        "group_size": -1,
        "axes": -1,
        "zero_point": False,
    }
    quantizer = FakeQuantizer.build(**bit_config)
    print(quantizer)
