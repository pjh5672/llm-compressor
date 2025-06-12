import torch.nn as nn

if __package__:
    from .quantizers.formats import ElemFormat
    from .quantizers.int_quant import INTQuantizer
    from .quantizers.fp_quant import FPQuantizer
    from .quantizers.mx_quant import MXQuantizer
    from .quantizers.dummy import DummyQuantizer
else:
    from quantizers.formats import ElemFormat
    from quantizers.int_quant import INTQuantizer
    from quantizers.fp_quant import FPQuantizer
    from quantizers.mx_quant import MXQuantizer
    from quantizers.dummy import DummyQuantizer


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
    else:
        raise RuntimeError(f"Invalid format, got {fmt}")


class FakeQuantizer(nn.Module):
    @staticmethod
    def build(**quant_config):
        quant_type = quant_config.pop("type")

        if quant_type is None:
            return DummyQuantizer()

        quant_config["format"] = create_fmt_ctx(quant_config["format"])

        if quant_type == "int":
            return INTQuantizer(**quant_config)
        elif quant_type == "fp":
            return FPQuantizer(**quant_config)
        elif quant_type == "mx":
            return MXQuantizer(**quant_config)
        else:
            raise RuntimeError(f"Unknown Quant type. got {quant_type}")


if __name__ == "__main__":
    quant_config = {
        "type": "fp",
        "format": "fp4_e2m1",
        "group_size": -1,
        "axes": -1,
        "zero_point": False,
    }
    # quant_config = {
    #     "type": None,
    #     "format": None,
    #     "group_size": None,
    #     "axes": None,
    #     "zero_point": None,
    # }
    quantizer = FakeQuantizer.build(**quant_config)
    print(quantizer)
