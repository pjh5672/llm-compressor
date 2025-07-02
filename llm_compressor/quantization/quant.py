import torch.nn as nn

if __package__:
    from .quantizers.dummy import DummyQuantizer
    from .quantizers.formats import ElemFormat
    from .quantizers.int_quant import INTQuantizer
    from .quantizers.fp_quant import FPQuantizer
    from .quantizers.mx_quant import MXQuantizer
    from .quantizers.nvfp_quant import NVFPQuantizer
else:
    from quantizers.dummy import DummyQuantizer
    from quantizers.formats import ElemFormat
    from quantizers.int_quant import INTQuantizer
    from quantizers.fp_quant import FPQuantizer
    from quantizers.mx_quant import MXQuantizer
    from quantizers.nvfp_quant import NVFPQuantizer


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
    def build(quant_config, **kwargs):
        quant_config_ = quant_config.copy()
        quant_type = quant_config_.get("type")
        is_profile = quant_config_.get("is_profile")
        op_name = kwargs.get("op_name", None)
        max_limit = kwargs.get("max_val", None)
        save_path = kwargs.get("save_path", "./")

        if quant_type is None:
            return DummyQuantizer(
                is_profile=is_profile,
                op_name=op_name,
                max_limit=max_limit,
                save_path=save_path,
            )

        if quant_type == "int":
            quantizer = INTQuantizer
        elif quant_type == "fp":
            quantizer = FPQuantizer
        elif quant_type == "mx":
            quantizer = MXQuantizer
        elif quant_type == "nvfp":
            quantizer = NVFPQuantizer
        else:
            raise RuntimeError(f"Unknown Quant type. got {quant_type}")

        quant_config_["format"] = create_fmt_ctx(quant_config_["format"])
        quant_config_.update(op_name=op_name, max_limit=max_limit, save_path=save_path)
        return quantizer(**quant_config_)


if __name__ == "__main__":
    quant_config = {
        "type": "nvfp",
        "format": "fp4_e2m1",
        "group_size": -1,
        "axes": -1,
        "zero_point": False,
        "is_profile": False,
    }
    quantizer = FakeQuantizer.build(quant_config)
    print(quantizer)
