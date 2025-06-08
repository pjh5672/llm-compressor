import torch
import torch.nn as nn

if __package__:
    from .formats import ElemFormat, _get_format_params
    from .utils import (
        _reshape_to_blocks,
        _undo_reshape_to_blocks,
        _shared_exponents,
        _quantize_elemwise_core,
    )
else:
    from formats import ElemFormat, _get_format_params
    from utils import (
        _reshape_to_blocks,
        _undo_reshape_to_blocks,
        _shared_exponents,
        _quantize_elemwise_core,
    )


class MXQuantizer(nn.Module):
    def __init__(
        self,
        format: ElemFormat,
        group_size=32,
        axes=-1,
        zero_point=False,
        device=torch.device("cpu"),
        **kwargs,
    ):
        """
        group_size:
            - 32: per-group quant.
        axes:
            - -1: row-wise quant.
            - -2: column-wise quant.
        """
        super().__init__()

        assert format in (
            ElemFormat.int4,
            ElemFormat.int8,
            ElemFormat.fp4_e2m1,
            ElemFormat.fp8_e4m3,
            ElemFormat.fp8_e5m2,
        ), f"Not support Format for {self.__class__.__name__}"

        ebits, mbits, emax, max_norm, min_norm = _get_format_params(format)
        self.ebits = torch.tensor(ebits).to(device)
        self.mbits = torch.tensor(mbits).to(device)
        self.emax = torch.tensor(emax).to(device)
        self.max_norm = torch.tensor(max_norm).to(device)
        self.min_norm = torch.tensor(min_norm).to(device)
        self.scale_ebits = kwargs.pop("scale_ebits", 8)
        self.scale_mbits = kwargs.pop("scale_mbits", 0)
        self.str_format = str(format)
        self.configure(zero_point=zero_point, group_size=group_size, axes=axes)
        self.enable()

    def configure(self, zero_point, group_size, axes):
        self.zero_point = zero_point
        self.group_size = group_size
        self.axes = axes

    def find_params(self, x, already_reshaped=False):
        dtype = x.dtype
        if not already_reshaped:
            x, self.shared_axes, *_ = _reshape_to_blocks(
                x,
                block_size=self.group_size,
                axes=self.axes,
            )
        else:
            self.shared_axes = [self.axes + 2]

        if self.zero_point:
            max_val = x.amax(dim=self.axes, keepdim=True)
            min_val = x.amin(dim=self.axes, keepdim=True)
            zeros = (max_val + min_val) / 2
        else:
            zeros = torch.tensor(0)

        # Get shared exponents & shared mantissas(NanoMantissa from Nanoscaling FP:NxFP)
        shared_exp, shared_mts = _shared_exponents(
            (x - zeros),
            method="max",
            axes=[x + 1 for x in self.shared_axes],
            mbits=self.scale_mbits,
        )

        # Offset the max exponent by the largest representable exponent
        # in the element data format
        shared_exp = shared_exp - self.emax

        # Restrict to [-emax, emax] range
        scale_emax = 2 ** (self.scale_ebits - 1) - 1
        shared_exp[shared_exp > scale_emax] = scale_emax + 1
        shared_exp[shared_exp < -scale_emax] = -scale_emax

        scales = (2**shared_exp) * shared_mts
        assert torch.isnan(scales).sum() == 0
        return scales.to(dtype), zeros.to(dtype)

    def forward(self, x, **kwargs):
        scales = kwargs.pop("scales", None)
        zeros = kwargs.pop("zeros", None)

        if self.is_enable:
            x, *meta = _reshape_to_blocks(
                x,
                block_size=self.group_size,
                axes=self.axes,
            )

            if (scales is not None) & (zeros is not None):
                x_dq = self.fake_quantize(x, scales=scales, zeros=zeros)
            else:
                scales, zeros = self.find_params(x, already_reshaped=True)
                x_dq = self.fake_quantize(x, scales=scales, zeros=zeros)

            return _undo_reshape_to_blocks(
                x_dq,
                padded_shape=meta[-2],
                orig_shape=meta[1],
                axes=meta[0],
                block_size=meta[-1],
            )
        return x

    def fake_quantize(self, x, scales, zeros):
        x = (x - zeros) / scales
        q = _quantize_elemwise_core(
            x,
            self.mbits,
            self.ebits,
            self.max_norm,
            round="nearest",
            allow_denorm=True,
            saturate_normals=True,
            custom_cuda=False,
        )
        return q * scales + zeros

    def enable(self):
        self.is_enable = True

    def disable(self):
        self.is_enable = False

    def extra_repr(self):
        s = f"Format: MX{self.str_format.split('.')[-1].upper()}, "
        s += f"Min: {-self.max_norm}, Max: {self.max_norm}"
        return s


if __name__ == "__main__":
    torch.manual_seed(0)

    device = torch.device("cuda")
    x = torch.randn(4, 6).to(device=device)
    print(x)

    quantizer = MXQuantizer(
        format=ElemFormat.int4,
        group_size=6,
        axes=-1,
        zero_point=False,
        device=device,
        scale_ebits=8,
        scale_mbits=0,
    )
    # quantizer = MXQuantizer(
    #     format=ElemFormat.fp8_e4m3, group_size=32, axes=-1, zero_point=False, device=device,
    #     scale_ebits=8, scale_mbits = 1,
    # )
    print(quantizer)
    x_dq = quantizer(x)
    print(x_dq)
    print(((x - x_dq) ** 2).mean())
