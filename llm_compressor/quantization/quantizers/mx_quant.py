import torch
import torch.nn as nn

if __package__:
    from .formats import ElemFormat, FP32_MIN_NORMAL, _get_format_params
    from .utils import (
        _reshape_to_blocks,
        _undo_reshape_to_blocks,
        _quantize_elemwise_core,
    )
else:
    from formats import ElemFormat, FP32_MIN_NORMAL, _get_format_params
    from utils import (
        _reshape_to_blocks,
        _undo_reshape_to_blocks,
        _quantize_elemwise_core,
    )


class MXQuantizer(nn.Module):
    def __init__(
        self,
        format: ElemFormat,
        group_size=32,
        axes=-1,
        zero_point=False,
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
        self.register_buffer("ebits", torch.tensor(ebits))
        self.register_buffer("mbits", torch.tensor(mbits))
        self.register_buffer("emax", torch.tensor(emax))
        self.register_buffer("max_norm", torch.tensor(max_norm))
        self.register_buffer("min_norm", torch.tensor(min_norm))
        self.scale_ebits = kwargs.get("scale_ebits", 8)
        self.scale_emax = 2 ** (self.scale_ebits - 1) - 1
        self.str_format = str(format)
        self.mse = False
        self.configure(zero_point=zero_point, group_size=group_size, axes=axes)

    def configure(self, zero_point, group_size, axes):
        self.zero_point = zero_point
        self.group_size = group_size
        self.axes = axes

    def find_params(self, x, already_reshaped=False):
        if not already_reshaped:
            x, self.shared_axes, *_ = _reshape_to_blocks(
                x,
                block_size=self.group_size,
                axes=self.axes,
            )
        else:
            if isinstance(self.axes, int):
                axes = [self.axes]
            axes = [(i + len(x.shape) - 1 if i < 0 else i) for i in axes]
            assert all(x >= 0 for x in axes)
            self.shared_axes = sorted(axes)

        def _get_scales(max_val):
            # Get shared exponents & shared mantissas
            shared_mts = torch.log2(
                max_val + FP32_MIN_NORMAL * (max_val == 0).type(max_val.dtype)
            )
            shared_exp = torch.floor(shared_mts)
            # Offset the max exponent by the largest representable exponent
            # in the element data format
            shared_exp -= self.emax

            # Restrict to [-emax, emax] range
            shared_exp[shared_exp > self.scale_emax] = self.scale_emax + 1
            shared_exp[shared_exp < -self.scale_emax] = -self.scale_emax
            return 2**shared_exp

        if self.zero_point:
            max_val = x.amax(dim=self.axes, keepdim=True)
            min_val = x.amin(dim=self.axes, keepdim=True)
            zeros = (max_val + min_val) / 2
            scales = _get_scales(max_val - zeros)
        else:
            max_val = x.abs().amax(dim=self.axes, keepdim=True)
            min_val = -max_val
            zeros = torch.zeros_like(max_val)
            scales = _get_scales(max_val)

        def _clip_range(x, norm=2.4, grid=100, maxshrink=0.8):
            nonlocal scales, zeros
            dtype = x.dtype
            if self.group_size != 0:
                best = torch.full(
                    [x.shape[0], x.shape[1]],
                    float("inf"),
                    dtype=dtype,
                    device=x.device,
                )
            else:
                best = torch.tensor([float("inf")], dtype=dtype, device=x.device)

            for i in range(int(maxshrink * grid)):
                p = 1 - i / grid
                max_val1 = p * max_val
                min_val1 = p * min_val

                if self.zero_point:
                    zeros1 = (max_val1 + min_val1) / 2
                    scales1 = _get_scales(max_val1 - zeros1)
                else:
                    zeros1 = zeros
                    scales1 = _get_scales(max_val1)

                dq = self.fake_quantize(x, scales=scales1, zeros=zeros1)
                dq -= x
                dq.abs_()
                dq.pow_(norm)
                if self.group_size != 0:
                    err = torch.sum(dq, dim=-1, dtype=dtype)
                    tmp = err < best
                    if torch.any(tmp):
                        best[tmp] = err[tmp]
                        tmp.unsqueeze_(-1)
                        scales[tmp] = scales1[tmp]
                        zeros[tmp] = zeros1[tmp]
                else:
                    err = torch.sum(dq, dtype=dtype)
                    tmp = err < best
                    if torch.any(tmp):
                        tmp.squeeze_(0)
                        best[tmp] = err[tmp]
                        scales[tmp] = scales1[tmp]
                        zeros[tmp] = zeros1[tmp]

        if self.mse:
            _clip_range(x)

        scales = torch.clamp(scales, min=1e-5)
        assert torch.isnan(scales).sum() == 0
        return scales, zeros

    def forward(self, x, **kwargs):
        scales = kwargs.pop("scales", None)
        zeros = kwargs.pop("zeros", None)

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

    def extra_repr(self):
        s = f"Format: MX{self.str_format.split('.')[-1].upper()}, "
        s += f"Min: {-self.max_norm}, Max: {self.max_norm}, Axes: {self.axes}"
        return s


if __name__ == "__main__":
    torch.manual_seed(0)

    device = torch.device("cuda")
    x = torch.randn(4, 6).to(device=device)
    # print(x)

    quantizer = MXQuantizer(
        format=ElemFormat.int4,
        group_size=32,
        axes=-1,
        zero_point=False,
        scale_ebits=8,
        scale_mbits=0,
    )
    quantizer.to(device)
    # quantizer = MXQuantizer(
    #     format=ElemFormat.fp8_e4m3, group_size=32, axes=-1, zero_point=False, device=device,
    #     scale_ebits=8, scale_mbits = 1,
    # )
    # print(quantizer)
    x_dq = quantizer(x)
    print(x)
    print(x_dq)
    # print(x_dq)
    print(((x - x_dq) ** 2).mean())
