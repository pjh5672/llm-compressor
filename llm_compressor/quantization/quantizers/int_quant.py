import torch
import torch.nn as nn

if __package__:
    from .base import BaseQuantizer
    from .formats import ElemFormat, _get_format_params
    from .utils import _reshape_to_blocks, _undo_reshape_to_blocks
else:
    from base import BaseQuantizer
    from formats import ElemFormat, _get_format_params
    from utils import _reshape_to_blocks, _undo_reshape_to_blocks


class INTQuantizer(nn.Module, BaseQuantizer):
    def __init__(
        self,
        format: ElemFormat,
        group_size=-1,
        axes=-1,
        zero_point=False,
        **kwargs,
    ):
        """
        PyTorch and ONNX use full-range quantization, while TensorFlow, NVIDIA,
        TensorRT, and Intel DNNL use restrictive-range. Full-range is slightly
        more accurate in theory, but there’s really no significant difference in practice."
        So we use resitrctive-range of [-127, 127] instead of [-128, 127].
        """

        """ Attributes
        group_size:
            - 0: per-tensor quant.
            - -1: per-token quant.
            - -2: per-channel quant.
            - >0: per-group quant.
            - 2d-list or 2d-tuple: 2d-block quant.
        axes:
            - -1: row-wise quant.
            - -2: column-wise quant.
        """
        super().__init__()

        assert format in (ElemFormat.int4, ElemFormat.int8), (
            f"Not support Format for {self.__class__.__name__}"
        )
        _, self.q_bits, _, max_norm, _ = _get_format_params(format)
        q_max = max_norm * 2 ** (self.q_bits - 2)
        q_min = -q_max
        self.register_buffer("q_max", torch.tensor(q_max))
        self.register_buffer("q_min", torch.tensor(q_min))
        self.str_format = str(format)
        self.mse = False
        self.configure(zero_point=zero_point, group_size=group_size, axes=axes)

    def configure(self, zero_point, group_size, axes):
        assert (group_size != 1) or not zero_point, (
            "Asymmetric quant with per-element quant. are exclusive."
        )
        self.zero_point = zero_point
        self.group_size = group_size

        if self.group_size == -1:
            self.axes = -1
        elif self.group_size == -2:
            self.axes = -2
        elif isinstance(self.group_size, list) or isinstance(self.group_size, tuple):
            self.axes = -1
        else:
            self.axes = axes

    def find_params(self, x, already_reshaped=False):
        if (self.group_size != 0) & (not already_reshaped):
            if self.group_size == -1:  # per-token quant.
                self.group_size = x.shape[-1]
            elif self.group_size == -2:  # per-channel quant.
                self.group_size = x.shape[-2]
            x, *_ = _reshape_to_blocks(
                x,
                block_size=self.group_size,
                axes=self.axes,
            )

        if self.group_size != 0:
            if self.zero_point:
                max_val = x.amax(dim=self.axes, keepdim=True)
                min_val = x.amin(dim=self.axes, keepdim=True)
                scales = (max_val - min_val) / (self.q_max - self.q_min)
                zeros = (self.q_min - min_val / scales).round()
            else:
                max_val = x.abs().amax(dim=self.axes, keepdim=True)
                min_val = -max_val
                scales = max_val / self.q_max
                zeros = torch.zeros_like(scales)
        else:
            if self.zero_point:
                max_val = x.amax()
                min_val = x.amin()
                scales = (max_val - min_val) / (self.q_max - self.q_min)
                zeros = (self.q_min - min_val / scales).round()
            else:
                max_val = x.abs().amax()
                min_val = -max_val
                scales = max_val / self.q_max
                zeros = torch.zeros_like(scales)

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
                    scales1 = (max_val1 - min_val1) / (self.q_max - self.q_min)
                    zeros1 = (self.q_min - min_val1 / scales1).round()
                else:
                    scales1 = max_val1 / self.q_max
                    zeros1 = torch.zeros_like(scales1)

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

        scales.clamp_(min=1e-5)
        assert torch.isnan(scales).sum() == 0
        return scales, zeros

    def forward(self, x, **kwargs):
        scales = kwargs.pop("scales", None)
        zeros = kwargs.pop("zeros", None)

        if self.group_size != 0:
            if self.group_size == -1:  # per-token quant.
                self.group_size = x.shape[-1]
            elif self.group_size == -2:  # per-channel quant.
                self.group_size = x.shape[-2]

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

        if self.group_size != 0:
            return _undo_reshape_to_blocks(
                x_dq,
                padded_shape=meta[-2],
                orig_shape=meta[1],
                axes=meta[0],
                block_size=meta[-1],
            )
        return x_dq

    def fake_quantize(self, x, scales, zeros):
        q = (x / scales + zeros).round().clamp(min=self.q_min, max=self.q_max)
        return (q - zeros) * scales

    def extra_repr(self):
        s = f"Format: {self.str_format.split('.')[-1].upper()}, "
        s += f"Min: {self.q_min}, Max: {self.q_max}, Axes: {self.axes}"
        return s


if __name__ == "__main__":
    torch.manual_seed(0)

    device = torch.device("cuda")
    x = torch.randn(4, 6).to(device=device)
    print(x)

    quantizer = INTQuantizer(
        format=ElemFormat.int4,
        group_size=-1,
        axes=-1,
        zero_point=True,
    )
    quantizer.to(device)
    quantizer.mse = False
    # quantizer = INTQuantizer(
    #     format=ElemFormat.int8, group_size=(6, 4), axes=-1, zero_point=False, device=device
    # )
    print(quantizer)
    # scales, zeros = quantizer.find_params(x)
    # print(scales, zeros, scales.shape)
    # x_dq = quantizer(x, scales=scales, zeros=zeros)
    x_dq = quantizer(x)
    print(x_dq)
    print(((x - x_dq) ** 2).mean())
