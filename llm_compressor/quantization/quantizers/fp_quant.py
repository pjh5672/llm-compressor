import torch

if __package__:
    from .base import BaseQuantizer
    from .formats import ElemFormat, _get_format_params
    from .utils import (
        _reshape_to_blocks,
        _undo_reshape_to_blocks,
        _quantize_elemwise_core,
    )
else:
    from base import BaseQuantizer
    from formats import ElemFormat, _get_format_params
    from utils import (
        _reshape_to_blocks,
        _undo_reshape_to_blocks,
        _quantize_elemwise_core,
    )


class FPQuantizer(BaseQuantizer):
    def __init__(
        self,
        format: ElemFormat,
        group_size=-1,
        axes=-1,
        zero_point=False,
        is_profile=False,
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
        self.is_profile = is_profile
        op_name = kwargs.get("op_name", None)
        save_path = kwargs.get("save_path", "./")
        self.op_name = op_name if op_name is not None else "None"
        self.save_path = save_path

        super().__init__(
            op_name=self.op_name,
            save_path=self.save_path,
        )

        assert format in (
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
        self.str_format = str(format).split(".")[-1].upper()
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
                scales = (max_val - min_val) / (2 * self.max_norm)
                zeros = (max_val + min_val) / 2
            else:
                max_val = x.abs().amax(dim=self.axes, keepdim=True)
                min_val = -max_val
                scales = max_val / self.max_norm
                zeros = torch.zeros_like(scales)
        else:
            if self.zero_point:
                max_val = x.amax()
                min_val = x.amin()
                scales = (max_val - min_val) / (2 * self.max_norm)
                zeros = (max_val + min_val) / 2
            else:
                max_val = x.abs().amax()
                min_val = -max_val
                scales = max_val / self.max_norm
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
                    scales1 = (max_val1 - min_val1) / (2 * self.max_norm)
                    zeros1 = (max_val1 + min_val1) / 2
                else:
                    scales1 = max_val1 / self.max_norm
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

        scales = torch.clamp(scales, min=1e-5)
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

        if self.is_profile:
            self.record_stats(
                x=x,
                qdq_x=x_dq,
                qtype="FP",
                qformat=self.str_format,
                group_size=self.group_size,
                zero_point=self.zero_point,
            )

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
        s = f"Format: {self.str_format}, "
        s += f"Min: {-self.max_norm}, Max: {self.max_norm}, Axes: {self.axes}"
        if self.is_profile:
            s += f", Op name: {self.op_name}"
        return s


if __name__ == "__main__":
    torch.manual_seed(0)

    device = torch.device("cuda")
    x = torch.randn(4, 6).to(device=device)
    print(x)

    quantizer = FPQuantizer(
        format=ElemFormat.fp4_e2m1,
        group_size=16,
        axes=-1,
        zero_point=False,
    )
    quantizer.to(device)
    quantizer.mse = False
    # print(quantizer)
    # x_dq = quantizer(x)
    # print(x_dq)
    # print(((x - x_dq) ** 2).mean())

    scales, zeros = quantizer.find_params(x)
    # print(scales, zeros, scales.shape)
    x_dq = quantizer(x, scales=scales, zeros=zeros)
    print(((x - x_dq) ** 2).mean())
