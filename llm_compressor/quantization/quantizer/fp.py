import torch
from torch import nn

if __package__:
    from .formats import ElemFormat, _get_format_params
    from .utils import _reshape_to_blocks, _undo_reshape_to_blocks
else:
    from formats import ElemFormat, _get_format_params
    from utils import _reshape_to_blocks, _undo_reshape_to_blocks


class FPQuantizer(nn.Module):

    def __init__(
            self, 
            fmt: ElemFormat,
            group_size=-1,
            axes = -1,
            asymmetric=False,
            device=torch.device('cpu'),
            **kwargs):
        """
        asymmetric: not support!
        group_size:
            - 0: per-tensor quant.
            - -1: per-token quant.
            - -2: per-channel quant.
            - >0: per-group quant.
            - 2d-list or 2d-tuple: 2d-block quant.
        axes:
            - -1: per-token wise
            - -2: per-channle wise
        """
        super().__init__()

        assert fmt in (ElemFormat.fp4_e2m1, ElemFormat.fp8_e4m3, ElemFormat.fp8_e5m2), \
            f"Not support Format for {self.__class__.__name__}"

        ebits, mbits, emax, max_norm, min_norm = _get_format_params(fmt)
        self.ebits = torch.tensor(ebits)
        self.mbits = torch.tensor(mbits)
        self.emax = torch.tensor(emax)
        self.max_norm = torch.tensor(max_norm)
        self.min_norm = torch.tensor(min_norm)
        self.str_fmt = str(fmt)
        self.configure(asymmetric=asymmetric, group_size=group_size, axes=axes)
        self.enable()

    def configure(self, asymmetric, group_size, axes):
        assert (group_size != 1) or not asymmetric, (
            "Asymmetric quant with per-element quant. are exclusive."
        )
        self.asymmetric = asymmetric
        self.group_size = group_size

        if self.group_size == -1:
            self.axes = -1
        elif self.group_size == -2:
            self.axes = -2
        else:
            self.axes = axes

    def set_bias(self, max_val):
        # This is refered to https://github.com/quic/aimet/blob/develop/TrainingExtensions/torch/src/python/aimet_torch/fp_quantization.py#L172
        # Math explanation of what happens here:
        # Bias is computed from maxval: $B=2^E - \log_2(M) + \log_2(2 - 2^{-M}) - 1$
        # This follows from maxval $M=(2 - 2^{-M}) \cdot 2^{2^E-1-B}$.
        if 'fp8_e4m3' in self.str_fmt:
            bias = 2 ** self.ebits - torch.log2(max_val) + torch.log2(2 - 2 ** (1-self.mbits)) - 1

        elif 'fp8_e5m2' in self.str_fmt:
            bias = 2 ** self.ebits - torch.log2(max_val) + torch.log2(2 - 2 ** (-self.mbits)) - 2

        elif 'fp4_e2m1' in self.str_fmt:
            bias = 2 ** self.ebits - torch.log2(max_val) + torch.log2(2 - 2 ** (-self.mbits)) - 1
        return bias
    
    def find_params(self, x, already_reshaped=False):
        if (self.group_size != 0) & (not already_reshaped):
            if self.group_size == -1: # per-token quant.
                self.group_size = x.shape[-1]
            if self.group_size == -2: # per-channel quant.
                self.group_size = x.shape[-2]
            x, *_ = _reshape_to_blocks(
                x, block_size=self.group_size, axes=self.axes,
            )

        if self.group_size != 0:
            self.max_val = x.abs().amax(dim=self.axes, keepdim=True)
        else:
            self.max_val = x.abs().amax()

        # Ensure no values are greater than the maximum value represented by an 8 bit float system
        # with M mantissa and E exponent bits. torch.min/torch.max are used to allow gradients to
        # flow to max_val
        x_clipped = x.clamp(-self.max_val, self.max_val)

        # FP quantization scale is determined per-element, and is computed as
        # \log_2 s = \left\lfloor \log_2 |x_c| + B \right\rfloor - M - B
        # the addition of bias inside the floor and subtraction outside ensures that a
        # tensor scaling $\alpha \neq 1$ is correctly incorporated
        bias = self.set_bias(self.max_val)
        log_scales = torch.floor(torch.log2(torch.abs(x_clipped)) + bias).detach()
        
        # This ensures scales are never smaller than the subnormal scale
        log_scales = torch.clamp(log_scales, 1.0)

        # Second step of computing scale $s$
        scales = 2.0 ** (log_scales - self.mbits - bias)
        return scales
    
    def forward(self, x, **kwargs):
        scales = kwargs.pop("scales", None)

        if self.is_enable:
            if self.group_size != 0:
                if self.group_size == -1: # per-token quant.
                    self.group_size = x.shape[-1]
                if self.group_size == -2: # per-channel quant.
                    self.group_size = x.shape[-2]
                x, *meta = _reshape_to_blocks(
                    x, block_size=self.group_size, axes=self.axes,
                )

            if scales is not None:
                x_dq = self.quantize(x, scales=scales)
            else:
                scales = self.find_params(x, already_reshaped=True)
                x_dq = self.quantize(x, scales=scales)

            if self.group_size != 0:
                return _undo_reshape_to_blocks(x_dq, 
                                               padded_shape=meta[-2], 
                                               orig_shape=meta[1], 
                                               axes=meta[0], 
                                               block_size=meta[-1])
            return x_dq
        return x

    def quantize(self, x, scales):
        x_clipped = x.clamp(-self.max_val, self.max_val)
        # Using the per-element scale we can quantize the clipped input tensor to the FP grid
        return torch.round(x_clipped / scales) * scales

    def enable(self):
        self.is_enable = True
    
    def disable(self):
        self.is_enable = False

    def extra_repr(self):
        s = f"Format: {self.str_fmt.split('.')[-1].upper()}, "
        s += f"Max: {self.max_norm}, Min: {self.min_norm}"
        return s


if __name__ == "__main__":
    torch.manual_seed(0)

    device = torch.device("cuda")
    x = torch.randn(4, 6).to(device=device)
    print(x)

    quantizer = FPQuantizer(
        fmt=ElemFormat.fp4_e2m1, group_size=2, axes=-2, device=device
    )
    print(quantizer)
    x_dq = quantizer(x)
    print(x_dq)
    print(((x-x_dq)**2).mean())

    