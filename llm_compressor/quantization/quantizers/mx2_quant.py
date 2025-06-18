import torch
import torch.nn as nn

if __package__:
    from .formats import ElemFormat, FP32_MIN_NORMAL, _get_format_params
    from .utils import (
        _reshape_to_blocks,
        _undo_reshape_to_blocks,
        _quantize_elemwise_core,
        _safe_lshift,
        _safe_rshift,
        _round_mantissa,
        _reshape,
    )
else:
    from formats import ElemFormat, FP32_MIN_NORMAL, _get_format_params
    from utils import (
        _reshape_to_blocks,
        _undo_reshape_to_blocks,
        _quantize_elemwise_core,
        _safe_lshift,
        _safe_rshift,
        _round_mantissa,
        _reshape,
    )

from torch.nn.functional import pad

def resolve_axis(axis, ndim):
    """Convert negative axis to positive."""
    return axis if axis >= 0 else axis + ndim

def two_level_grouping(tensor, axis=-1, group_size=128, sub_group_size=32):
    assert group_size % sub_group_size == 0, "group_size must be divisible by sub_group_size"

    def _reshape_to_blocks(A, axes, block_size):
        if axes is None:
            raise Exception("axes required in order to determine which dimension to apply block size to")
        if block_size == 0:
            raise Exception("block_size == 0 in _reshape_to_blocks")

        # Fix axes to be positive and sorted
        axes = [(x + len(A.shape) if x < 0 else x) for x in axes]
        axes = sorted(axes)

        for i in range(len(axes)):
            axes[i] += i
            A = torch.unsqueeze(A, dim=axes[i] + 1)

        orig_shape = A.size()
        pad_config = [0, 0] * len(orig_shape)
        do_padding = False
        for axis in axes:
            dim_len = orig_shape[axis]
            if dim_len % block_size != 0:
                pad_amount = block_size - (dim_len % block_size)
                pad_config[2 * axis] = pad_amount
                do_padding = True

        if do_padding:
            pad_config = list(reversed(pad_config))
            A = pad(A, pad_config, mode='constant')

        padded_shape = A.size()

        def _reshape(shape, bsize):
            for axis in axes:
                if shape[axis] >= bsize:
                    assert shape[axis] % bsize == 0
                    shape[axis + 1] = bsize
                    shape[axis] = shape[axis] // bsize
                else:
                    shape[axis + 1] = shape[axis]
                    shape[axis] = 1
            return shape

        reshape = _reshape(list(padded_shape), block_size)
        A = A.view(reshape)
        return A, axes

    # Step 1: group into [*, num_groups, 128]
    grouped_128, _ = _reshape_to_blocks(tensor, axes=[axis], block_size=group_size)

    # Step 2: further reshape each 128 group → [num_groups, 4, 32]
    last_dim = grouped_128.shape[-1]
    assert last_dim == group_size, f"Last dim must be {group_size}"
    num_subgroups = group_size // sub_group_size

    grouped_32 = grouped_128.view(*grouped_128.shape[:-1], num_subgroups, sub_group_size)

    return grouped_32  # shape: [*, num_groups, 4, 32]


class MX2Quantizer(nn.Module):
    def __init__(
        self,
        format: ElemFormat,
        group_size=[128, 32],
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
        self.sp_scale_ebits = kwargs.get("sp_scale_ebits", 4)
        self.sb_scale_ebits = kwargs.get("sb_scale_ebits", 2)
        self.sp_scale_emax = 2 ** (self.sp_scale_ebits - 1) - 1
        self.sb_scale_emax = 2 ** (self.sb_scale_ebits - 1) - 1
        self.str_format = str(format)
        self.mse = False
        self.configure(zero_point=zero_point, group_size=group_size, axes=axes)

    def configure(self, zero_point, group_size, axes):
        self.zero_point = zero_point
        self.group_size = group_size
        self.axes = axes

    def find_params(self, tensor, axis=-1, group_size=128, sub_group_size=32, eps=1e-8):
        assert group_size % sub_group_size == 0

        # Resolve axis to positive
        axis = resolve_axis(axis, tensor.ndim)

        # Step 1: reshape to [*, num_blocks_128, 4, 32]
        reshaped, axes, orig_shape, padded_shape, _ = _reshape_to_blocks(tensor, group_size, [axis])
        reshaped = reshaped.view(*reshaped.shape[:-1], group_size//sub_group_size, sub_group_size)
        # reshaped = two_level_grouping(tensor, axis=axis, group_size=group_size, sub_group_size=sub_group_size)

        # Step 2: Compute scale per 128-block
        scale_128 = reshaped.abs().amax(dim=(-2, -1), keepdim=True) + eps  # shape: [..., G128, 1, 1]

        # Step 3: Compute scale per 32-block (subgroup)
        scale_32 = reshaped.abs().amax(dim=-1, keepdim=True) + eps  # shape: [..., G128, 4, 1]

        # Step 4: Normalize tensor (two-level scaling)
        normalized = reshaped / scale_128  # global scale
        normalized = normalized / scale_32  # local scale

        meta = {
            "axis": axis,
            "orig_shape": orig_shape,
            "padded_shape": padded_shape,
            "scale_128": scale_128.squeeze(-1).squeeze(-1),
            "scale_32": scale_32.squeeze(-1)
        }
        return normalized, meta

    def two_level_dequantize(self, normalized, meta):
        scale_128 = meta["scale_128"]
        scale_32 = meta["scale_32"]
        axis = meta["axis"]
        orig_shape = meta["orig_shape"]
        padded_shape = meta["padded_shape"]

        while scale_128.ndim < normalized.ndim:
            scale_128 = scale_128.unsqueeze(-1)
        while scale_32.ndim < normalized.ndim:
            scale_32 = scale_32.unsqueeze(-1)

        recovered = normalized * scale_32 * scale_128

        group_size = scale_32.shape[-1] * scale_128.shape[-2]
        reshape_back = recovered.view(*padded_shape)

        slices = []
        for i in range(len(orig_shape)):
            dim_len = orig_shape[i]
            slices.append(slice(0, dim_len))
        trimmed = reshape_back[tuple(slices)]
        return trimmed.view(orig_shape)

    def forward(self, x, **kwargs):
        scales = kwargs.pop("scales", None)
        zeros = kwargs.pop("zeros", None)

        qW, meta = self.find_params(x)
        x_dq = self.fake_quantize(qW,  meta)
        return x_dq.squeeze_()

    def fake_quantize(self, x, meta):
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
        dqW = self.two_level_dequantize(q, meta)
        return dqW

    def extra_repr(self):
        s = f"Format: MX{self.str_format.split('.')[-1].upper()}, "
        s += f"Min: {-self.max_norm}, Max: {self.max_norm}, Axes: {self.axes}"
        return s


if __name__ == "__main__":
    torch.manual_seed(0)

    device = torch.device("cuda")
    x = torch.randn(4, 6).to(device=device)
    # print(x)

    quantizer = MX2Quantizer(
        format=ElemFormat.int4,
        group_size=[128, 32],
        axes=-1,
        zero_point=False,
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
