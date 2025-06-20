import torch
import torch.nn.functional as F

if __package__:
    from .formats import _get_min_norm
else:
    from formats import _get_min_norm


def _reshape(shape, reshape_block_size, axes):
    for axis in axes:
        # Reshape to tiles if axis length > reshape_block_size
        if shape[axis] >= reshape_block_size:
            assert shape[axis] % reshape_block_size == 0
            shape[axis + 1] = reshape_block_size
            shape[axis] = shape[axis] // reshape_block_size
        # Otherwise preserve length and insert a 1 into the shape
        else:
            shape[axis + 1] = shape[axis]
            shape[axis] = 1
    return shape


def tile_matrix(x, block_size):
    orig_shape = x.shape
    *p, h, w = orig_shape

    pad_bottom = (block_size[0] - h % block_size[0]) % block_size[0]
    pad_right = (block_size[1] - w % block_size[1]) % block_size[1]

    padded_x = F.pad(x, (0, pad_right, 0, pad_bottom), mode="constant", value=0)
    *p, new_h, new_w = padded_x.shape

    num_tiles_h = new_h // block_size[0]
    num_tiles_w = new_w // block_size[1]

    tiles = padded_x.view(*p, num_tiles_h, block_size[0], num_tiles_w, block_size[1])

    if x.dim() == 2:
        tiles = tiles.permute(0, 2, 1, 3)

    elif x.dim() == 3:
        tiles = tiles.permute(0, 1, 3, 2, 4)

    elif x.dim() == 4:
        tiles = tiles.permute(0, 2, 4, 1, 3, 5)

    tiles = tiles.reshape(*p, -1, block_size[0] * block_size[1])
    return tiles, orig_shape, padded_x.shape


def untile_matrix(x, block_size, padded_shape, orig_shape):
    *p, h, w = padded_shape

    num_tiles_h = h // block_size[0]
    num_tiles_w = w // block_size[1]

    if x.dim() == 2:
        tiles = x.reshape(*p, num_tiles_h, num_tiles_w, block_size[0], block_size[1])
        tiles = tiles.permute(0, 2, 1, 3)

    elif x.dim() == 3:
        tiles = x.reshape(*p, num_tiles_h, num_tiles_w, block_size[0], block_size[1])
        tiles = tiles.permute(0, 1, 3, 2, 4)

    elif x.dim() == 4:
        tiles = x.reshape(
            p[0], num_tiles_h, num_tiles_w, p[1], block_size[0], block_size[1]
        )
        tiles = tiles.permute(0, 3, 1, 4, 2, 5)

    tiles = tiles.reshape(*p, num_tiles_h * block_size[0], num_tiles_w * block_size[1])

    pad_bottom = padded_shape[0] - orig_shape[0]
    pad_right = padded_shape[1] - orig_shape[1]

    if pad_bottom > 0:
        tiles = tiles[..., :-pad_bottom, :]
    if pad_right > 0:
        tiles = tiles[..., :-pad_right]

    return tiles


def _reshape_to_blocks(A, block_size, axes=None):
    # if isinstance(block_size, list) or isinstance(block_size, tuple):
    #     if len(block_size) == 2:
    #         A, orig_shape, padded_shape = tile_matrix(A, block_size=block_size)
    #         return A, None, orig_shape, padded_shape, block_size
    #     else:
    #         raise Exception("block_size must be 2d-list or 2d-tuple")

    if axes is None:
        raise Exception(
            "axes required in order to determine which dimension toapply block size to"
        )
    if block_size == 0:
        raise Exception("block_size == 0 in _reshape_to_blocks")

    # Fix axes to be positive and sort them
    if isinstance(axes, int):
        axes = [axes]
    axes = [(x + len(A.shape) if x < 0 else x) for x in axes]
    assert all(x >= 0 for x in axes)
    axes = sorted(axes)

    # Add extra dimension for tiles
    for i in range(len(axes)):
        axes[i] += i  # Shift axes due to added dimensions
        A = torch.unsqueeze(A, dim=axes[i] + 1)

    # Pad to block_size
    orig_shape = A.size()
    pad = []
    for i in range(len(orig_shape)):
        pad += [0, 0]

    do_padding = False
    for axis in axes:
        pre_pad_size = orig_shape[axis]
        if isinstance(pre_pad_size, torch.Tensor):
            pre_pad_size = int(pre_pad_size.value)
        # Don't pad if the axis is short enough to fit inside one tile
        if pre_pad_size % block_size == 0:
            pad[2 * axis] = 0
        else:
            pad[2 * axis] = block_size - pre_pad_size % block_size
            do_padding = True

    if do_padding:
        pad = list(reversed(pad))
        A = torch.nn.functional.pad(A, pad, mode="constant")

    def _reshape(shape, reshape_block_size):
        for axis in axes:
            # Reshape to tiles if axis length > reshape_block_size
            if shape[axis] >= reshape_block_size:
                assert shape[axis] % reshape_block_size == 0
                shape[axis + 1] = reshape_block_size
                shape[axis] = shape[axis] // reshape_block_size
            # Otherwise preserve length and insert a 1 into the shape
            else:
                shape[axis + 1] = shape[axis]
                shape[axis] = 1
        return shape

    # Reshape to tiles
    padded_shape = A.size()
    reshape = _reshape(list(padded_shape), block_size)
    A = A.view(reshape)
    return A, axes, orig_shape, padded_shape, None


def _undo_reshape_to_blocks(A, padded_shape, orig_shape, axes, **kwargs):
    if block_size := kwargs.pop("block_size", False):
        return untile_matrix(A, block_size, padded_shape, orig_shape)

    # Undo tile reshaping
    A = A.view(padded_shape)
    # Undo padding
    if not list(padded_shape) == list(orig_shape):
        slices = [slice(0, x) for x in orig_shape]
        A = A[slices]
    for axis in reversed(axes):
        # Remove extra dimension
        A = torch.squeeze(A, dim=axis + 1)
    return A


def _safe_lshift(x, bits, exp):
    if exp is None:
        return x * (2**bits)
    else:
        return x / (2**exp) * (2**bits)


def _safe_rshift(x, bits, exp):
    if exp is None:
        return x / (2**bits)
    else:
        return x / (2**bits) * (2**exp)


def _round_mantissa(A, bits, round, clamp=False):
    """
    Rounds mantissa to nearest bits depending on the rounding method 'round'
    Args:
      A     {PyTorch tensor} -- Input tensor
      round {str}            --  Rounding method
                                 "floor" rounds to the floor
                                 "nearest" rounds to ceil or floor, whichever is nearest
    Returns:
      A {PyTorch tensor} -- Tensor with mantissas rounded
    """

    if round == "dither":
        rand_A = torch.rand_like(A, requires_grad=False)
        A = torch.sign(A) * torch.floor(torch.abs(A) + rand_A)
    elif round == "floor":
        A = torch.sign(A) * torch.floor(torch.abs(A))
    elif round == "nearest":
        A = torch.sign(A) * torch.floor(torch.abs(A) + 0.5)
    elif round == "even":
        absA = torch.abs(A)
        # find 0.5, 2.5, 4.5 ...
        maskA = ((absA - 0.5) % 2 == torch.zeros_like(A)).type(A.dtype)
        A = torch.sign(A) * (torch.floor(absA + 0.5) - maskA)
    else:
        raise Exception("Unrecognized round method %s" % (round))

    # Clip values that cannot be expressed by the specified number of bits
    if clamp:
        max_mantissa = 2 ** (bits - 1) - 1
        A = torch.clamp(A, -max_mantissa, max_mantissa)
    return A


def _quantize_elemwise_core(
    A,
    bits,
    exp_bits,
    max_norm,
    round="nearest",
    saturate_normals=False,
    allow_denorm=True,
    custom_cuda=False,
):
    """Core function used for element-wise quantization
    Arguments:
      A         {PyTorch tensor} -- A tensor to be quantized
      bits      {int}            -- Number of mantissa bits. Includes
                                    sign bit and implicit one for floats
      exp_bits  {int}            -- Number of exponent bits, 0 for ints
      max_norm  {float}          -- Largest representable normal number
      round     {str}            -- Rounding mode: (floor, nearest, even)
      saturate_normals {bool}    -- If True, normal numbers (i.e., not NaN/Inf)
                                    that exceed max norm are clamped.
                                    Must be True for correct MX conversion.
      allow_denorm     {bool}    -- If False, flush denorm numbers in the
                                    elem_format to zero.
      custom_cuda      {str}     -- If True, use custom CUDA kernels
    Returns:
      quantized tensor {PyTorch tensor} -- A tensor that has been quantized
    """

    # Flush values < min_norm to zero if denorms are not allowed
    if not allow_denorm and exp_bits > 0:
        min_norm = _get_min_norm(exp_bits)
        out = (torch.abs(A) >= min_norm).type(A.dtype) * A
    else:
        out = A

    if exp_bits != 0:
        private_exp = torch.floor(torch.log2(torch.abs(A) + (A == 0).type(A.dtype)))

        # The minimum representable exponent for 8 exp bits is -126
        min_exp = -(2 ** (exp_bits - 1)) + 2
        private_exp = private_exp.clip(min=min_exp)
    else:
        private_exp = None

    # Scale up so appropriate number of bits are in the integer portion of the number
    out = _safe_lshift(out, bits - 2, private_exp)

    out = _round_mantissa(out, bits, round, clamp=False)

    # Undo scaling
    out = _safe_rshift(out, bits - 2, private_exp)

    # Set values > max_norm to Inf if desired, else clamp them
    if saturate_normals or exp_bits == 0:
        out = torch.clamp(out, min=-max_norm, max=max_norm)
    else:
        out = torch.where(
            (torch.abs(out) > max_norm), torch.sign(out) * float("Inf"), out
        )

    # handle Inf/NaN
    if not custom_cuda:
        out[A == float("Inf")] = float("Inf")
        out[A == -float("Inf")] = -float("Inf")
        out[A == float("NaN")] = float("NaN")

    return out
