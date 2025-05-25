import torch
import torch.nn.functional as F


def tile_matrix(x, block_size):
    orig_shape = x.shape
    *p, h, w = orig_shape
    
    pad_bottom = (block_size[0] - h % block_size[0]) % block_size[0]
    pad_right = (block_size[1] - w % block_size[1]) % block_size[1]

    padded_x = F.pad(x, (0, pad_right, 0, pad_bottom), mode='constant', value=0)
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
        tiles = x.reshape(p[0], num_tiles_h, num_tiles_w, p[1], block_size[0], block_size[1])
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
    if isinstance(block_size, list) or isinstance(block_size, tuple) and len(block_size) > 1:
        A, orig_shape, padded_shape = tile_matrix(A, block_size=block_size)
        return A, None, orig_shape, padded_shape, block_size
    
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
    if block_size := kwargs.pop('block_size', False):
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
