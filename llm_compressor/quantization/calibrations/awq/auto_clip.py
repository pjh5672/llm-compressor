import sys
from pathlib import Path

import torch
from torch import nn

PATH = Path(__file__).resolve().parents[3]
if str(PATH) not in sys.path:
    sys.path.append(str(PATH))

from utils.general import LOGGER  # noqa: E402
from utils.module import get_op_by_name  # noqa: E402
from utils.torch_utils import cleanup_memory  # noqa: E402


@torch.no_grad()
def auto_clip_layer(
    layer, input_feat, n_grid=20, max_shrink=0.5, n_sample_token=512
):
    w = layer.weight

    assert w.dim() == 2
    # w           [co, ci]      -> [co, 1, n_group, group size]
    # input_feat  [n_token, ci] -> [1, n_token, n_group, group size]
    group_size = layer.weight_quantizer.group_size
    if group_size == 0:
        return 
    if isinstance(group_size, (tuple, list)):
        group_size = group_size[-1]
    
    input_feat = input_feat.view(-1, input_feat.shape[-1])
    input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)
    input_feat = input_feat[:, 0 :: input_feat.shape[1] // n_sample_token]
    w = w.reshape(w.shape[0], 1, -1, group_size)

    oc_batch_size = 256 if w.shape[0] % 256 == 0 else 64  # prevent OOM
    assert w.shape[0] % oc_batch_size == 0
    w_all = w
    best_max_val_all = []

    for i_b in range(w.shape[0] // oc_batch_size):
        w = w_all[i_b * oc_batch_size : (i_b + 1) * oc_batch_size]

        org_max_val = w.abs().amax(dim=-1, keepdim=True)  # co, 1, n_group, 1
        best_max_val = org_max_val.clone()
        min_errs = torch.ones_like(org_max_val) * 1e9
        input_feat = input_feat.to(w.device)
        org_out = (input_feat * w).sum(dim=-1)  # co, n_token, n_group

        for i_s in range(int(max_shrink * n_grid)):
            max_val = org_max_val * (1 - i_s / n_grid)
            min_val = -max_val
            cur_w = torch.clamp(w, min_val, max_val)
            q_w = layer.weight_quantizer(cur_w)
            cur_out = (input_feat * q_w).sum(dim=-1)

            # co, 1, n_group, 1
            err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
            del cur_w
            del cur_out
            cur_best_idx = err < min_errs
            min_errs[cur_best_idx] = err[cur_best_idx]
            best_max_val[cur_best_idx] = max_val[cur_best_idx]

        best_max_val_all.append(best_max_val)
    best_max_val = torch.cat(best_max_val_all, dim=0)
    del input_feat
    del org_out
    cleanup_memory(verbose=False)
    return best_max_val.squeeze(1)


@torch.no_grad()
def auto_clip_block(module, input_feat, device):
    named_linears = {
        name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)
    }

    clip_list = []
    for name in named_linears:
        # due to qk bmm, it is hard to clip precisely
        if any([_ in name for _ in ["q_", "k_", "query", "key", "Wqkv"]]):
            continue

        named_linears[name].to(device)
        max_val = auto_clip_layer(named_linears[name], input_feat[name])
        clip_list.append((name, max_val))
        named_linears[name].cpu()
    return clip_list


@torch.no_grad()
def apply_clip(module, clip_list, device):
    for name, max_val in clip_list:
        if max_val is None:
            continue
        layer = get_op_by_name(module, name)
        layer.to(device)
        max_val = max_val.to(device).to(layer.weight.dtype)
        org_shape = layer.weight.shape
        layer.weight.data = layer.weight.data.reshape(*max_val.shape[:2], -1)
        layer.weight.data = torch.clamp(layer.weight.data, -max_val, max_val)
        layer.weight.data = layer.weight.data.reshape(org_shape)
        layer.cpu()