import sys
from pathlib import Path

import torch
from torch import nn
from transformers.models.opt.modeling_opt import OPTDecoderLayer

PATH = Path(__file__).resolve().parents[3]
if str(PATH) not in sys.path:
    sys.path.append(str(PATH))

from utils.general import LOGGER  # noqa: E402
from utils.module import get_op_name, get_op_by_name  # noqa: E402


@torch.no_grad()
def get_act_scale(x):
    return x.abs().view(-1, x.shape[-1]).mean(0)


@torch.no_grad()
def scale_ln_fcs(ln, fcs, scales):
    if not isinstance(fcs, list):
        fcs = [fcs]

    scales = scales.to(ln.weight.device).to(ln.weight.dtype)

    ln.weight.div_(scales)
    if hasattr(ln, "bias") and ln.bias is not None:
        ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))

    for p in ln.parameters():
        assert torch.isnan(p).sum() == 0
    for fc in fcs:
        for p in fc.parameters():
            assert torch.isnan(p).sum() == 0


@torch.no_grad()
def scale_fc_fc(fc1, fc2, scales):
    assert isinstance(fc1, nn.Linear)
    assert isinstance(fc2, nn.Linear)

    scales = scales.to(fc1.weight.device).to(fc1.weight.dtype)

    fc1.weight[-scales.size(0) :].div_(scales.view(-1, 1))
    if fc1.bias is not None:
        fc1.bias.div_(scales.view(-1))

    fc2.weight.mul_(scales.view(1, -1))

    for p in fc1.parameters():
        assert torch.isnan(p).sum() == 0
    for p in fc2.parameters():
        assert torch.isnan(p).sum() == 0


@torch.no_grad()
def auto_scale_block(module, module_kwargs, input_feat):

    # find the best scale ratio
    def _search_module_scale(block, linears2scale: list, x, kwargs={}):
        # w: co, ci
        # x: n, ci
        device = next(block.parameters()).device
        x = x.to(device)
        with torch.no_grad():
            org_out = block(x, **kwargs)
            if isinstance(org_out, tuple):
                org_out = org_out[0]

        x_max = get_act_scale(x)
        
        best_error = float("inf")
        best_ratio = -1
        best_scales = None

        n_grid = 20
        history = []
        
        org_sd = {k: v.cpu() for k, v in block.state_dict().items()}
        for ratio in range(n_grid):
            ratio = ratio * 1 / n_grid
            scales = x_max.pow(ratio).clamp(min=1e-4).view(-1)
            scales = scales / (scales.max() * scales.min()).sqrt()
            
            for fc in linears2scale:
                fc.weight.mul_(scales.view(1, -1).to(device))
                fc.weight.data = fc.weight_quantizer(fc.weight.data) / (scales.view(1, -1))

            out = block(x, **kwargs)
            if isinstance(out, tuple):
                out = out[0]
            
            loss = (
                (org_out - out).float().pow(2).mean().item()
            )  # float prevents overflow
            history.append(loss)
            is_best = loss < best_error
            if is_best:
                best_error = loss
                best_ratio = ratio
                best_scales = scales
            block.load_state_dict(org_sd)

        if best_ratio == -1:
            LOGGER.error(f"Loss got infinity. : {history}")
            raise Exception

        LOGGER.debug(f"Best ratio : {best_ratio}")
        best_scales = best_scales.view(-1)

        assert torch.isnan(best_scales).sum() == 0, best_scales
        return best_scales.detach()
    
    def _auto_get_scale(prev_op, layers, inp, module2inspect=None, kwargs={}):
        # module2inspect: if given, we will check the output diff of this module instead of layers
        if module2inspect is None:
            assert len(layers) == 1
            module2inspect = layers[0]

        scales = _search_module_scale(module2inspect, layers, inp, kwargs)
        scales = scales.detach().cpu()
        # prev_op_name, [layer_name], scale
        return (
            get_op_name(module, prev_op),
            tuple([get_op_name(module, m) for m in layers]),
            scales,
        )
    
    scales_list = []  # return the searched scales

    if isinstance(module, OPTDecoderLayer):
        # attention input
        scales_list.append(
            _auto_get_scale(
                prev_op=module.self_attn_layer_norm,
                layers=[
                    module.self_attn.q_proj,
                    module.self_attn.k_proj,
                    module.self_attn.v_proj,
                ],
                inp=input_feat["self_attn.q_proj"],
                module2inspect=module.self_attn,
                kwargs=module_kwargs,
            )
        )
        # attn out
        scales_list.append(
            _auto_get_scale(
                prev_op=module.self_attn.v_proj,
                layers=[module.self_attn.out_proj],
                inp=input_feat["self_attn.out_proj"],
            )
        )
        # fc1
        scales_list.append(
            _auto_get_scale(
                prev_op=module.final_layer_norm,
                layers=[module.fc1],
                inp=input_feat["fc1"],
            )
        )

    else:
        raise NotImplementedError(f"{type(module)} not supported yet!")
    
    return scales_list


def apply_scale(module, scales_list, device, input_feat_dict=None):
    for prev_op_name, layer_names, scales in scales_list:
        prev_op = get_op_by_name(module, prev_op_name)
        layers = [get_op_by_name(module, name) for name in layer_names]

        prev_op.to(device)
        for layer in layers:
            layer.to(device)
        scales.to(device)

        if isinstance(prev_op, nn.Linear):
            assert len(layers) == 1
            scale_fc_fc(prev_op, layers[0], scales)
        elif isinstance(prev_op, (nn.LayerNorm, )):
            scale_ln_fcs(prev_op, layers, scales)
        elif isinstance(prev_op, (nn.ReLU, )):
            LOGGER.info(f"We don't use extra scaling node for activation, {prev_op}.")
        else:
            raise NotImplementedError(f"prev_op {type(prev_op)} not supported yet!")
        
        # apply the scaling to input feat if given; prepare it for clipping
        if input_feat_dict is not None:
            for layer_name in layer_names:
                inp = input_feat_dict[layer_name]
                inp.div_(scales.view(1, -1).to(inp.device).to(inp.dtype))

        prev_op.cpu()
        for layer in layers:
            layer.cpu()
        scales.cpu()