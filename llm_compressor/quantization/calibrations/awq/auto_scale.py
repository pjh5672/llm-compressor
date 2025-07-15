import sys
from pathlib import Path

import torch
from torch import nn
from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.models.bloom.modeling_bloom import BloomBlock
from transformers.models.phi.modeling_phi import PhiDecoderLayer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RMSNorm
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RMSNorm
from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer, GemmaRMSNorm
from transformers.models.gemma2.modeling_gemma2 import Gemma2DecoderLayer, Gemma2RMSNorm
from transformers.models.gemma3.modeling_gemma3 import Gemma3DecoderLayer, Gemma3RMSNorm

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
def auto_scale_block(module, module_kwargs, input_feat, **kwargs):
    # find the best scale ratio
    def _search_module_scale(block, linears2scale: list, x, kwargs={}):
        # w: co, ci
        # x: n, ci
        device = next(block.parameters()).device
        x = x.to(device)

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
                fc.weight.data = fc.weight_quantizer(fc.weight.data) / (
                    scales.view(1, -1)
                )

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
        model_name = kwargs.get("model_name")
        if "350m" in model_name.lower():  # opt-350m
            # attn out
            scales_list.append(
                _auto_get_scale(
                    prev_op=module.self_attn.v_proj,
                    layers=[module.self_attn.out_proj],
                    inp=input_feat["self_attn.out_proj"],
                )
            )
        else:
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

    elif isinstance(module, BloomBlock):
        # attention input
        scales_list.append(
            _auto_get_scale(
                prev_op=module.input_layernorm,
                layers=[module.self_attention.query_key_value],
                inp=input_feat["self_attention.query_key_value"],
                module2inspect=module,
                kwargs=module_kwargs,
            )
        )
        # fc1
        scales_list.append(
            _auto_get_scale(
                prev_op=module.post_attention_layernorm,
                layers=[module.mlp.dense_h_to_4h],
                inp=input_feat["mlp.dense_h_to_4h"],
                module2inspect=module,
                kwargs=module_kwargs,
            )
        )

    elif isinstance(
        module,
        (
            LlamaDecoderLayer,
            Qwen2DecoderLayer,
            Qwen3DecoderLayer,
            GemmaDecoderLayer,
        ),
    ):
        # attention input
        scales_list.append(
            _auto_get_scale(
                prev_op=module.input_layernorm,
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
        # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
        if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
            scales_list.append(
                _auto_get_scale(
                    prev_op=module.self_attn.v_proj,
                    layers=[module.self_attn.o_proj],
                    inp=input_feat["self_attn.o_proj"],
                )
            )
        # fc1
        scales_list.append(
            _auto_get_scale(
                prev_op=module.post_attention_layernorm,
                layers=[module.mlp.gate_proj, module.mlp.up_proj],
                inp=input_feat["mlp.gate_proj"],
                module2inspect=module.mlp,
            )
        )
        # fc2
        scales_list.append(
            _auto_get_scale(
                prev_op=module.mlp.up_proj,
                layers=[module.mlp.down_proj],
                inp=input_feat["mlp.down_proj"],
            )
        )
    elif isinstance(module, PhiDecoderLayer):
        # attention input
        scales_list.append(
            _auto_get_scale(
                prev_op=module.input_layernorm,
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
                layers=[module.self_attn.dense],
                inp=input_feat["self_attn.dense"],
            )
        )
        # fc1
        scales_list.append(
            _auto_get_scale(
                prev_op=module.self_attn.dense,
                layers=[module.mlp.fc1],
                inp=input_feat["mlp.fc1"],
            )
        )
    elif isinstance(module, Gemma2DecoderLayer, Gemma3DecoderLayer):
        # attention input
        scales_list.append(
            _auto_get_scale(
                prev_op=module.input_layernorm,
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
        # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
        if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
            scales_list.append(
                _auto_get_scale(
                    prev_op=module.self_attn.v_proj,
                    layers=[module.self_attn.o_proj],
                    inp=input_feat["self_attn.o_proj"],
                )
            )
        # fc1
        scales_list.append(
            _auto_get_scale(
                prev_op=module.pre_feedforward_layernorm,
                layers=[module.mlp.gate_proj, module.mlp.up_proj],
                inp=input_feat["mlp.gate_proj"],
                module2inspect=module.mlp,
            )
        )
        # fc2
        scales_list.append(
            _auto_get_scale(
                prev_op=module.mlp.up_proj,
                layers=[module.mlp.down_proj],
                inp=input_feat["mlp.down_proj"],
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
        elif isinstance(
            prev_op,
            (
                nn.LayerNorm,
                LlamaRMSNorm,
                Qwen2RMSNorm,
                Qwen3RMSNorm,
                GemmaRMSNorm,
                Gemma2RMSNorm,
                Gemma3RMSNorm,
            ),
        ):
            scale_ln_fcs(prev_op, layers, scales)
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
