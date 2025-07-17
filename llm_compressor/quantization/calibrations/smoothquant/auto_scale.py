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

from utils.module import get_op_name, get_op_by_name  # noqa: E402


@torch.no_grad()
def get_act_scale(x):
    return x.abs().view(-1, x.shape[-1]).max(0)[0]


@torch.no_grad()
def scale_ln_fcs(ln, fcs, act_scales, alpha=0.5):
    if not isinstance(fcs, list):
        fcs = [fcs]

    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device).to(dtype)
    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
    )
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)
    scales = (
        (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
        .clamp(min=1e-5)
        .to(device)
        .to(dtype)
    )

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
def auto_scale_block(module, input_feat, **kwargs):
    def _auto_get_scale(prev_op, layers, inp):
        module2inspect = layers[0]
        device = next(module2inspect.parameters()).device
        scales = get_act_scale(inp.to(device))
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
        if "350m" not in model_name.lower():  # opt-350m
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
            )
        )
        # fc1
        scales_list.append(
            _auto_get_scale(
                prev_op=module.post_attention_layernorm,
                layers=[module.mlp.dense_h_to_4h],
                inp=input_feat["mlp.dense_h_to_4h"],
            )
        )

    elif isinstance(
        module,
        (
            LlamaDecoderLayer,
            Qwen2DecoderLayer,
            Qwen3DecoderLayer,
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
            )
        )
        # fc1
        scales_list.append(
            _auto_get_scale(
                prev_op=module.post_attention_layernorm,
                layers=[module.mlp.gate_proj, module.mlp.up_proj],
                inp=input_feat["mlp.gate_proj"],
            )
        )
    else:
        raise NotImplementedError(f"{type(module)} not supported yet!")

    return scales_list


def apply_scale(module, scales_list, device, alpha=0.5):
    for prev_op_name, layer_names, act_scales in scales_list:
        prev_op = get_op_by_name(module, prev_op_name)
        layers = [get_op_by_name(module, name) for name in layer_names]

        prev_op.to(device)
        for layer in layers:
            layer.to(device)
        act_scales.to(device)

        if isinstance(
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
            scale_ln_fcs(prev_op, layers, act_scales, alpha)
        else:
            raise NotImplementedError(f"prev_op {type(prev_op)} not supported yet!")
