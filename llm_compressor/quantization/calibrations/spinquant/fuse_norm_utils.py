import typing

import torch
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.phi.modeling_phi import PhiForCausalLM


def fuse_ln_linear(
    layernorm: torch.nn.Module, linear_layers: typing.Iterable[torch.nn.Linear]
) -> None:
    """
    fuse the linear operations in Layernorm into the adjacent linear blocks.
    """
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype

        # Calculating new weight and bias
        W_ = linear.weight.data.double()
        linear.weight.data = (W_ * layernorm.weight.double()).to(linear_dtype)

        if hasattr(layernorm, "bias"):
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(
                    torch.zeros(linear.out_features, dtype=torch.float64)
                )
            linear.bias.data = linear.bias.data.double() + torch.matmul(
                W_, layernorm.bias.double()
            )
            linear.bias.data = linear.bias.data.to(linear_dtype)


def fuse_layer_norms(model):
    # Embedding fusion
    for W in [model.model.embed_tokens]:
        W_ = W.weight.data.double()
        W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(W.weight.data.dtype)

    layers = model.get_layers()
    # Fuse the linear operations in Layernorm into the adjacent linear blocks.
    if isinstance(model, LlamaForCausalLM):
        for layer in layers:
            # fuse the input layernorms into the linear layers
            fuse_ln_linear(
                layer.post_attention_layernorm, [layer.mlp.up_proj, layer.mlp.gate_proj]
            )
            fuse_ln_linear(
                layer.input_layernorm,
                [
                    layer.self_attn.q_proj,
                    layer.self_attn.k_proj,
                    layer.self_attn.v_proj,
                ],
            )
            W_norm = layer.post_attention_layernorm.weight.data
            layer.post_attention_layernorm.weight.data = torch.ones_like(W_norm)
            W_norm = layer.input_layernorm.weight.data
            layer.input_layernorm.weight.data = torch.ones_like(W_norm)

        fuse_ln_linear(
            model.model.norm,
            [model.lm_head],
        )
        W_norm = model.model.norm.weight.data
        model.model.norm.weight.data = torch.ones_like(W_norm)

    elif isinstance(model, PhiForCausalLM):
        for layer in layers:
            # fuse the input layernorms into the linear layers
            fuse_ln_linear(
                layer.input_layernorm,
                [
                    layer.self_attn.q_proj,
                    layer.self_attn.k_proj,
                    layer.self_attn.v_proj,
                ],
            )
            W_norm = layer.input_layernorm.weight.data
            layer.input_layernorm.weight.data = torch.ones_like(W_norm)

        fuse_ln_linear(
            model.model.final_layernorm,
            [model.lm_head],
        )
        W_norm = model.model.final_layernorm.weight.data
        model.model.final_layernorm.weight.data = torch.ones_like(W_norm)

    else:
        raise RuntimeError(f"Not support model yet, got {model}")