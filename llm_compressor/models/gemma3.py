import os
import sys
from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
from torch import nn
from transformers import AutoTokenizer
from accelerate import init_empty_weights
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3Attention,
    Gemma3ForCausalLM,
    repeat_kv,
    apply_rotary_pos_emb,
)

PATH = Path(__file__).resolve().parents[1]
if str(PATH) not in sys.path:
    sys.path.append(str(PATH))

from utils.general import LOGGER  # noqa: E402
from modules.qmatmul import QMatmul  # noqa: E402
from modules.qlinear import QLinear  # noqa: E402
from models.base import CompressForCausalLM  # noqa: E402


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    softcap: Optional[float] = None,
    **kwargs,
):
    qk_matmul = kwargs.get("qk_matmul")
    sv_matmul = kwargs.get("sv_matmul")

    if scaling is None:
        scaling = module.head_dim**-0.5

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = qk_matmul(query, key_states.transpose(2, 3)) * scaling

    if softcap is not None:
        attn_weights = attn_weights / softcap
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * softcap
    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query.dtype
    )
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )
    attn_output = sv_matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class QuantGemma3Attention(Gemma3Attention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        attention: Gemma3Attention,
        quant_config,
        **kwargs,
    ):
        super().__init__(
            attention.config,
            attention.layer_idx,
        )
        op_name = kwargs.get("op_name", None)
        save_path = kwargs.get("save_path", "./")
        mixed_precision = kwargs.get("mixed_precision", None)

        qk_matmul_config = sv_matmul_config = quant_config
        if mixed_precision is not None:
            for lname in mixed_precision.layers:
                if f"{op_name}.qk_matmul" == lname:
                    qk_matmul_config = mixed_precision.layers[lname]
                elif f"{op_name}.sv_matmul" == lname:
                    sv_matmul_config = mixed_precision.layers[lname]

        self.qk_matmul = QMatmul(
            qk_matmul_config,
            axes=-1,
            op_name=f"{op_name}.qk_matmul",
            save_path=save_path,
        )
        self.sv_matmul = QMatmul(
            sv_matmul_config,
            axes=-2,
            op_name=f"{op_name}.sv_matmul",
            save_path=save_path,
        )
        self.q_proj = attention.q_proj
        self.k_proj = attention.k_proj
        self.v_proj = attention.v_proj
        self.o_proj = attention.o_proj
        self.attn_logit_softcapping = attention.attn_logit_softcapping
        self.sliding_window = attention.sliding_window

        self.q_norm = attention.q_norm
        self.k_norm = attention.k_norm

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
                "sliding_window": self.sliding_window,
            }
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attention_interface: Callable = eager_attention_forward

        if attention_mask is not None:
            # backwards compatibility
            attention_mask = attention_mask.to(query_states)
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            qk_matmul=self.qk_matmul,
            sv_matmul=self.sv_matmul,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class CompressGemma3ForCausalLM(Gemma3ForCausalLM, CompressForCausalLM):
    def __init__(
        self,
        config,
    ):
        super().__init__(config)

    def _prepare_qmodule(self, quant_config, save_path="./", **kwargs):
        mixed_precision = kwargs.get("mixed_precision")

        for name, module in self.named_modules():
            if isinstance(module, Gemma3Attention):
                parent, child_name = name.rsplit(".", 1)
                parent_module = dict(self.named_modules())[parent]
                qattn = QuantGemma3Attention(
                    attention=getattr(parent_module, child_name),
                    quant_config=quant_config.matmul,
                    op_name=name.replace("model.", ""),
                    save_path=save_path,
                    mixed_precision=mixed_precision,
                )
                setattr(parent_module, child_name, qattn)

        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                op_name = name.replace("model.", "")

                if "lm_head" not in name:
                    parent, child_name = name.rsplit(".", 1)
                    parent_module = dict(self.named_modules())[parent]
                    quant_config_linear = quant_config.linear
                    if mixed_precision is not None:
                        for lname in mixed_precision.layers:
                            if op_name == lname:
                                quant_config_linear = mixed_precision.layers[lname]

                    qlinear = QLinear(
                        linear=getattr(parent_module, child_name),
                        quant_config=quant_config_linear,
                        dtype=self.dtype,
                        op_name=op_name,
                        save_path=save_path,
                    )
                    setattr(parent_module, child_name, qlinear)

                if "lm_head" in name:
                    head_module = dict(self.named_modules())[name]
                    qlinear = QLinear(
                        linear=head_module,
                        quant_config=quant_config.head,
                        dtype=self.dtype,
                        op_name=op_name,
                        save_path=save_path,
                    )
                    setattr(self, name, qlinear)

    def save_compressed(self, local_save_path):
        LOGGER.info("Saving compressed model...")
        base_model_path = self.config._name_or_path.rstrip(os.sep)

        with init_empty_weights():
            tokenizer = AutoTokenizer.from_pretrained(base_model_path)
            base_model = Gemma3ForCausalLM.from_pretrained(
                base_model_path, low_cpu_mem_usage=True
            )

        compressed_sd = {}
        for k, v in self.state_dict().items():
            compressed_sd[k] = v

        Gemma3ForCausalLM.save_pretrained(
            base_model,
            local_save_path,
            state_dict=compressed_sd,
        )
        tokenizer.save_pretrained(local_save_path)
        LOGGER.info(f"Save complete ! : {local_save_path}")

    def get_layers(self):
        return self.model.layers

    def get_sequential(self, mode="true"):
        if mode == "true":
            return [
                ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
                ["self_attn.o_proj"],
                ["mlp.up_proj", "mlp.gate_proj"],
                ["mlp.down_proj"],
            ]
        else:
            return [
                [
                    "self_attn.k_proj",
                    "self_attn.v_proj",
                    "self_attn.q_proj",
                    "self_attn.o_proj",
                    "mlp.up_proj",
                    "mlp.gate_proj",
                    "mlp.down_proj",
                ]
            ]

    def move_embed(self, device):
        self.model.embed_tokens = self.model.embed_tokens.to(device)
        self.model.rotary_emb = self.model.rotary_emb.to(device)


if __name__ == "__main__":
    from evaluation.eval import LMEvaluator
    from utils.args import build_parser, QuantConfigParser

    ROOT = Path(__file__).resolve().parents[1]
    args, device = build_parser(ROOT)

    qparser = QuantConfigParser(profile=args.profile)
    quant_config = qparser.build_cfg(args.weight, args.act_in, args.act_out, args.head)

    model_path = "d:\\models\\gemma-3-1b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = CompressGemma3ForCausalLM.from_pretrained(
        model_path,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    model.prune(
        tokenizer=tokenizer,
        prune_method=args.prune_method,
        prune_config=args.prune_config,
        device=device,
        prune=args.prune,
    )
    
    if args.profile:
        model.profile(
            quant_config=quant_config,
            device=device,
            save_path=args.exp_dir,
        )
    
    # qparser.register_org_config([

    # ])

    # if args.profile:
    #     model.profile(
    #         quant_config=quant_config,
    #         device=device,
    #         save_path=args.exp_dir,
    #         mixed_precision=qparser.mpq
    #     )

    quant_kwargs = {
        "n_samples": 128,
        "seq_len": 512,
        "rotation_path": args.rotation_path,
    }
    model.quantize(
        tokenizer=tokenizer,
        quant_method=args.quant_method,
        quant_config=quant_config,
        device=device,
        quantize=args.quantize,
        **quant_kwargs,
    )
    # print(model)

    evaluator = LMEvaluator(model=model, device=device, n_samples=128)
    eval_kwargs = {
        "seq_len": 512,
        "batch_size": 1,
        "is_check_sparsity": args.prune,
    }
    results = evaluator.eval(tasks="ppl", **eval_kwargs)
    print(results)
