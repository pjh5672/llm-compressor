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
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaForCausalLM,
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
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    qk_matmul = kwargs.get("qk_matmul")
    sv_matmul = kwargs.get("sv_matmul")

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = qk_matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query.dtype
    )
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )
    attn_output = sv_matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class QuantLlamaAttention(LlamaAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        attention: LlamaAttention,
        quant_config,
        **kwargs,
    ):
        super().__init__(
            attention.config,
            attention.layer_idx,
        )
        op_name = kwargs.get("op_name", None)
        save_path = kwargs.get("save_path", "./")

        self.qk_matmul = QMatmul(
            quant_config,
            axes=-1,
            op_name=f"{op_name}.qk_matmul",
            save_path=save_path,
        )
        self.sv_matmul = QMatmul(
            quant_config,
            axes=-2,
            op_name=f"{op_name}.sv_matmul",
            save_path=save_path,
        )
        self.q_proj = attention.q_proj
        self.k_proj = attention.k_proj
        self.v_proj = attention.v_proj
        self.o_proj = attention.o_proj

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

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attention_interface: Callable = eager_attention_forward

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            qk_matmul=self.qk_matmul,
            sv_matmul=self.sv_matmul,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class CompressLlamaForCausalLM(LlamaForCausalLM, CompressForCausalLM):
    def __init__(
        self,
        config,
    ):
        super().__init__(config)

    def _prepare_attention_module(self, quant_config, save_path="./"):
        for name, module in self.named_modules():
            if isinstance(module, LlamaAttention):
                parent, child_name = name.rsplit(".", 1)
                parent_module = dict(self.named_modules())[parent]
                qattn = QuantLlamaAttention(
                    attention=getattr(parent_module, child_name),
                    quant_config=quant_config.matmul,
                    op_name=name.replace("model.", ""),
                    save_path=save_path,
                )
                setattr(parent_module, child_name, qattn)

        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                op_name = name.replace("model.", "")

                if "lm_head" not in name:
                    parent, child_name = name.rsplit(".", 1)
                    parent_module = dict(self.named_modules())[parent]
                    qlinear = QLinear(
                        linear=getattr(parent_module, child_name),
                        quant_config=quant_config.linear,
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
            base_model = LlamaForCausalLM.from_pretrained(
                base_model_path, low_cpu_mem_usage=True
            )

        compressed_sd = {}
        for k, v in self.state_dict().items():
            compressed_sd[k] = v

        LlamaForCausalLM.save_pretrained(
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
                ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
                ["self_attn.o_proj"],
                ["mlp.up_proj", "mlp.gate_proj"],
                ["mlp.down_proj"],
            ]
        else:
            return [
                [
                    "self_attn.q_proj",
                    "self_attn.k_proj",
                    "self_attn.v_proj",
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

    model_path = "d:\\models\\llama-3.2-1b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = CompressLlamaForCausalLM.from_pretrained(
        model_path,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    # if args.profile:
    #     model.profile(
    #         quant_config=quant_config,
    #         device=device,
    #         save_path=args.exp_dir,
    #     )

    quant_kwargs = {
        "n_samples": 128,
        "seq_len": 512,
        "rotation_path": args.rotation_path,
    }
    model = model.quantize(
        tokenizer=tokenizer,
        quant_method=args.quant_method,
        quant_config=quant_config,
        device=device,
        quantize=args.quantize,
        **quant_kwargs,
    )
    # print(model)
   
    # qparser.enable_profile(quant_config)
    # model.profile(
    #     quant_config=quant_config,
    #     device=device,
    #     save_path=args.exp_dir,
    # )
    # print(model)

    evaluator = LMEvaluator(model=model, n_samples=30)
    eval_kwargs = {
        "tokenizer_path": model_path,
        "seq_len": 512,
        "batch_size": 1,
        "check_sparsity": False,
    }
    results = evaluator.eval(tasks="ppl", **eval_kwargs)
    print(results)
