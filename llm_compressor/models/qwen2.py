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
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2ForCausalLM,
    repeat_kv,
    apply_rotary_pos_emb,
)

PATH = Path(__file__).resolve().parents[1]
if str(PATH) not in sys.path:
    sys.path.append(str(PATH))

from utils.general import LOGGER  # noqa: E402
from models.base import CompressForCausalLM  # noqa: E402
from modules.qmatmul import QMatmul  # noqa: E402
from modules.qlinear import QLinear  # noqa: E402
from pruning.magnitude.core import magnitude  # noqa: E402
from quantization.calibrations.rtn.core import rtn  # noqa: E402
from quantization.calibrations.awq.core import awq  # noqa: E402
from quantization.calibrations.gptq.core import gptq  # noqa: E402
from quantization.calibrations.awq_plus.core import awq_plus  # noqa: E402
from quantization.calibrations.spinquant.core import spinquant  # noqa: E402


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


class QuantQwen2Attention(Qwen2Attention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        attention: Qwen2Attention,
        quant_config,
    ):
        super().__init__(
            attention.config,
            attention.layer_idx,
        )
        self.quant_config = quant_config
        self.qk_matmul = QMatmul(self.quant_config, axes=-2)  # Q@K.T - column-wise
        self.sv_matmul = QMatmul(self.quant_config, axes=-1)  # S@V - row-wise
        self.q_proj = attention.q_proj
        self.k_proj = attention.k_proj
        self.v_proj = attention.v_proj
        self.o_proj = attention.o_proj
        self.layer_types = [
            "sliding_attention"
            if attention.config.sliding_window is not None
            and i >= attention.config.max_window_layers
            else "full_attention"
            for i in range(attention.config.num_hidden_layers)
        ]
        self.sliding_window = (
            attention.config.sliding_window
            if self.layer_types[attention.layer_idx] == "sliding_attention"
            else None
        )

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
            sliding_window=self.sliding_window,  # main diff with Llama
            qk_matmul=self.qk_matmul,
            sv_matmul=self.sv_matmul,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class CompressQwen2ForCausalLM(Qwen2ForCausalLM, CompressForCausalLM):
    def __init__(
        self,
        config,
    ):
        super().__init__(config)

    def _prepare_attention_module(self, quant_config):
        for name, module in self.named_modules():
            if isinstance(module, Qwen2Attention):
                parent, child_name = name.rsplit(".", 1)
                parent_module = dict(self.named_modules())[parent]
                qattn = QuantQwen2Attention(
                    attention=getattr(parent_module, child_name),
                    quant_config=quant_config.matmul,
                )
                setattr(parent_module, child_name, qattn)

        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if "lm_head" not in name:
                    parent, child_name = name.rsplit(".", 1)
                    parent_module = dict(self.named_modules())[parent]
                    qlinear = QLinear(
                        linear=getattr(parent_module, child_name),
                        quant_config=quant_config.linear,
                        dtype=self.dtype,
                    )
                    setattr(parent_module, child_name, qlinear)
                if "lm_head" in name:
                    head_module = dict(self.named_modules())[name]
                    qlinear = QLinear(
                        linear=head_module,
                        quant_config=quant_config.head,
                        dtype=self.dtype,
                    )
                    setattr(self, name, qlinear)

    def quantize(self, tokenizer, quant_method, quant_config, device, **kwargs):
        if kwargs.get("quantize"):
            self._prepare_attention_module(quant_config)

            if quant_method == "rtn":
                rtn(self, device, mse=False, verbose=True)

            elif quant_method == "awq":
                n_samples = kwargs.get("n_samples", 128)
                seq_len = kwargs.get("seq_len", 2048)
                awq(
                    self,
                    device,
                    tokenizer,
                    n_samples=n_samples,
                    seq_len=seq_len,
                    verbose=True,
                )
            elif quant_method == "gptq":
                n_samples = kwargs.get("n_samples", 128)
                seq_len = kwargs.get("seq_len", 2048)
                gptq(
                    self,
                    device,
                    n_samples=n_samples,
                    seq_len=seq_len,
                    verbose=True,
                )
            elif quant_method == "awq_plus":
                n_samples = kwargs.get("n_samples", 128)
                seq_len = kwargs.get("seq_len", 2048)
                awq_plus(
                    self,
                    device,
                    tokenizer,
                    n_samples=n_samples,
                    seq_len=seq_len,
                    verbose=True,
                )
            elif quant_method == "spinquant":
                n_samples = kwargs.get("n_samples", 128)
                seq_len = kwargs.get("seq_len", 2048)
                spinquant(
                    self,
                    device,
                    tokenizer,
                    n_samples=n_samples,
                    seq_len=seq_len,
                    mse=False,
                    verbose=True,
                )
        else:
            return

    def prune(self, tokenizer, prune_method, prune_config, device, **kwargs):
        if kwargs.get("prune"):
            sparsity_ratio = prune_config.pop("sparsity_ratio")

            if prune_method == "magnitude":
                magnitude(self, device, sparsity_ratio)
                return
        else:
            return

    def save_compressed(self, local_save_path):
        LOGGER.info("Saving compressed model...")
        base_model_path = self.config._name_or_path.rstrip(os.sep)

        with init_empty_weights():
            tokenizer = AutoTokenizer.from_pretrained(base_model_path)
            base_model = Qwen2ForCausalLM.from_pretrained(
                base_model_path, low_cpu_mem_usage=True
            )

        compressed_sd = {}
        for k, v in self.state_dict().items():
            compressed_sd[k] = v

        Qwen2ForCausalLM.save_pretrained(
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
    from easydict import EasyDict
    from evaluation.eval import LMEvaluator

    group_size = -1
    device = torch.device("cuda:0")
    quant_config = EasyDict({})
    quant_config.linear = EasyDict({})
    quant_config.linear.weight = {
        "type": "int",
        "format": "int4",
        "group_size": group_size,
        "axes": -1,
        "zero_point": False,
        "device": device,
    }
    quant_config.linear.act_in = {
        "type": "int",
        "format": "int8",
        "group_size": group_size,
        "axes": -1,
        "zero_point": False,
        "device": device,
    }
    quant_config.linear.act_out = {
        "type": "int",
        "format": "int8",
        "group_size": group_size,
        "axes": -1,
        "zero_point": False,
        "device": device,
    }

    quant_config.matmul = EasyDict({})
    quant_config.matmul.act_in = {
        "type": "int",
        "format": "int8",
        "group_size": group_size,
        "axes": -1,
        "zero_point": False,
        "device": device,
    }
    quant_config.matmul.act_out = {
        "type": "int",
        "format": "int8",
        "group_size": group_size,
        "axes": -1,
        "zero_point": False,
        "device": device,
    }

    quant_config.head = EasyDict({})
    quant_config.head.weight = {
        "type": "int",
        "format": "int8",
        "group_size": group_size,
        "axes": -1,
        "zero_point": False,
        "device": device,
    }
    quant_config.head.act_in = {
        "type": None,
        "format": None,
        "group_size": None,
        "axes": None,
        "zero_point": None,
        "device": None,
    }
    quant_config.head.act_out = {
        "type": None,
        "format": None,
        "group_size": None,
        "axes": None,
        "zero_point": None,
        "device": None,
    }

    model_path = "d:\\models\\qwen2.5-0.5b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = CompressQwen2ForCausalLM.from_pretrained(
        model_path,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    quant_kwargs = {
        "n_samples": 128,
        "seq_len": 512,
    }
    model.quantize(
        tokenizer=tokenizer,
        quant_method="spinquant",
        quant_config=quant_config,
        device=device,
        quantize=True,
        **quant_kwargs,
    )
    print(model)

    evaluator = LMEvaluator(model=model, device=device, n_samples=128)
    eval_kwargs = {
        "tokenizer_path": model_path,
        "seq_len": 512,
        "batch_size": 1,
        "check_sparsity": False,
    }
    results = evaluator.eval(tasks="ppl", **eval_kwargs)
    print(results)
