import os
import sys
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from transformers import AutoTokenizer
from accelerate import init_empty_weights
from transformers.cache_utils import Cache
from transformers.models.bloom.modeling_bloom import (
    BloomAttention,
    BloomForCausalLM,
    dropout_add,
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


class QuantBloomAttention(BloomAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        attention: BloomAttention,
        quant_config,
        **kwargs,
    ):
        config = kwargs.get("config")
        super().__init__(
            config,
            attention.layer_idx,
        )
        self.quant_config = quant_config
        self.qk_matmul = QMatmul(self.quant_config, axes=-2)  # Q@K.T - column-wise
        self.sv_matmul = QMatmul(self.quant_config, axes=-1)  # S@V - row-wise
        self.query_key_value = attention.query_key_value
        self.dense = attention.dense
        self.attention_dropout = attention.attention_dropout

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past: Optional[Cache] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        batch_size, q_length, _ = hidden_states.shape
        fused_qkv = self.query_key_value(
            hidden_states
        )  # [batch_size, seq_length, 3 x hidden_size]
        # 3 x [batch_size, num_heads, seq_length, head_dim]
        query_layer, key_layer, value_layer = self._reshape(fused_qkv)

        if layer_past is not None:
            cache_kwargs = {"cache_position": cache_position}
            key_layer, value_layer = layer_past.update(
                key_layer, value_layer, self.layer_idx, cache_kwargs
            )

        # reshape qkv for further computations
        query_layer = query_layer.reshape(
            batch_size * self.num_heads, -1, self.head_dim
        )
        key_layer = key_layer.reshape(
            batch_size * self.num_heads, -1, self.head_dim
        ).transpose(-1, -2)
        value_layer = value_layer.reshape(
            batch_size * self.num_heads, -1, self.head_dim
        )

        # [batch_size * num_heads, q_length, kv_length]
        attention_scores = (
            alibi * self.beta
            + self.qk_matmul(query_layer, key_layer) * self.inv_norm_factor
        )

        # change view to [batch_size, num_heads, q_length, kv_length]
        attn_weights = attention_scores.view(batch_size, self.num_heads, q_length, -1)
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_layer.shape[-1]]
            attn_weights = attn_weights + causal_mask

        # cast attention scores to fp32, compute scaled softmax and cast back to initial dtype
        attention_probs = torch.nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_layer.dtype)

        # [batch_size, num_heads, q_length, kv_length]
        attention_probs = self.attention_dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # change view [batch_size x num_heads, q_length, kv_length]
        attention_probs_reshaped = attention_probs.view(
            batch_size * self.num_heads, q_length, -1
        )

        # matmul: [batch_size * num_heads, q_length, head_dim]
        context_layer = self.sv_matmul(attention_probs_reshaped, value_layer)

        # change view [batch_size, q_length, num_heads * head_dim]
        context_layer = self._merge_heads(context_layer)

        # aggregate results across tp ranks. See here: https://github.com/pytorch/pytorch/issues/76232
        if self.pretraining_tp > 1 and self.slow_but_exact:
            slices = self.hidden_size / self.pretraining_tp
            output_tensor = torch.zeros_like(context_layer)
            for i in range(self.pretraining_tp):
                output_tensor = output_tensor + torch.nn.functional.linear(
                    context_layer[:, :, int(i * slices) : int((i + 1) * slices)],
                    self.dense.weight[:, int(i * slices) : int((i + 1) * slices)],
                )
        else:
            output_tensor = self.dense(context_layer)

        output_tensor = dropout_add(
            output_tensor, residual, self.hidden_dropout, self.training
        )

        outputs = (output_tensor, layer_past)
        if output_attentions:
            outputs += (attention_probs,)

        return outputs


class CompressBloomForCausalLM(BloomForCausalLM, CompressForCausalLM):
    def __init__(
        self,
        config,
    ):
        super().__init__(config)

    def _prepare_attention_module(self, quant_config):
        for name, module in self.named_modules():
            if isinstance(module, BloomAttention):
                parent, child_name = name.rsplit(".", 1)
                parent_module = dict(self.named_modules())[parent]
                qattn = QuantBloomAttention(
                    attention=getattr(parent_module, child_name),
                    quant_config=quant_config.matmul,
                    config=self.config,
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
                rtn(self, device)

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
            base_model = BloomForCausalLM.from_pretrained(
                base_model_path, low_cpu_mem_usage=True
            )

        compressed_sd = {}
        for k, v in self.state_dict().items():
            compressed_sd[k] = v

        BloomForCausalLM.save_pretrained(
            base_model,
            local_save_path,
            state_dict=compressed_sd,
        )
        tokenizer.save_pretrained(local_save_path)
        LOGGER.info(f"Save complete ! : {local_save_path}")

    def get_layers(self):
        return self.transformer.h

    def get_sequential(self, mode="true"):
        if mode == "true":
            return [
                ["self_attention.query_key_value"],
                ["self_attention.dense"],
                ["mlp.dense_h_to_4h"],
                ["mlp.dense_4h_to_h"],
            ]
        else:
            return [
                [
                    "self_attention.query_key_value",
                    "self_attention.dense",
                    "mlp.dense_h_to_4h",
                    "mlp.dense_4h_to_h",
                ]
            ]

    def move_embed(self, device):
        self.transformer.word_embeddings = self.transformer.word_embeddings.to(device)
        self.transformer.word_embeddings_layernorm = (
            self.transformer.word_embeddings_layernorm.to(device)
        )


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

    model_path = "d:\\models\\bloom-560m"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = CompressBloomForCausalLM.from_pretrained(
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
        quant_method="awq",  # "rtn" / "awq"
        quant_config=quant_config,
        device=device,
        quantize=True,
        **quant_kwargs,
    )
    # print(model)

    evaluator = LMEvaluator(model=model, n_samples=128)
    eval_kwargs = {
        "tokenizer_path": model_path,
        "seq_len": 512,
        "batch_size": 1,
        "check_sparsity": False,
    }
    results = evaluator.eval(tasks="ppl", **eval_kwargs)
    print(results)
