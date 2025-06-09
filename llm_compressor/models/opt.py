import sys
from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.models.opt.modeling_opt import OPTAttention, OPTForCausalLM

PATH = Path(__file__).resolve().parents[1]
if str(PATH) not in sys.path:
    sys.path.append(str(PATH))
    
from modules.qmatmul import QMatmul  # noqa: E402
from modules.qlinear import QLinear # noqa: E402


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
    attn_weights = qk_matmul(query, key.transpose(-1, -2)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    attn_output = sv_matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class QuantOPTAttention(OPTAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        attention: OPTAttention,
        bit_config=None,
    ):
        super().__init__(
            attention.config,
            attention.layer_idx,
        )
        self.bit_config = bit_config # it is for KV matmul

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        """Input shape: Batch x Time x Channel"""
        bsz, tgt_len, _ = hidden_states.size()

        # Scaling is susceptible to floating point arithmetics' inprecisions
        # which can lead to different results (this is dependent from model
        # to model, e.g. whisper is one such case). We therefore keep the
        # original order of scaling to follow the original implementation
        # and enforce no scaling (1.0) in the attention call below.
        query_states = self.q_proj(hidden_states) * self.scaling
        query_states = query_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)

        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        key_states = key_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)

        if past_key_value is not None:
            # save all key/value_states to cache to be re-used for fast auto-regressive generation
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, {"cache_position": cache_position}
            )

        attention_interface: Callable = eager_attention_forward

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.dropout,
            scaling=1.0,
            qk_matmul = QMatmul(self.bit_config, axes=-2),
            sv_matmul = QMatmul(self.bit_config, axes=-1),
            **kwargs,
        )

        attn_output = attn_output.reshape(bsz, tgt_len, -1).contiguous()
        attn_output = self.out_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class CompressOPTForCausalLM(OPTForCausalLM):
    def __init__(
        self,
        config,
    ):
        super().__init__(config)

    def _prepare_attention_module(self, quant_config):
        for name, module in self.named_modules():
            if isinstance(module, OPTAttention):
                parent, child_name = name.rsplit('.', 1)
                parent_module = dict(model.named_modules())[parent]
                qattn = QuantOPTAttention(
                    attention=getattr(parent_module, child_name),
                    bit_config=quant_config.matmul,
                )
                setattr(parent_module, child_name, qattn)

        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) and "lm_head" not in name:
                parent, child_name = name.rsplit('.', 1)
                parent_module = dict(model.named_modules())[parent]
                qlinear = QLinear(
                    linear=getattr(parent_module, child_name),
                    bit_config=quant_config.linear,
                )
                setattr(parent_module, child_name, qlinear)

    def quantize(self, tokenizer, quant_config, calib_samples, calib_seq_len):
        self._prepare_attention_module(quant_config)

    def prune(self, tokenizer, prune_config, calib_samples, calib_seq_len):
        pass

    def save_compressed(self, local_save_path, **kwargs):
        self.save_pretrained(save_directory=local_save_path, **kwargs)
    

if __name__ == "__main__":
    from easydict import EasyDict
    from transformers import AutoConfig

    quant_config = EasyDict({})
    quant_config.linear = EasyDict({})
    quant_config.linear.weight = {
        "type": "int",
        "format": "int4",
        "group_size": -1,
        "axes": -1,
        "zero_point": False,
    }
    quant_config.linear.act_in = {
        "type": "int",
        "format": "int8",
        "group_size": -1,
        "axes": -1,
        "zero_point": False,
    }
    quant_config.linear.act_out = {
        "type": "int",
        "format": "int8",
        "group_size": -1,
        "axes": -1,
        "zero_point": False,
    }

    quant_config.matmul = EasyDict({})
    quant_config.matmul.act_in = {
        "type": "int",
        "format": "int8",
        "group_size": -1,
        "axes": -1,
        "zero_point": False,
    }
    quant_config.matmul.act_out = {
        "type": "int",
        "format": "int8",
        "group_size": -1,
        "axes": -1,
        "zero_point": False,
    }
    # print(quant_config)
    model_path = "d:\\models\\opt-125m"
    config = AutoConfig.from_pretrained(model_path)
    model = CompressOPTForCausalLM(config)
    model.quantize(None, quant_config, None, None)