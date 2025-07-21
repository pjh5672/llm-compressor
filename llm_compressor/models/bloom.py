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
from modules.qmatmul import QMatmul  # noqa: E402
from modules.qlinear import QLinear  # noqa: E402
from models.base import CompressForCausalLM  # noqa: E402


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

    def _prepare_qmodule(self, quant_config, save_path="./", **kwargs):
        mixed_precision = kwargs.get("mixed_precision")

        for name, module in self.named_modules():
            if isinstance(module, BloomAttention):
                parent, child_name = name.rsplit(".", 1)
                parent_module = dict(self.named_modules())[parent]
                qattn = QuantBloomAttention(
                    attention=getattr(parent_module, child_name),
                    quant_config=quant_config.matmul,
                    config=self.config,
                    op_name=name.replace("transformer.", ""),
                    save_path=save_path,
                    mixed_precision=mixed_precision,
                )
                setattr(parent_module, child_name, qattn)

        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                op_name = name.replace("transformer.", "")

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
    from evaluation.eval import LMEvaluator
    from utils.args import build_parser, QuantConfigParser

    ROOT = Path(__file__).resolve().parents[1]
    args, device = build_parser(ROOT)

    qparser = QuantConfigParser(profile=args.profile)
    quant_config = qparser.build_cfg(args.weight, args.act_in, args.act_out, args.head)

    model_path = "d:\\models\\bloom-560m"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = CompressBloomForCausalLM.from_pretrained(
        model_path,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    prune_kwargs = {
        "n_samples": 128,
        "seq_len": 512,
        "alpha": args.ria_alpha,
    }
    model.prune(
        tokenizer=tokenizer,
        prune_method=args.prune_method,
        prune_config=args.prune_config,
        device=device,
        prune=args.prune,
        **prune_kwargs,
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
        "w_clip": args.w_clip,
        "alpha": args.sq_alpha,
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
