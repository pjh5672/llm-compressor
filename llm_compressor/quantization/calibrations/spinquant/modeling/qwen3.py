from typing import Union

import sys
from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3MLP,
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3Model,
    Qwen3ForCausalLM,
    BaseModelOutputWithPast,
    logger,
    DynamicCache,
    KwargsForCausalLM,
    repeat_kv,
    apply_rotary_pos_emb,
)

PATH = Path(__file__).resolve().parents[4]
if str(PATH) not in sys.path:
    sys.path.append(str(PATH))

from modules.qlinear import QLinear  # noqa: E402
from modules.qmatmul import QMatmul  # noqa: E402


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


class SpinQwen2MLP(Qwen3MLP):
    def __init__(
        self,
        mlp: Qwen3MLP,
        quant_config,
        dtype,
    ):
        super().__init__(
            mlp.config,
        )
        self.gate_proj = QLinear(
            linear=mlp.gate_proj, quant_config=quant_config.linear, dtype=dtype
        )
        self.up_proj = QLinear(
            linear=mlp.up_proj, quant_config=quant_config.linear, dtype=dtype
        )
        self.down_proj = QLinear(
            linear=mlp.down_proj, quant_config=quant_config.linear, dtype=dtype
        )

    def forward(self, x, R1):
        down_proj = self.down_proj(
            self.act_fn(self.gate_proj(x, R1=R1)) * self.up_proj(x, R1=R1),
            R1=R1,
            transpose=True,
        )
        return down_proj


class SpinQwen3Attention(Qwen3Attention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        attention: Qwen3Attention,
        quant_config,
        dtype,
    ):
        super().__init__(
            attention.config,
            attention.layer_idx,
        )
        self.qk_matmul = QMatmul(quant_config.matmul, axes=-1)
        self.sv_matmul = QMatmul(quant_config.matmul, axes=-2)
        self.q_proj = QLinear(
            linear=attention.q_proj, quant_config=quant_config.linear, dtype=dtype
        )
        self.k_proj = QLinear(
            linear=attention.k_proj, quant_config=quant_config.linear, dtype=dtype
        )
        self.v_proj = QLinear(
            linear=attention.v_proj, quant_config=quant_config.linear, dtype=dtype
        )
        self.o_proj = QLinear(
            linear=attention.o_proj, quant_config=quant_config.linear, dtype=dtype
        )
        self.sliding_window = attention.sliding_window
        self.R2 = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        R1=None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = (
            self.q_proj(hidden_states, R1=R1).view(hidden_shape).transpose(1, 2)
        )
        key_states = (
            self.k_proj(hidden_states, R1=R1).view(hidden_shape).transpose(1, 2)
        )
        value_states = (
            self.v_proj(hidden_states, R1=R1, R2=self.R2.weight)
            .view(hidden_shape)
            .transpose(1, 2)
        )

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
        attn_output = self.o_proj(attn_output, R1=R1, R2=self.R2.weight, transpose=True)
        return attn_output, attn_weights


class SpinQwen3DecoderLayer(Qwen3DecoderLayer):
    def __init__(
        self,
        config,
        layer_idx,
        decoder: Qwen3DecoderLayer,
        quant_config,
        dtype,
    ):
        super().__init__(
            config,
            layer_idx,
        )
        self.self_attn = SpinQwen3Attention(decoder.self_attn, quant_config, dtype)
        self.mlp = SpinQwen2MLP(decoder.mlp, quant_config, dtype)
        self.input_layernorm = decoder.input_layernorm
        self.post_attention_layernorm = decoder.post_attention_layernorm

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # necessary, but kept here for BC
        R1=None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            R1=R1,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, R1=R1)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class SpinQwen3Model(Qwen3Model):
    def __init__(
        self,
        model: Qwen3Model,
        quant_config,
        dtype,
    ):
        super().__init__(model.config)

        self.embed_tokens = model.embed_tokens
        self.layers = nn.ModuleList(
            [
                SpinQwen3DecoderLayer(
                    model.config, idx, self.layers[idx], quant_config, dtype
                )
                for idx in range(len(self.layers))
            ]
        )
        self.norm = model.norm

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        R1=None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError(
                "The `past_key_values` should be either a `Cache` object or `None`."
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if R1 is not None:
            dtype = inputs_embeds.dtype
            inputs_embeds = (inputs_embeds.to(torch.float64) @ R1.to(torch.float64)).to(
                dtype
            )
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values,
            output_attentions,
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                R1=R1,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class SpinQwen3ForCausalLM(Qwen3ForCausalLM):
    def __init__(
        self,
        config,
    ):
        super().__init__(config)

    def _prepare_model(self, quant_config):
        self.model = SpinQwen3Model(self.model, quant_config, self.dtype)
        self.lm_head = QLinear(self.lm_head, quant_config.head, self.dtype)
        self.R1 = None

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            R1=self.R1.weight,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state

        if self.R1 is not None:
            dtype = hidden_states.dtype
            hidden_states = (
                hidden_states.to(torch.float64) @ self.R1.weight.T.to(torch.float64)
            ).to(dtype)

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
