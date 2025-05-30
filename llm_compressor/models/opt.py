from .base import CompressBaseForCausalLM
from transformers.models.opt.modeling_opt import OPTForCausalLM, OPTDecoderLayer


class CompressOptForCausalLM(CompressBaseForCausalLM):
    layer_type = "OPTDecoderLayer"
    max_seq_len_key = "max_position_embeddings"

    @staticmethod
    def get_model_layers(model: OPTForCausalLM):
        return model.model.decoder.layers
    
    @staticmethod
    def move_embed(model: OPTForCausalLM, device: str):
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(device)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(device)
    
    
