import sys
import functools
from pathlib import Path
from collections import defaultdict

import torch
from torch import nn
from tqdm import tqdm

PATH = Path(__file__).resolve().parents[3]
if str(PATH) not in sys.path:
    sys.path.append(str(PATH))

from utils.general import LOGGER  # noqa: E402
from utils.dataset import get_calib_dataset  # noqa: E402
from utils.torch_utils import cleanup_memory  # noqa: E402
from quantization.calibrations.utils import find_layers  # noqa: E402


def awq(model, device, tokenizer, n_samples=512, seq_len=2048):
    LOGGER.info("Quantizing model... [Quant-method : AWQ]")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.get_layers()

    samples = get_calib_dataset(
        data="pileval", tokenizer=tokenizer, n_samples=n_samples, block_size=seq_len
    )
    samples = torch.cat(samples, dim=0)

    inps = []
    layer_kwargs = {}

    layers[0] = layers[0].to(device)
    model.embed_tokens.to(device)
    if hasattr(model, "embed_positions"):
        model.embed_positions.to(device)

    # get input and kwargs to layer 0
    # with_kwargs is only supported in PyTorch 2.0
    # use this Catcher hack for now
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp)
            layer_kwargs.update(kwargs)
            raise ValueError  # early exit to break later inference
    
    # patch layer 0 to catch input and kwargs
    layers[0] = Catcher(layers[0])
    try:
        model(samples.to(device))
    except ValueError:  # work with early exit
        pass
    del samples
    layers[0] = layers[0].module  # restore
    inps = inps[0]

    layers[0] = layers[0].cpu()
    model.embed_tokens.cpu()
    if hasattr(model, "embed_positions"):
        model.embed_positions.cpu()
    
    cleanup_memory(verbose=False)

    awq_results = {
        "scale": [],
        "clip": [],
    }

    # solve layer by layer
    pg_bar = tqdm(range(len(layers)))
    for i in pg_bar:
        s = f"Quantizing layer.{i:02}..."
        pg_bar.set_description(s)
        LOGGER.debug(s)

        layer = layers[i].to(device)
        named_linears = find_layers(layer)
        
        # firstly, get input features of all linear layers
        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0]
            x = x.detach().cpu()
            feat_dict[name].append(x)

        input_feat = defaultdict(list)
        handles = []
        for name in named_linears:
            handles.append(
                named_linears[name].register_forward_hook(
                    functools.partial(cache_input_hook, name=name, feat_dict=input_feat)
                )
            )
        inps = inps.to(device)  # in case multi-gpu
        # get output as next layer's input
        inps = layer(inps, **layer_kwargs)[0]
        
        for h in handles:
            h.remove()
        # now solve for scaling and clipping
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}
        
        cleanup_memory(verbose=False)

        scales_list = auto_scale_block(
                layer,
                layer_kwargs,
                w_bit=w_bit,
                q_config=q_config,
                input_feat=input_feat,
            )


    #     layers[i] = layer.cpu()
    #     del layer
    #     torch.cuda.empty_cache()

    # model.config.use_cache = use_cache
    # LOGGER.info("Quantization complete !")
    return


if __name__ == "__main__":
    from easydict import EasyDict
    from transformers import AutoTokenizer
    from models.opt import CompressOPTForCausalLM  # noqa: F401
    from models.bloom import CompressBloomForCausalLM  # noqa: F401
    from models.llama import CompressLlamaForCausalLM  # noqa: F401
    from models.phi import CompressPhiForCausalLM  # noqa: F401

    device = torch.device("cuda:0")
    quant_config = EasyDict({})
    quant_config.linear = EasyDict({})
    quant_config.linear.weight = {
        "type": "int",
        "format": "int4",
        "group_size": -1,
        "axes": -1,
        "zero_point": False,
        "device": device,
    }
    quant_config.linear.act_in = {
        "type": "int",
        "format": "int8",
        "group_size": -1,
        "axes": -1,
        "zero_point": False,
        "device": device,
    }
    quant_config.linear.act_out = {
        "type": "int",
        "format": "int8",
        "group_size": -1,
        "axes": -1,
        "zero_point": False,
        "device": device,
    }

    quant_config.matmul = EasyDict({})
    quant_config.matmul.act_in = {
        "type": "int",
        "format": "int8",
        "group_size": -1,
        "axes": -1,
        "zero_point": False,
        "device": device,
    }
    quant_config.matmul.act_out = {
        "type": "int",
        "format": "int8",
        "group_size": -1,
        "axes": -1,
        "zero_point": False,
        "device": device,
    }

    quant_config.head = EasyDict({})
    quant_config.head.weight = {
        "type": "int",
        "format": "int8",
        "group_size": -1,
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

    model_path = "d:\\models\\opt-125m"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = CompressOPTForCausalLM.from_pretrained(
        model_path,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    # model_path = "d:\\models\\bloom-560m"
    # model = CompressLlamaForCausalLM.from_pretrained(
    #     model_path,
    #     attn_implementation="eager",
    #     torch_dtype=torch.bfloat16,
    #     device_map="cpu",
    # )
    # model_path = "d:\\models\\llama-3.2-1b-it"
    # model = CompressLlamaForCausalLM.from_pretrained(
    #     model_path,
    #     attn_implementation="eager",
    #     torch_dtype=torch.bfloat16,
    #     device_map="cpu",
    # )
    # model_path = "d:\\models\\phi-1.5"
    # model = CompressPhiForCausalLM.from_pretrained(
    #     model_path,
    #     attn_implementation="eager",
    #     torch_dtype=torch.bfloat16,
    #     device_map="cpu",
    # )
    model._prepare_attention_module(quant_config)
    awq(model, device, tokenizer, 128, 512)
    # print(model)
