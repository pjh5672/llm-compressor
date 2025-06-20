import os
import sys
import functools
from pathlib import Path
from collections import defaultdict

import torch
from torch import nn
from tqdm import tqdm
from transformers.models.opt.modeling_opt import OPTForCausalLM

PATH = Path(__file__).resolve().parents[3]
if str(PATH) not in sys.path:
    sys.path.append(str(PATH))

from utils.general import LOGGER  # noqa: E402
from utils.dataset import get_calib_dataset  # noqa: E402
from utils.torch_utils import cleanup_memory  # noqa: E402
from utils.module import find_layers, get_op_name, append_str_prefix  # noqa: E402
from quantization.calibrations.gptq.core import gptq  # noqa: E402
from quantization.calibrations.awq.auto_scale import auto_scale_block, apply_scale  # noqa: E402
from quantization.calibrations.awq.auto_clip import auto_clip_block, apply_clip  # noqa: E402


@torch.no_grad()
def awq_plus(model, device, tokenizer, n_samples=512, seq_len=2048, verbose=True):
    if verbose:
        LOGGER.info("Calibrating model... [Quant-method : AWQ_PLUS]")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    model.eval()

    orig_state_dict = model.state_dict()
    model_name = model.config._name_or_path.rstrip(os.sep).split(os.sep)[-1]
    layers = model.get_layers()

    samples = get_calib_dataset(
        data="pileval", tokenizer=tokenizer, n_samples=n_samples, block_size=seq_len
    )
    samples = torch.cat(samples, dim=0)

    inps = []
    layer_kwargs = {}

    layers[0] = layers[0].to(device)
    model.move_embed(device)
    if isinstance(model, OPTForCausalLM) and "350m" in model_name.lower():
        model.model.decoder.project_in.to(device)

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
    model.move_embed("cpu")
    if isinstance(model, OPTForCausalLM) and "350m" in model_name.lower():
        model.model.decoder.project_in.cpu()

    cleanup_memory(verbose=False)

    awq_results = {
        "scale": [],
        "clip": [],
    }

    # solve layer by layer
    pg_bar = tqdm(range(len(layers)), leave=verbose)
    for i in pg_bar:
        s = f"Calibrating layer.{i:02}..."
        pg_bar.set_description(s)
        if verbose:
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
            input_feat=input_feat,
            model_name=model_name,
        )
        apply_scale(layer, scales_list, device, input_feat_dict=input_feat)

        # append prefix to make names global
        awq_results["scale"] += append_str_prefix(
            scales_list, get_op_name(model, layer) + "."
        )
        cleanup_memory(verbose=False)

        clip_list = auto_clip_block(layer, input_feat=input_feat, device=device)
        apply_clip(layer, clip_list, device=device)

        # append prefix to make names global
        awq_results["clip"] += append_str_prefix(
            clip_list, get_op_name(model, layer) + "."
        )

        layer.cpu()
        del input_feat, layer
        cleanup_memory(verbose=False)

    LOGGER.info("Applying AWQ results into model...")
    model.load_state_dict(orig_state_dict)
    apply_scale(model, awq_results["scale"], device)
    apply_clip(model, awq_results["clip"], device)
    gptq(model, device, n_samples=n_samples, seq_len=seq_len, mse=True, verbose=False)

    model.config.use_cache = use_cache
    if verbose:
        LOGGER.info("Quantization complete !")
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
    model = CompressOPTForCausalLM.from_pretrained(
        model_path,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    # model_path = "d:\\models\\bloom-560m"
    # model = CompressBloomForCausalLM.from_pretrained(
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
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model._prepare_attention_module(quant_config)
    awq_results = awq_plus(model, device, tokenizer, 128, 512)
    print(awq_results)
