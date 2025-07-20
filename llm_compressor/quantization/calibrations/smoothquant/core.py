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
from utils.module import find_layers  # noqa: E402
from quantization.calibrations.rtn.core import rtn  # noqa: E402
from quantization.calibrations.smoothquant.auto_scale import (
    auto_scale_block,
    apply_scale,
)  # noqa: E402


@torch.no_grad()
def smoothquant(
    model,
    device,
    tokenizer,
    alpha=0.5,
    n_samples=512,
    seq_len=2048,
    mse=False,
    verbose=True,
):
    if verbose:
        LOGGER.info("Smoothing model... [Quant-method : SmoothQuant]")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    model.eval()

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

    # solve layer by layer
    pg_bar = tqdm(range(len(layers)), leave=verbose)
    for i in pg_bar:
        s = f"Smoothing layer.{i:02}..."
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
            input_feat=input_feat,
            model_name=model_name,
        )
        apply_scale(layer, scales_list, device, alpha=alpha)
        layer.cpu()
        del input_feat, layer
        cleanup_memory(verbose=False)

    rtn(model, device, mse=mse, verbose=False)

    model.config.use_cache = use_cache
    if verbose:
        LOGGER.info("Quantization complete !")
    return
