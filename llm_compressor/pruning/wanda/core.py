import os
import sys
from pathlib import Path

import torch
import transformers
from torch import nn
from tqdm import tqdm
from transformers.models.opt.modeling_opt import OPTForCausalLM

PATH = Path(__file__).resolve().parents[3]
if str(PATH) not in sys.path:
    sys.path.append(str(PATH))

from utils.general import LOGGER  # noqa: E402
from utils.dataset import get_loaders  # noqa: E402
from utils.torch_utils import cleanup_memory  # noqa: E402
from utils.module import find_layers  # noqa: E402


@torch.no_grad()
def wanda(model, device, sparsity_ratio, n_samples=512, seq_len=2048, verbose=True):
    if verbose:
        LOGGER.info("Pruning model... [Prune-method : WANDA]")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    model.eval()

    model_name = model.config._name_or_path.rstrip(os.sep).split(os.sep)[-1]
    layers = model.get_layers()

    tokenizer_path = model.config._name_or_path
    dataloader, _ = get_loaders(
        name="c4", tokenizer_path=tokenizer_path, nsamples=n_samples, seqlen=seq_len
    )

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
            inps.append(inp)  # noqa: F821
            layer_kwargs.update(kwargs)
            raise ValueError  # early exit to break later inference

    # patch layer 0 to catch input and kwargs
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    del dataloader, batch
    layers[0] = layers[0].module  # restore
    layers[0] = layers[0].cpu()
    model.move_embed("cpu")
    if isinstance(model, OPTForCausalLM) and "350m" in model_name.lower():
        model.model.decoder.project_in.cpu()

    inps = torch.cat(inps, dim=0)
    outs = torch.zeros_like(inps)
    cleanup_memory(verbose=False)

    pg_bar = tqdm(range(len(layers)), leave=verbose)
    for i in pg_bar:
        s = f"Pruning layer.{i:02}..."
        pg_bar.set_description(s)
        if verbose:
            LOGGER.debug(s)

        layer = layers[i].to(device)
        subset = find_layers(layer)

        for name in subset:
            columns = subset[name].weight.shape[1]
            subset[name].nsamples = 0
            subset[name].scaler_row = torch.zeros((columns), device=device)

        def cache_scalar_row(m, x, y):
            x = x[0]
            x = x.detach()
            if len(x.shape) == 2:
                x = x.unsqueeze(0)
            tmp = x.shape[0]
            if isinstance(m, nn.Linear) or isinstance(m, transformers.Conv1D):
                if len(x.shape) == 3:
                    x = x.reshape((-1, x.shape[-1]))
                x = x.t()

            m.scaler_row *= m.nsamples / (m.nsamples + tmp)
            m.nsamples += tmp
            m.scaler_row += torch.norm(x.float(), p=2, dim=1) ** 2 / m.nsamples

        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(cache_scalar_row))
        for j in range(n_samples):
            layer(inps[j].unsqueeze(0), **layer_kwargs)
        for h in handles:
            h.remove()

        for name in subset:
            W = subset[name].weight.data
            W_metric = torch.abs(W) * torch.sqrt(
                subset[name].scaler_row.reshape((1, -1))
            )
            W_mask = torch.zeros_like(W_metric) == 1

            sort_res = torch.sort(W_metric, dim=-1, stable=True)
            indices = sort_res[1][:, : int(W_metric.shape[1] * sparsity_ratio)]
            W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0

            del subset[name].nsamples, subset[name].scaler_row

        for j in range(n_samples):
            outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]

        layers[i] = layer.cpu()
        del layer
        cleanup_memory(verbose=False)

        inps, outs = outs, inps

    del inps, outs
    cleanup_memory(verbose=False)

    model.config.use_cache = use_cache
    if verbose:
        LOGGER.info("Pruning complete !")
    return
