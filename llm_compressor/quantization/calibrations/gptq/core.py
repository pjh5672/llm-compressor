import os
import sys
import math
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
def gptq(model, device, n_samples=512, seq_len=2048, mse=False, verbose=True):
    if verbose:
        LOGGER.info("Updating model... [Quant-method : GPTQ]")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    model.eval()

    model_name = model.config._name_or_path.rstrip(os.sep).split(os.sep)[-1]
    layers = model.get_layers()
    sequential = model.get_sequential(mode="true")

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
            inps.append(inp)
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
        s = f"Updating layer.{i:02}..."
        pg_bar.set_description(s)
        if verbose:
            LOGGER.debug(s)

        layer = layers[i].to(device)
        full = find_layers(layer)

        for names in sequential:
            subset = {n: full[n] for n in names}

            for name in subset:
                columns = subset[name].weight.shape[1]
                subset[name].weight_quantizer.nsamples = 0
                subset[name].weight_quantizer.H = torch.zeros(
                    (columns, columns), device=device
                )

            def cache_hessian_weight(m, x, y):
                x = x[0]
                x = x.detach()
                if len(x.shape) == 2:
                    x = x.unsqueeze(0)
                tmp = x.shape[0]
                if isinstance(m, nn.Linear) or isinstance(m, transformers.Conv1D):
                    if len(x.shape) == 3:
                        x = x.reshape((-1, x.shape[-1]))
                    x = x.t()

                m.weight_quantizer.H *= m.weight_quantizer.nsamples / (
                    m.weight_quantizer.nsamples + tmp
                )
                m.weight_quantizer.nsamples += tmp
                inp = math.sqrt(2 / m.weight_quantizer.nsamples) * x.float()
                m.weight_quantizer.H += inp.matmul(inp.t())

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(cache_hessian_weight))
            for j in range(n_samples):
                layer(inps[j].unsqueeze(0), **layer_kwargs)[0]
            for h in handles:
                h.remove()

            for name in subset:
                update_weight(
                    layer=subset[name],
                    device=device,
                    mse=mse,
                    block_size=128,
                    percdamp=0.01,
                    actorder=True,
                )

        for j in range(n_samples):
            outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]
        layers[i] = layer.cpu()
        del layer
        cleanup_memory(verbose=False)

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    if verbose:
        LOGGER.info("Quantization complete !")
    return


def update_weight(
    layer, device, mse=False, block_size=128, percdamp=0.1, actorder=False
):
    W = layer.weight.data.clone()
    if isinstance(layer, transformers.Conv1D):
        W = W.t()
    W = W.float()

    columns = W.shape[-1]

    if mse:
        layer.weight_quantizer.mse = True
    group_size = layer.weight_quantizer.group_size

    H = layer.weight_quantizer.H
    del layer.weight_quantizer.H
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0

    scales, zeros = layer.weight_quantizer.find_params(W)
    if actorder:
        if group_size in (0, -1):
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)
        else:
            N, K = W.shape
            W = W.reshape(N, K // group_size, group_size)
            perm = torch.argsort(
                torch.diag(H).reshape(-1, group_size).sum(-1), descending=True
            )
            W = W[:, perm, :]
            W = W.reshape(N, K)
            # get scales, zeros for re-ordered W
            scales, zeros = layer.weight_quantizer.find_params(W)
            H = H.reshape(K // group_size, group_size, K // group_size, group_size)
            H = H[perm][..., perm, :]
            H = H.reshape(K, K)
        invperm = torch.argsort(perm)

    Q = torch.zeros_like(W)

    def _adjust_dump(percdamp, H, cols):
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(cols, device=device)
        H[diag, diag] += damp
        return H

    try:
        H = _adjust_dump(percdamp, H, columns)
        H = torch.linalg.cholesky(H)
    except Exception as e:
        LOGGER.info(
            f"{e}, Change damping ratio {percdamp} -> {percdamp * 10} for decomposition"
        )
        H = _adjust_dump(percdamp * 10, H, columns)
        H = torch.linalg.cholesky(H)
    H = torch.cholesky_inverse(H)
    H = torch.linalg.cholesky(H, upper=True)
    Hinv = H

    for i1 in range(0, columns, block_size):
        i2 = min(i1 + block_size, columns)
        count = i2 - i1

        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]

        if group_size in (0, -1):
            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]
                q = layer.weight_quantizer(
                    w.unsqueeze(1), scales=scales, zeros=zeros
                ).flatten()
                Q1[:, i] = q
                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1
        else:
            for i in range(0, count, group_size):
                j = i1 + i
                w = W1[:, i : i + group_size]
                d = Hinv1[i : i + group_size, i : i + group_size]
                s = scales[:, [j // group_size], :]
                z = zeros[:, [j // group_size], :]
                q = layer.weight_quantizer(w, scales=s, zeros=z)
                Q1[:, i : i + group_size] = q
                err1 = (w - q) / torch.diag(d)
                W1[:, i:] -= err1.matmul(Hinv1[i : i + group_size, i:])
                Err1[:, i : i + group_size] = err1

        Q[:, i1:i2] = Q1
        W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

    if actorder:
        if group_size in (0, -1):
            Q = Q[:, invperm]
        else:
            Q = Q.reshape(N, K // group_size, group_size)
            Q = Q[:, invperm, :]
            Q = Q.reshape(N, K)

    if isinstance(layer, transformers.Conv1D):
        Q = Q.t()

    layer.weight.data = Q.reshape(layer.weight.shape).to(layer.weight.data.dtype)

    del Q, H, Hinv, W1, Q1, Err1, Hinv1
    cleanup_memory(verbose=False)
