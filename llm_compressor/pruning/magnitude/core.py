import sys
from pathlib import Path

import torch
from tqdm import tqdm

PATH = Path(__file__).resolve().parents[3]
if str(PATH) not in sys.path:
    sys.path.append(str(PATH))

from utils.general import LOGGER  # noqa: E402
from utils.module import find_layers  # noqa: E402
from utils.torch_utils import cleanup_memory  # noqa: E402


@torch.no_grad()
def magnitude(model, device, sparsity_ratio, verbose=True):
    if verbose:
        LOGGER.info("Pruning model... [Prune-method : MAGNITUDE]")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    model.eval()

    layers = model.get_layers()

    pg_bar = tqdm(range(len(layers)), leave=verbose)
    for i in pg_bar:
        s = f"Pruning layer.{i:02}..."
        pg_bar.set_description(s)
        if verbose:
            LOGGER.debug(s)

        layer = layers[i].to(device)
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data
            W_metric = torch.abs(W)
            W_mask = torch.zeros_like(W_metric) == 1

            sort_res = torch.sort(W_metric, dim=-1, stable=True)
            indices = sort_res[1][:, : int(W_metric.shape[1] * sparsity_ratio)]
            W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0

        layers[i] = layer.cpu()
        del layer
        cleanup_memory(verbose=False)

    model.config.use_cache = use_cache
    if verbose:
        LOGGER.info("Pruning complete !")
    return
