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
def rtn(model, device, mse=False, verbose=True):
    if verbose:
        LOGGER.info("Quantizing model... [Quant-method : RTN]")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    model.eval()

    layers = model.get_layers()

    pg_bar = tqdm(range(len(layers)), leave=verbose)
    for i in pg_bar:
        s = f"Quantizing layer.{i:02}..."
        pg_bar.set_description(s)
        if verbose:
            LOGGER.debug(s)

        layer = layers[i].to(device)
        subset = find_layers(layer)

        for name in subset:
            subset[name].weight_quantizer.mse = mse
            W = subset[name].weight.data
            MASK = W != 0
            subset[name].weight.data = subset[name].weight_quantizer(W) * MASK
            del subset[name].weight_quantizer

        layers[i] = layer.cpu()
        del layer
        cleanup_memory(verbose=False)

    model.lm_head.to(device)
    model.lm_head.weight_quantizer.mse = mse
    model.lm_head.weight.data = model.lm_head.weight_quantizer(
        model.lm_head.weight.data
    )
    del model.lm_head.weight_quantizer
    model.lm_head.cpu()
    cleanup_memory(verbose=False)

    model.config.use_cache = use_cache
    if verbose:
        LOGGER.info("Quantization complete !")
    return
