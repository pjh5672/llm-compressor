import sys
from pathlib import Path

import torch
from tqdm import tqdm

PATH = Path(__file__).resolve().parents[3]
if str(PATH) not in sys.path:
    sys.path.append(str(PATH))

from utils.general import LOGGER  # noqa: E402
from quantization.calibrations.utils import find_layers  # noqa: E402


def check_sparsity(model, device):
    LOGGER.info("Checking model sparsity...")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.get_layers()

    count = 0
    total_params = 0
    pg_bar = tqdm(range(len(layers)))
    for i in pg_bar:
        s = f"Checking layer.{i:02}..."
        pg_bar.set_description(s)
        LOGGER.debug(s)

        layer = layers[i].to(device)
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W == 0).sum().item()
            total_params += W.numel()

            sub_count += (W == 0).sum().item()
            sub_params += W.numel()

        LOGGER.debug(f"Layer {i} sparsity : {float(sub_count) / sub_params:.4f}")

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache
    print(f"Model sparsity : {float(count) / total_params:.4f}")
    return
