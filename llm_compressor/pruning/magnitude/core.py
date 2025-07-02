import sys
from pathlib import Path

import torch
from tqdm import tqdm

PATH = Path(__file__).resolve().parents[3]
if str(PATH) not in sys.path:
    sys.path.append(str(PATH))

from utils.general import LOGGER  # noqa: E402
from utils.module import find_layers  # noqa: E402


def magnitude(model, device, sparsity_ratio):
    LOGGER.info("Pruning model... [Prune-method : MAGNITUDE]")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.get_layers()

    pg_bar = tqdm(range(len(layers)))
    for i in pg_bar:
        s = f"Pruning layer.{i:02}..."
        pg_bar.set_description(s)
        LOGGER.debug(s)

        layer = layers[i].to(device)
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data

            W_metric = torch.abs(W)
            thresh = torch.sort(W_metric.flatten())[0][int(W.numel() * sparsity_ratio)]
            W_mask = W_metric <= thresh

            subset[name].weight.data[W_mask] = 0

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache
    LOGGER.info("Pruning complete !")
    return


if __name__ == "__main__":
    from models.llama import CompressLlamaForCausalLM  # noqa: F401

    device = torch.device("cuda:0")

    model_path = "d:\\models\\llama-3.2-1b-it"
    model = CompressLlamaForCausalLM.from_pretrained(
        model_path,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    magnitude(model, device, sparsity_ratio=0.5)
    print(model)
