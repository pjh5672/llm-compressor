import sys
from pathlib import Path

import torch

PATH = Path(__file__).resolve().parents[3]
if str(PATH) not in sys.path:
    sys.path.append(str(PATH))

from utils.general import LOGGER  # noqa: E402
from quantization.calibrations.rtn.core import rtn  # noqa: E402
from quantization.calibrations.spinquant.rotation_utils import rotate_model  # noqa: E402
from llm_compressor.quantization.calibrations.spinquant.fuse_norm_utils import (
    fuse_layer_norms,
)  # noqa: E402


@torch.no_grad()
def spinquant(model, device, tokenizer, n_samples=512, seq_len=2048, verbose=True):
    if verbose:
        LOGGER.info("Rotating model... [Quant-method : SpinQuant]")

    model.eval()
    use_cache = model.config.use_cache
    model.config.use_cache = False

    fuse_layer_norms(model)
    rotate_model(model, "hadamard", device, verbose=True)
    rtn(model, device, mse=False, verbose=False)
    # gptq(model, device, n_samples, seq_len, verbose=False)

    model.config.use_cache = use_cache
    if verbose:
        LOGGER.info("Quantization complete !")
    return
