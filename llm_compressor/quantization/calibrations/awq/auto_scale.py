import sys
from pathlib import Path

import torch

PATH = Path(__file__).resolve().parents[3]
if str(PATH) not in sys.path:
    sys.path.append(str(PATH))

from utils.general import LOGGER  # noqa: E402


@torch.no_grad()
def get_act_scale(x):
    return x.abs().view(-1, x.shape[-1]).mean(0)


@torch.no_grad()
def auto_scale_block(module, module_kwargs, input_feat):

    # find the best scale ratio
    def _search_module_scale(block, linears2scale: list, x, kwargs={}):
        # w: co, ci
        # x: n, ci
        device = next(block.parameters()).device
        x = x.to(device)
        with torch.no_grad():
            org_out = block(x, **kwargs)
            if isinstance(org_out, tuple):
                org_out = org_out[0]

        x_max = get_act_scale(x)
        
        best_error = float("inf")
        best_ratio = -1
        best_scales = None

        n_grid = 20
        history = []
        
        org_sd = {k: v.cpu() for k, v in block.state_dict().items()}
        for ratio in range(n_grid):
            ratio = ratio * 1 / n_grid
            scales = x_max.pow(ratio).clamp(min=1e-4).view(-1)
            scales = scales / (scales.max() * scales.min()).sqrt()
            
            for fc in linears2scale:
                fc.weight.mul_(scales.view(1, -1).to(device))
                fc.weight.data = fc.weight_quantizer((fc.weight.data) / (scales.view(1, -1)))
            
            out = block(x, **kwargs)
            if isinstance(out, tuple):
                out = out[0]
            
            loss = (
                (org_out - out).float().pow(2).mean().item()
            )  # float prevents overflow
            history.append(loss)
            is_best = loss < best_error
            if is_best:
                best_error = loss
                best_ratio = ratio
                best_scales = scales
            block.load_state_dict(org_sd)

        if best_ratio == -1:
            LOGGER.error(f"Loss got infinity. : {history}")
            raise Exception
        
        LOGGER.debug(f"Best ratio : {best_ratio}")
        best_scales = best_scales.view(-1)

        assert torch.isnan(best_scales).sum() == 0, best_scales
        return best_scales.detach()