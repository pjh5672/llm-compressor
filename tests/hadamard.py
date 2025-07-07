import sys
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

PATH = Path(__file__).resolve().parents[1]
if str(PATH) not in sys.path:
    sys.path.append(str(PATH))

from llm_compressor.quantization.calibrations.spinquant.rotation_utils import random_hadamard_matrix
from llm_compressor.utils.general import init_seeds


init_seeds(0)

x = torch.randn(20, 20)
dtype = x.dtype

r = random_hadamard_matrix(size=20, device=torch.device("cpu"))
x_col = x.clone()
x_col[:, 10] *= 20
x_col_rot = (x_col.to(r) @ r).to(dtype)
x_col_inv = (x_col_rot.to(r) @ r.T).to(dtype)

x_row = x.clone()
x_row[10, :] *= 20
x_row_rot = (r @ x_row.to(r)).to(dtype)
x_row_inv = (r.T @ x_row_rot.to(r)).to(dtype)

fig = plt.figure()
fig.suptitle("outlier-reduce with hadamard matrix")

ax = fig.add_subplot(231)
ax.set_title("row-wise outlier", fontsize=9)
_ax = ax.matshow(x_row.abs().numpy(), cmap="cool")
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(_ax, cax=cax)

ax = fig.add_subplot(232)
ax.set_title("hadamard applying", fontsize=9)
_ax = ax.matshow(x_row_rot.abs().numpy(), cmap="cool")
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(_ax, cax=cax)

ax = fig.add_subplot(233)
ax.set_title("hadamard inverse", fontsize=9)
_ax = ax.matshow(x_row_inv.abs().numpy(), cmap="cool")
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(_ax, cax=cax)

ax = fig.add_subplot(234)
ax.set_title("column-wise outlier", fontsize=9)
_ax = ax.matshow(x_col.abs().numpy(), cmap="cool")
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(_ax, cax=cax)

ax = fig.add_subplot(235)
ax.set_title("hadamard applying", fontsize=9)
_ax = ax.matshow(x_col_rot.abs().numpy(), cmap="cool")
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(_ax, cax=cax)

ax = fig.add_subplot(236)
ax.set_title("hadamard inverse", fontsize=9)
_ax = ax.matshow(x_col_inv.abs().numpy(), cmap="cool")
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(_ax, cax=cax)

fig.tight_layout()
fig.savefig(Path("./assets") / f"hadamard_apply.png")
