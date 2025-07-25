import os
import sys
import platform
import argparse
from pathlib import Path

import torch

PATH = Path(__file__).resolve().parents[1]
if str(PATH) not in sys.path:
    sys.path.append(str(PATH))

from utils.general import (
    PROJECT_NAME,
    LOGGER,
    init_seeds,
    print_args,
    colorstr,
    file_date,
)  # noqa: E402
from utils.torch_utils import select_device  # noqa: E402
from utils.parser import PruneConfigParser, QuantConfigParser  # noqa: E402


def build_parser(root_dir):
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True, help="Path to HF model")

    parser.add_argument("--exp", type=str, default="test", help="Name to project")

    parser.add_argument(
        "--profile", action="store_true", help="Enable to profile model"
    )

    parser.add_argument(
        "--quantize", action="store_true", help="Enable to quantize model"
    )

    parser.add_argument(
        "--quant-method", type=str, default=None, help="Quantization method"
    )

    parser.add_argument(
        "--weight",
        type=str,
        default=None,
        help="""Quantization config for weight, 
        following pattern of [type]-[group_size]-[zero_point]-[quant_wise]. 
        (e.g. 'int4-g[-1]-zp-rw' means int4-asymetric-per_token quant)
        """,
    )

    parser.add_argument(
        "--act-in",
        type=str,
        default=None,
        help="""Quantization config for input activation, 
        following pattern of [type]-[group_size]-[zero_point]-[quant_wise]. 
        (e.g. 'int8-g[-1]-zp-rw' means int8-asymetric-per_token quant)
        """,
    )

    parser.add_argument(
        "--act-out",
        type=str,
        default=None,
        help="""Quantization config for output activation, 
        following pattern of [type]-[group_size]-[zero_point]-[quant_wise]. 
        (e.g. 'int8-g[-1]-zp-rw' means int8-asymetric-per_token quant)
        """,
    )

    parser.add_argument(
        "--head",
        type=str,
        default=None,
        help="""Quantization config for head weight, 
        following pattern of [type]-[group_size]-[zero_point]-[quant_wise]. 
        (e.g. 'int8-g[-1]-zp-rw' means int8-asymetric-per_token quant)
        """,
    )

    parser.add_argument(
        "--rotation-path",
        type=str,
        default=None,
        help="Path to rotation matrix for spinquant",
    )

    parser.add_argument(
        "--w-clip", action="store_true", help="Enable MSE-based weight clipping"
    )

    parser.add_argument(
        "--sq-alpha", type=float, default=0.8, help="Hyp-param for SmoothQuant"
    )

    parser.add_argument("--prune", action="store_true", help="Enable to prune model")

    parser.add_argument("--prune-method", type=str, default=None, help="Prune method")

    parser.add_argument("--sparsity", type=float, default=0.0, help="Sparsity ratio")

    parser.add_argument(
        "--ria-alpha", type=float, default=0.5, help="Hyp-param for RIA"
    )

    parser.add_argument(
        "--calib-num", type=int, default=128, help="Number of calibration dataset"
    )

    parser.add_argument(
        "--save-path", type=str, default=None, help="Path to save compressed model"
    )

    parser.add_argument("--tasks", type=str, default="ppl", help="Evaluation tasks")

    parser.add_argument(
        "--seq-len",
        type=int,
        default=512,
        help="Sequence length for calibration and evaluation",
    )

    parser.add_argument(
        "--batch-size", type=int, default=1, help="Evaluation batch size"
    )

    parser.add_argument(
        "--device", default="0", help="cuda devices, i.e. 0 or 0,1,2,3 or cpu"
    )

    parser.add_argument("--seed", type=int, default=0, help="Inference seed")

    args = parser.parse_args()
    args.exp_dir = root_dir / "experiments" / args.exp
    os.makedirs(args.exp_dir, exist_ok=True)

    LOGGER.info(f"🐯 {colorstr('bright_blue', 'bold', PROJECT_NAME)} 🐯")
    LOGGER.info(f"{colorstr('bright_blue', 'Date')}: {file_date()}")
    LOGGER.info(
        f"{colorstr('bright_blue', 'Version')}: Python-{platform.python_version()} torch-{torch.__version__}"
    )
    init_seeds(args.seed, deterministic=True)
    device = select_device(device=args.device, batch_size=args.batch_size)
    LOGGER.add(args.exp_dir / f"{file_date()}.log", level="DEBUG")

    args.pparser = PruneConfigParser()
    args.prune_config = args.pparser.build_cfg(
        sparsity=args.sparsity,
    )
    args.qparser = QuantConfigParser(profile=args.profile)
    args.quant_config = args.qparser.build_cfg(
        args.weight, args.act_in, args.act_out, args.head
    )
    print_args(
        args=args,
        exclude_keys=("pparser", "qparser", "exp_dir", "quant_config", "prune_config"),
        logger=LOGGER,
    )
    return args, device


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[1]
    args, device = build_parser(root_dir=ROOT)
    print(args)
