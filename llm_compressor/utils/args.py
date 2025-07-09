import os
import re
import sys
import platform
import argparse
from pathlib import Path

import torch
from easydict import EasyDict

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


class PruneConfigParser:
    def __init__(self):
        self.prune_config = EasyDict({})

    def build_cfg(self, sparsity, **kwargs):
        self.prune_config.sparsity_ratio = sparsity
        return self.prune_config


class QuantConfigParser:
    def __init__(self, profile=False):
        self.profile = profile
        self.quant_config = EasyDict({})
        self.linear = {}
        self.matmul = {}
        self.head = {}

    def build_cfg(self, weight, act_in, act_out, head):
        self.get_linear(weight, act_in, act_out)
        self.get_matmul(act_in, act_out)
        self.get_head(head)
        return self.quant_config

    def get_linear(self, weight, act_in, act_out):
        self.linear["weight"] = self.parse_config(weight)
        self.linear["act_in"] = self.parse_config(act_in)
        self.linear["act_out"] = self.parse_config(act_out)
        self.quant_config.linear = self.linear

    def get_matmul(self, act_in, act_out):
        self.matmul["act_in"] = self.parse_config(act_in)
        self.matmul["act_out"] = self.parse_config(act_out)
        self.quant_config.matmul = self.matmul

    def get_head(self, head):
        self.head["weight"] = self.parse_config(head)
        self.head["act_in"] = self.parse_config(None)
        self.head["act_out"] = self.parse_config(None)
        self.quant_config.head = self.head

    def define_qtype(self, fmt):
        if "mx" in fmt:
            return "mx"
        elif "nvfp" in fmt:
            return "nvfp"
        elif "fp" in fmt:
            return "fp"
        elif "int" in fmt:
            return "int"
        else:
            raise RuntimeError(f"Invalid format, got {fmt}.")

    def parse_config(self, s):
        config = {}

        if s is None:
            config["type"] = None
            config["format"] = None
            config["group_size"] = None
            config["axes"] = None
            config["zero_point"] = None
            config["is_profile"] = self.profile
            return config

        pattern = (
            r"(?P<format>[^-]+)"
            r"-(?P<group>g\[-?\d+(?:,\d+)*\]+)"
            r"-(?:(?P<zp>zp)-)?"
            r"(?P<wise>rw|cw)$"
        )
        m = re.match(pattern, s)
        if m:
            result = m.groupdict()
            dtype = self.define_qtype(result["format"])
            config["type"] = dtype
            if dtype == "mx":
                config["format"] = result["format"].replace("mx", "")
                config["scale_ebits"] = 8
            elif dtype == "nvfp":
                config["format"] = result["format"].replace("nv", "")
            else:
                config["format"] = result["format"]
            config["group_size"] = self.parse_group_values(result["group"])
            config["axes"] = -1 if result["wise"] == "rw" else -2
            config["zero_point"] = result["zp"] == "zp"
            config["is_profile"] = self.profile
            return config

        raise RuntimeError(f"Cannot update Qconfig. No matched pattern, got {s}.")

    def parse_group_values(self, group_str):
        nums_str = re.search(r"g\[(.*?)\]", group_str)
        if nums_str:
            xs = nums_str.group(1).split(",")
            if len(xs) == 1:
                return int(xs[0])
            else:
                return [int(n) for n in xs]

        raise RuntimeError(f"Invalid group size, got {nums_str}.")

    def disable_profile(self, quant_config):
        quant_config.linear["weight"]["is_profile"] = False
        quant_config.linear["act_in"]["is_profile"] = False
        quant_config.linear["act_out"]["is_profile"] = False
        quant_config.matmul["act_out"]["is_profile"] = False
        quant_config.matmul["act_in"]["is_profile"] = False
        quant_config.head["weight"]["is_profile"] = False
        quant_config.head["act_in"]["is_profile"] = False
        quant_config.head["act_out"]["is_profile"] = False

    def enable_profile(self, quant_config):
        quant_config.linear["weight"]["is_profile"] = True
        quant_config.linear["act_in"]["is_profile"] = True
        quant_config.linear["act_out"]["is_profile"] = True
        quant_config.matmul["act_out"]["is_profile"] = True
        quant_config.matmul["act_in"]["is_profile"] = True
        quant_config.head["weight"]["is_profile"] = True
        quant_config.head["act_in"]["is_profile"] = True
        quant_config.head["act_out"]["is_profile"] = True


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

    parser.add_argument("--prune", action="store_true", help="Enable to prune model")

    parser.add_argument("--prune-method", type=str, default=None, help="Prune method")

    parser.add_argument("--sparsity", type=float, default=0.0, help="Sparsity ratio")

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
        default=2048,
        help="Sequence length for calibration and evaluation",
    )

    parser.add_argument(
        "--batch-size", type=int, default=8, help="Evaluation batch size"
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
    args.prune_config = args.pparser.build_cfg(sparsity=args.sparsity)
    args.qparser = QuantConfigParser(profile=args.profile)
    args.quant_config = args.qparser.build_cfg(
        args.weight, args.act_in, args.act_out, args.head
    )
    print_args(
        args=args, exclude_keys=("qparser", "exp_dir", "quant_config"), logger=LOGGER
    )
    return args, device


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[1]
    args, device = build_parser(root_dir=ROOT)
    print(args)
