import os
import sys
import random
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from loguru import logger

logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | {message}",
    level="INFO",
)
LOGGER = logger
PROJECT_NAME = "LLM-Compressor"
ROOT = Path(__file__).resolve().parents[1]


def init_seeds(seed=0, deterministic=False):
    """
    Initializes RNG seeds and sets deterministic options if specified.

    See https://pytorch.org/docs/stable/notes/randomness.html
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for Multi-GPU, exception safe
    torch.backends.cudnn.benchmark = False
    if deterministic:  # https://github.com/ultralytics/yolov5/pull/8213
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["PYTHONHASHSEED"] = str(seed)


def print_args(args=None, include_keys=(), exclude_keys=(), logger=None):
    # Print function arguments (optional args dict)
    s = colorstr("bright_red", "\n[Argument Info] \n")
    if len(include_keys):
        s += "".join(
            f"- {k} = {v}\n" for k, v in args.__dict__.items() if k in include_keys
        )
    if len(exclude_keys):
        s += "".join(
            f" - {k} = {v}\n" for k, v in args.__dict__.items() if k not in exclude_keys
        )
    if logger is not None:
        logger.info(s.rstrip("\n"))
    else:
        print(s.rstrip("\n"))


def file_date(path=__file__):
    # Return human-readable file modification date, i.e. '2021-3-26'
    t = datetime.now()
    return f"{t.year:04}-{t.month:02}-{t.day:02}"


def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = (
        input if len(input) > 1 else ("bright_red", "bold", input[0])
    )  # color arguments, string
    colors = {
        "black": "\033[30m",
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bright_black": "\033[90m",
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
        "end": "\033[0m",
        "bold": "\033[1m",
        "underline": "\033[4m",
    }
    return "".join(colors[x] for x in args) + f"{string}" + colors["end"]


def print_eval(result):
    s = "\n"
    s += "=====" * 2 + f" {colorstr('Evaluation Results')} " + "=====" * 2
    s += "\n"
    for k, v in result.items():
        s += f"{k.upper():>20s}{v:>20g}\n"
    s += "=====" * 8
    LOGGER.info(s)
