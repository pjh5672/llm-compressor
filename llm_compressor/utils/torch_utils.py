import os
import platform

import torch

if __package__:
    from .general import LOGGER, PROJECT_NAME, file_date, colorstr
else:
    from general import LOGGER, PROJECT_NAME, file_date, colorstr


def select_device(device="", batch_size=0, newline=True):
    # device = None or 'cpu' or 0 or '0' or '0,1,2,3'
    s = (
        f"{colorstr('bright_magenta', 'bold', PROJECT_NAME)} "
        + f"🚀 {file_date()} Python-{platform.python_version()} torch-{torch.__version__} "
    )
    device = (
        str(device).strip().lower().replace("cuda:", "").replace("none", "")
    )  # to string, 'cuda:0' to '0'
    cpu = device == "cpu"
    if cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif device:
        os.environ["CUDA_VISIBLE_DEVICES"] = (
            device  # set environment variable - must be before assert is_available()
        )
        assert torch.cuda.is_available() and torch.cuda.device_count() >= len(
            device.replace(",", "")
        ), (
            f"Invalid CUDA '--device {device}' requested, use '--device cpu' or pass valid CUDA device(s)"
        )

    if not cpu and torch.cuda.is_available():  # prefer GPU if available
        devices = (
            device.split(",") if device else "0"
        )  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # device count
        if n > 1 and batch_size > 0:  # check batch_size is divisible by device_count
            assert batch_size % n == 0, (
                f"batch-size {batch_size} not multiple of GPU count {n}"
            )
        space = " " * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)\n"  # bytes to MB
        arg = "cuda:0"
    else:  # revert to CPU
        s += "CPU\n"
        arg = "cpu"

    if not newline:
        s = s.rstrip()
    LOGGER.info(s)
    return torch.device(arg)
