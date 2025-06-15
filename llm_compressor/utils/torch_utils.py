import os

import torch

if __package__:
    from .general import LOGGER, colorstr
else:
    from general import LOGGER, colorstr


def select_device(device="", batch_size=0, newline=False):
    # device = None or 'cpu' or 0 or '0' or '0,1,2,3'
    s = colorstr("Device: ")
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


def cleanup_memory(verbose=True) -> None:
    """Run GC and clear GPU memory."""
    import gc
    import inspect

    caller_name = ""
    try:
        caller_name = f" (from {inspect.stack()[1].function})"
    except (ValueError, KeyError):
        pass

    def total_reserved_mem() -> int:
        return sum(
            torch.cuda.memory_reserved(device=i)
            for i in range(torch.cuda.device_count())
        )

    memory_before = total_reserved_mem()

    # gc.collect and empty cache are necessary to clean up GPU memory if the model was distributed
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_after = total_reserved_mem()
        if verbose:
            LOGGER.debug(
                f"GPU memory{caller_name}: {memory_before / (1024**3):.2f} -> {memory_after / (1024**3):.2f} GB"
                f" ({(memory_after - memory_before) / (1024**3):.2f} GB)"
            )
