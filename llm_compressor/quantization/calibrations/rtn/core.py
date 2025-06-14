import sys
from pathlib import Path

import torch
from tqdm import tqdm

PATH = Path(__file__).resolve().parents[3]
if str(PATH) not in sys.path:
    sys.path.append(str(PATH))

from utils.general import LOGGER  # noqa: E402
from utils.torch_utils import cleanup_memory  # noqa: E402
from quantization.calibrations.utils import find_layers  # noqa: E402


def rtn(model, device):
    LOGGER.info("Quantizing model... [Quant-method : RTN]")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.get_layers()
    sequential = model.get_sequential(mode="true")

    pg_bar = tqdm(range(len(layers)))
    for i in pg_bar:
        s = f"Quantizing layer.{i:02}..."
        pg_bar.set_description(s)
        LOGGER.debug(s)

        layer = layers[i].to(device)
        full = find_layers(layer)

        for names in sequential:
            subset = {n: full[n] for n in names}

            for name in subset:
                W = subset[name].weight.data
                subset[name].weight.data = subset[name].weight_quantizer(W)
                del subset[name].weight_quantizer

        layers[i] = layer.cpu()
        del layer
        cleanup_memory(verbose=False)

    model.config.use_cache = use_cache
    LOGGER.info("Quantization complete !")
    return


if __name__ == "__main__":
    from easydict import EasyDict
    from models.opt import CompressOPTForCausalLM  # noqa: F401
    from models.bloom import CompressBloomForCausalLM  # noqa: F401
    from models.llama import CompressLlamaForCausalLM  # noqa: F401
    from models.phi import CompressPhiForCausalLM  # noqa: F401

    device = torch.device("cuda:0")
    quant_config = EasyDict({})
    quant_config.linear = EasyDict({})
    quant_config.linear.weight = {
        "type": "int",
        "format": "int4",
        "group_size": -1,
        "axes": -1,
        "zero_point": False,
        "device": device,
    }
    quant_config.linear.act_in = {
        "type": "int",
        "format": "int8",
        "group_size": -1,
        "axes": -1,
        "zero_point": False,
        "device": device,
    }
    quant_config.linear.act_out = {
        "type": "int",
        "format": "int8",
        "group_size": -1,
        "axes": -1,
        "zero_point": False,
        "device": device,
    }

    quant_config.matmul = EasyDict({})
    quant_config.matmul.act_in = {
        "type": "int",
        "format": "int8",
        "group_size": -1,
        "axes": -1,
        "zero_point": False,
        "device": device,
    }
    quant_config.matmul.act_out = {
        "type": "int",
        "format": "int8",
        "group_size": -1,
        "axes": -1,
        "zero_point": False,
        "device": device,
    }

    # model_path = "d:\\models\\opt-125m"
    # model = CompressOPTForCausalLM.from_pretrained(
    #     model_path,
    #     attn_implementation="eager",
    #     torch_dtype=torch.bfloat16,
    #     device_map="cpu",
    # )
    # model_path = "d:\\models\\bloom-560m"
    # model = CompressLlamaForCausalLM.from_pretrained(
    #     model_path,
    #     attn_implementation="eager",
    #     torch_dtype=torch.bfloat16,
    #     device_map="cpu",
    # )
    # model_path = "d:\\models\\llama-3.2-1b-it"
    # model = CompressLlamaForCausalLM.from_pretrained(
    #     model_path,
    #     attn_implementation="eager",
    #     torch_dtype=torch.bfloat16,
    #     device_map="cpu",
    # )
    model_path = "d:\\models\\phi-1.5"
    model = CompressPhiForCausalLM.from_pretrained(
        model_path,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    model._prepare_attention_module(quant_config)
    rtn(model, device)
    print(model)
