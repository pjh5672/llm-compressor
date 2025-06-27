import torch
from torch import nn
from tqdm import tqdm

if __package__:
    from .general import LOGGER  # noqa: E402
    from .torch_utils import cleanup_memory  # noqa: E402
else:
    from general import LOGGER  # noqa: E402
    from torch_utils import cleanup_memory  # noqa: E402


def get_op_name(module, op):
    # get the name of the op relative to the module
    for name, m in module.named_modules():
        if m is op:
            return name
    raise ValueError(f"Cannot find op {op} in module {module}")


def get_op_by_name(module, op_name):
    # get the op by its name relative to the module
    for name, m in module.named_modules():
        if name == op_name:
            return m
    raise ValueError(f"Cannot find op {op_name} in module {module}")


def set_op_by_name(layer, name, new_module):
    levels = name.split(".")
    if len(levels) > 1:
        mod_ = layer
        for l_idx in range(len(levels) - 1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], new_module)
    else:
        setattr(layer, name, new_module)


def append_str_prefix(x, prefix):
    if isinstance(x, str):
        return prefix + x
    elif isinstance(x, tuple):
        return tuple([append_str_prefix(y, prefix) for y in x])
    elif isinstance(x, list):
        return [append_str_prefix(y, prefix) for y in x]
    else:
        return x


def find_layers(module, layers=(nn.Conv2d, nn.Linear), name=""):
    if isinstance(module, layers):
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res


def check_sparsity(model, device):
    LOGGER.info("Checking model sparsity...")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.get_layers()

    count = 0
    total_params = 0
    pg_bar = tqdm(range(len(layers)))
    for i in pg_bar:
        s = f"Checking layer.{i:02}..."
        pg_bar.set_description(s)
        LOGGER.debug(s)

        layer = layers[i].to(device)
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W == 0).sum().item()
            total_params += W.numel()

            sub_count += (W == 0).sum().item()
            sub_params += W.numel()

        LOGGER.debug(f"Layer {i} sparsity : {float(sub_count) / sub_params:.4f}")

        layers[i] = layer.cpu()
        del layer
        cleanup_memory(verbose=False)

    model.config.use_cache = use_cache
    print(f"Model sparsity : {float(count) / total_params:.4f}")
    return


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist(), skip_special_tokens=True)


def chat_template(message):
    prompt = f"""Below is an instruction that describes a task. 
Write a response that appropriately completes the request.

### Instruction:
{message}
"""
    return prompt


def extract_response(response_text, input_text):
    return response_text[len(input_text) :].replace("### Response:", "").strip()
