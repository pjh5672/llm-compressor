from torch import nn


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=""):
    for m in layers:
        if isinstance(module, m):
            return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res
