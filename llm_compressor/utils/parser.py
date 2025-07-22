import re
from copy import deepcopy

from easydict import EasyDict


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
        self.mpq = EasyDict({})
        self.mpq.layers = {}

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

    def register_4_to_8bit_config(self, layer_names):
        for name in layer_names:
            if "weight" in name:
                linear_config = deepcopy(self.linear)
                fmt = linear_config["weight"]["format"]
                if fmt is not None:
                    if fmt.startswith("int"):
                        linear_config["weight"]["format"] = fmt.replace(fmt, "int8")
                    elif fmt.startswith("fp4"):
                        linear_config["weight"]["format"] = fmt.replace(fmt, "fp8_e4m3")
                    self.mpq.layers.update({name.rstrip(".weight"): linear_config})

    def register_8_to_4bit_config(self, layer_names):
        for name in layer_names:
            if "matmul" in name:
                matmul_config = deepcopy(self.matmul)

                if "input" in name:
                    m_name = name.rstrip(".input")
                    if m_name in self.mpq.layers:
                        matmul_config = self.mpq.layers[m_name]
                    fmt = matmul_config["act_in"]["format"]
                    if fmt is not None:
                        if fmt.startswith("int"):
                            matmul_config["act_in"]["format"] = fmt.replace(fmt, "int4")
                        elif fmt.startswith("fp8"):
                            matmul_config["act_in"]["format"] = fmt.replace(
                                fmt, "fp4_e2m1"
                            )
                        self.mpq.layers.update({m_name: matmul_config})

                elif "output" in name:
                    m_name = name.rstrip(".output")
                    if m_name in self.mpq.layers:
                        matmul_config = self.mpq.layers[m_name]
                    fmt = matmul_config["act_out"]["format"]
                    if fmt is not None:
                        if fmt.startswith("int"):
                            matmul_config["act_out"]["format"] = fmt.replace(
                                fmt, "int4"
                            )
                        elif fmt.startswith("fp8"):
                            matmul_config["act_out"]["format"] = fmt.replace(
                                fmt, "fp4_e2m1"
                            )
                        self.mpq.layers.update({m_name: matmul_config})
            else:
                linear_config = deepcopy(self.linear)

                if "input" in name:
                    m_name = name.rstrip(".input")
                    if m_name in self.mpq.layers:
                        linear_config = self.mpq.layers[m_name]
                    fmt = linear_config["act_in"]["format"]
                    if fmt is not None:
                        if fmt.startswith("int"):
                            linear_config["act_in"]["format"] = fmt.replace(fmt, "int4")
                        elif fmt.startswith("fp8"):
                            linear_config["act_in"]["format"] = fmt.replace(
                                fmt, "fp4_e2m1"
                            )
                        self.mpq.layers.update({m_name: linear_config})

                elif "output" in name:
                    m_name = name.rstrip(".output")
                    if m_name in self.mpq.layers:
                        linear_config = self.mpq.layers[m_name]
                    fmt = linear_config["act_out"]["format"]
                    if fmt is not None:
                        if fmt.startswith("int"):
                            linear_config["act_out"]["format"] = fmt.replace(
                                fmt, "int4"
                            )
                        elif fmt.startswith("fp8"):
                            linear_config["act_out"]["format"] = fmt.replace(
                                fmt, "fp4_e2m1"
                            )
                        self.mpq.layers.update({m_name: linear_config})

    def register_org_config(self, layer_names):
        for name in layer_names:
            if "matmul" in name:
                matmul_config = deepcopy(self.matmul)

                if "input" in name:
                    m_name = name.rstrip(".input")
                    if m_name in self.mpq.layers:
                        matmul_config = self.mpq.layers[m_name]
                    matmul_config["act_in"]["type"] = None
                    self.mpq.layers.update({m_name: matmul_config})

                elif "output" in name:
                    m_name = name.rstrip(".output")
                    if m_name in self.mpq.layers:
                        matmul_config = self.mpq.layers[m_name]
                    matmul_config["act_out"]["type"] = None
                    self.mpq.layers.update({m_name: matmul_config})
            else:
                linear_config = deepcopy(self.linear)

                if "input" in name:
                    m_name = name.rstrip(".input")
                    if m_name in self.mpq.layers:
                        linear_config = self.mpq.layers[m_name]
                    linear_config["act_in"]["type"] = None
                    self.mpq.layers.update({m_name: linear_config})

                elif "output" in name:
                    m_name = name.rstrip(".output")
                    if m_name in self.mpq.layers:
                        linear_config = self.mpq.layers[m_name]
                    linear_config["act_out"]["type"] = None
                    self.mpq.layers.update({m_name: linear_config})
