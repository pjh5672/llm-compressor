import os
import re
import argparse


class QConfigParser:
    def __init__(self):
        self.q_config = argparse.Namespace()
        self.linear = {}
        self.matmul = {}

    def build_cfg(self, weight, act_in, act_out):
        self.get_linear(weight, act_in, act_out)
        self.get_matmul(act_in, act_out)
        return self.q_config

    def get_linear(self, weight, act_in, act_out):
        self.linear["weight"] = self.parse_qconfig(weight)
        self.linear["act_in"] = self.parse_qconfig(act_in)
        self.linear["act_out"] = self.parse_qconfig(act_out)
        self.q_config.linear = self.linear

    def get_matmul(self, act_in, act_out):
        self.matmul["act_in"] = self.parse_qconfig(act_in)
        self.matmul["act_out"] = self.parse_qconfig(act_out)
        self.q_config.matmul = self.matmul

    def parse_qconfig(self, s):
        config = {}
        pattern = (
            r"(?P<type>[^-]+)"
            r"-(?P<group>g\[-?\d+(?:,\d+)*\]+)"
            r"-(?:(?P<zp>zp)-)?"
            r"(?P<wise>rw|cw)$"
        )
        m = re.match(pattern, s)
        if m:
            result = m.groupdict()
            config["type"] = result["type"]
            config["group"] = self.parse_group_values(result["group"])
            config["axes"] = -1 if result["wise"] == "rw" else -2
            config["zero_point"] = result["zp"] == "zp"
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


def build_parser(root_dir):
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True, help="Path to HF model")

    parser.add_argument("--exp-name", type=str, default="test", help="Name to project")

    parser.add_argument(
        "--quantize", action="store_true", help="enable to quantize model"
    )

    parser.add_argument(
        "--quant-method", type=str, default="rtn", help="Quantization method"
    )

    parser.add_argument(
        "--weight",
        type=str,
        default="int4-g[-1]-zp-rw",
        help="Quantization config for weight",
    )

    parser.add_argument(
        "--act-in",
        type=str,
        default="int8-g[-1]-zp-rw",
        help="Quantization config for input activation",
    )

    parser.add_argument(
        "--act-out",
        type=str,
        default="int8-g[-1]-zp-rw",
        help="Quantization config for output activation",
    )

    parser.add_argument(
        "--calib-data", type=str, default="wiki2", help="Calibration dataset"
    )

    parser.add_argument(
        "--calib-num", type=int, default=128, help="Number of calibration dataset"
    )

    parser.add_argument("--save", action="store_true", help="save to quantized model")

    parser.add_argument("--tasks", type=str, default=None, help="Evaluation tasks")

    parser.add_argument(
        "--batch-size", type=int, default=8, help="Evaluation batch size"
    )

    parser.add_argument("--seed", type=int, default=15, help="Inference seed")

    args = parser.parse_args()
    args.exp_dir = root_dir / "experiments" / args.exp_name
    args.save_dir = args.exp_dir / "model"

    os.makedirs(args.exp_dir, exist_ok=True)
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    parser = QConfigParser()
    args.q_config = parser.build_cfg(args.weight, args.act_in, args.act_out)
    return args


if __name__ == "__main__":
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]

    args = build_parser(root_dir=ROOT)
    print(args)
