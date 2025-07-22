from pathlib import Path

import torch
import torch.nn as nn


class BaseQuantizer(nn.Module):
    def __init__(self, op_name, save_path):
        super().__init__()
        self.op_name = op_name
        if isinstance(save_path, str):
            save_path = Path(save_path)
        self.stats_csv = save_path / "stats.csv"

    def configure(self):
        raise NotImplementedError

    def find_params(self):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError

    def fake_quantize(self):
        raise NotImplementedError

    def extra_repr(self):
        raise NotImplementedError

    def record_stats(self, x, qdq_x, **kwargs):
        qtype = kwargs.get("qtype")
        qformat = kwargs.get("qformat")
        group_size = kwargs.get("group_size")
        zero_point = kwargs.get("zero_point")

        keys = (
            "Op Name",
            "PC99%",
            "Max",
            "QDQ(Max)",
            "SQNR",
            "ClipError",
            "Elem",
            "BPV",
        )

        def compute_sqnr(t, qdq_t):
            t_ = (t - t.min()) / (t.max() - t.min())
            qdq_t_ = (qdq_t - qdq_t.min()) / (qdq_t.max() - qdq_t.min())
            return -10 * torch.log10(torch.mean((t_ - qdq_t_) ** 2) + 1e-10)

        def extract_percentile(t, q):
            k = round(q * (t.numel() - 1))
            return torch.sort(t.flatten())[0][k].item()

        x_ = x.clone().detach().cpu()
        qdq_x_ = qdq_x.clone().detach().cpu()
        maxval = x_.max().item()
        qdq_maxval = qdq_x_.max().item()
        pc99_val = extract_percentile(x_, 0.99)
        sqnr_val = compute_sqnr(x_, qdq_x_)
        clip_err = maxval - qdq_maxval

        if qtype == "DUMMY":
            bpv = 16

        elif qtype == "INT":
            bpv_scales = 16 / group_size
            bpv_zeros = 16 / group_size if zero_point else 0
            bpv_elem = 4 if qformat == "INT4" else 8
            bpv = bpv_elem + bpv_scales + bpv_zeros

        elif qtype == "FP":
            bpv_scales = 16 / group_size
            bpv_zeros = 16 / group_size if zero_point else 0
            bpv_elem = 4 if qformat == "FP4_E2M1" else 8
            bpv = bpv_elem + bpv_scales + bpv_zeros

        elif qtype == "MX":
            bpv_scales = 16 / group_size
            bpv_zeros = 16 / group_size if zero_point else 0
            if qformat == "INT4" or qformat == "FP4_E2M1":
                bpv_elem = 4
            else:
                bpv_elem = 8
            bpv = bpv_elem + bpv_scales + bpv_zeros

        elif qtype == "NV":
            bpv_scales = 16 / x_.numel() + 8 / group_size
            bpv_zeros = 16 / group_size if zero_point else 0
            bpv_elem = 4
            bpv = bpv_elem + bpv_scales + bpv_zeros

        vals = (
            self.op_name,
            pc99_val,
            maxval,
            qdq_maxval,
            sqnr_val,
            clip_err,
            x_.numel(),
            bpv,
        )

        s = (
            ""
            if self.stats_csv.exists()
            else ((("%46s," + "%14s," * (len(keys) - 1)) % keys).rstrip(",") + "\n")
        )
        with open(self.stats_csv, "a") as f:
            f.write(
                s + (("%46s," + "%14.5g," * (len(vals) - 1)) % vals).rstrip(",") + "\n"
            )
