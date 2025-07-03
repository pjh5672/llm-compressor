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
        eps = torch.finfo(x.dtype).eps
        keys = ("Op Name", "PC99%", "Max", "QDQ(Max)", "SQNR")

        def compute_sqnr(t, qdq_t):
            sqnr_value = (torch.sum(t**2) + eps) / (torch.sum((t - qdq_t) ** 2) + eps)
            return 10 * torch.log10(sqnr_value)

        def extract_percentile(t, q):
            k = round(q * (t.numel() - 1))
            return torch.sort(t.flatten())[0][k].item()

        x_ = x.clone().detach().cpu()
        qdq_x_ = qdq_x.clone().detach().cpu()
        maxval = x_.max().item()
        qdq_maxval = qdq_x_.max().item()
        sqnr_val = compute_sqnr(x_, qdq_x_)
        pc99_val = extract_percentile(x_, 0.99)

        vals = (self.op_name, pc99_val, maxval, qdq_maxval, sqnr_val)

        s = (
            ""
            if self.stats_csv.exists()
            else ((("%46s" + "%14s" * (len(keys) - 1)) % keys) + "\n")
        )
        with open(self.stats_csv, "a") as f:
            f.write(s + (("%46s" + "%14.5g" * (len(vals) - 1)) % vals + "\n"))
