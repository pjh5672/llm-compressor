import torch

if __package__:
    from .base import BaseQuantizer
else:
    from base import BaseQuantizer


class DummyQuantizer(BaseQuantizer):
    def __init__(
        self,
        is_profile=False,
        **kwargs,
    ):
        """
        Dummy Quantizer for passing original tensor
        """
        self.is_profile = is_profile
        op_name = kwargs.get("op_name", None)
        max_limit = kwargs.get("max_limit", None)
        save_path = kwargs.get("save_path", "./")
        self.op_name = op_name if op_name is not None else "None"
        self.max_limit = max_limit
        self.save_path = save_path

        super().__init__(
            op_name=self.op_name,
            max_limit=self.max_limit,
            save_path=self.save_path,
        )

    def configure(self):
        pass

    def find_params(self):
        pass

    def forward(self, x, **kwargs):
        if self.is_profile:
            self.record_maxval(x=x, qdq_x=x)
        return x

    def fake_quantize(self):
        pass

    def extra_repr(self):
        s = "Format: BF16"
        if self.is_profile:
            s += f", Op name: {self.op_name}, "
            s += f"Dynamic range limit: {self.max_limit}"
        return s


if __name__ == "__main__":
    torch.manual_seed(0)

    device = torch.device("cuda")
    x = torch.randn(4, 6).to(device=device)
    print(x)

    quantizer = DummyQuantizer()
    print(quantizer)
    x_dq = quantizer(x)
    print(x_dq)
    print(((x - x_dq) ** 2).mean())
