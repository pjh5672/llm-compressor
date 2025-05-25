from typing import Dict, Optional, List
from dataclasses import dataclass, field


@dataclass
class QuantConfig:
    quant_method: str = field(default='rtn')
    zero_point: bool = field(default=False)
    w_bit: int = field(default=4)
    q_group_size: int = field(default=128)

    @classmethod
    def from_dict(cls, quant_config: Dict = {}):
        if not quant_config:
            quant_config = cls()
        else:
            quant_config = cls(**quant_config)
        return quant_config

    @classmethod
    def from_pretrained(cls, save_dir: str, **kwargs):
        pass

    def to_dict(self):
        return {
            "zero_point": self.zero_point,
            "q_group_size": self.q_group_size,
            "w_bit": self.w_bit,
        }

    def to_transformers_dict(self):
        return {
            "quant_method": self.quant_method,
            "zero_point": self.zero_point,
            "group_size": self.q_group_size,
            "bits": self.w_bit,
        }

    def from_transformers_dict(self, transformers_dict: Dict):
        return {
            "quant_method": transformers_dict.get("quant_method"),
            "zero_point": transformers_dict.get("zero_point"),
            "q_group_size": transformers_dict.get("group_size"),
            "w_bit": transformers_dict.get("bits"),
        }