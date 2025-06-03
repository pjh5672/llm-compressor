from typing import Dict, Union, List, Tuple
from dataclasses import dataclass, field


@dataclass
class QuantConfig:
    quant_method: str = field(default="rtn")
    zero_point: bool = field(default=False)
    w_bit: int = field(default=4)
    a_bit: int = field(default=16)
    q_group_size: Union[int, List[int], Tuple[int]] = field(default=128)
