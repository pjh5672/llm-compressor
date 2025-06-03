from typing import List
from enum import Enum


FP32_EXPONENT_BIAS = 127
FP32_MIN_NORMAL = 2 ** (-FP32_EXPONENT_BIAS + 1)
_FORMAT_CACHE = {}


# Enum for scalar data formats
class ElemFormat(Enum):
    int4 = 1
    int8 = 2
    fp4_e2m1 = 3
    fp8_e4m3 = 4
    fp8_e5m2 = 5
    int32 = 6

    @staticmethod
    def from_str(s):
        assert s is not None, "String elem_format == None"
        s = s.lower()
        if hasattr(ElemFormat, s):
            return getattr(ElemFormat, s)
        else:
            raise Exception("Undefined elem format", s)


def _get_min_norm(ebits):
    """Valid for all float formats"""
    emin = 2 - (2 ** (ebits - 1))
    return 0 if ebits == 0 else 2**emin


def _get_max_norm(ebits, mbits):
    """Valid only for floats that define NaN"""
    assert ebits >= 5, "invalid for floats that don't define NaN"
    emax = 0 if ebits == 0 else 2 ** (ebits - 1) - 1
    return 2**emax * float(2 ** (mbits - 1) - 1) / 2 ** (mbits - 2)


def _get_format_params(fmt):
    """Allowed formats:
    - intX:     X={4, 8}bit
    - fpX:      X={4, 8}bit

    Returns:
      ebits: exponent bits
      mbits: mantissa bits: includes sign and implicit bits
      emax: max normal exponent
      max_norm: max normal number
      min_norm: min normal number
    """

    if type(fmt) is str:
        fmt = ElemFormat.from_str(fmt)

    if fmt in _FORMAT_CACHE:
        return _FORMAT_CACHE[fmt]

    elif fmt == ElemFormat.int4:
        ebits, mbits = 0, 4
        emax = 0

    elif fmt == ElemFormat.int8:
        ebits, mbits = 0, 8
        emax = 0

    elif fmt == ElemFormat.fp4_e2m1:
        ebits, mbits = 2, 3
        emax = 2 ** (ebits - 1)

    elif fmt == ElemFormat.fp8_e4m3:
        ebits, mbits = 4, 5
        emax = 2 ** (ebits - 1)

    elif fmt == ElemFormat.fp8_e5m2:
        ebits, mbits = 5, 4
        emax = 2 ** (ebits - 1) - 1

    elif fmt == ElemFormat.int32:
        ebits, mbits = 0, 32
        emax = 0
    else:
        raise Exception("Unknown element format %s" % fmt)

    if fmt != ElemFormat.fp8_e4m3:
        max_norm = 2**emax * float(2 ** (mbits - 1) - 1) / 2 ** (mbits - 2)
    else:
        max_norm = 2**emax * 1.75  # FP8 has custom max_norm

    min_norm = _get_min_norm(ebits)

    _FORMAT_CACHE[fmt] = (ebits, mbits, emax, max_norm, min_norm)

    return ebits, mbits, emax, max_norm, min_norm


if __name__ == "__main__":

    def cprint(
        fmt: ElemFormat,
        bcfg: List,
    ):
        print(fmt, end=" -> ")
        print(f"bit config: E{bcfg[0]}-M{bcfg[1]}", end=", ")
        print(f"emax: {bcfg[2]}", end=", ")
        print(f"max_norm: {bcfg[3]}, min_norm: {bcfg[4]}")

    cprint(ElemFormat.int4, _get_format_params(ElemFormat.int4))
    cprint(ElemFormat.int8, _get_format_params(ElemFormat.int8))
    cprint(ElemFormat.fp4_e2m1, _get_format_params(ElemFormat.fp4_e2m1))
    cprint(ElemFormat.fp8_e4m3, _get_format_params(ElemFormat.fp8_e4m3))
    cprint(ElemFormat.fp8_e5m2, _get_format_params(ElemFormat.fp8_e5m2))
    cprint(ElemFormat.int32, _get_format_params(ElemFormat.int32))
