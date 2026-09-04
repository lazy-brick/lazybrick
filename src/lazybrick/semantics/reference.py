"""Small binary32 oracle using exact rationals and explicit ties-even rounding."""
from __future__ import annotations

from fractions import Fraction
import re
from lazybrick.semantics.profile import SemanticError

_HEX = re.compile(r"[0-9a-f]{8}\Z")


def _pow2(exponent: int) -> Fraction:
    return Fraction(1 << exponent) if exponent >= 0 else Fraction(1, 1 << -exponent)


def round_even(value: Fraction) -> int:
    floor, remainder = divmod(value.numerator, value.denominator)
    twice = remainder * 2
    return floor + int(twice > value.denominator or (twice == value.denominator and floor % 2 != 0))


def decode_bits(value: object) -> Fraction:
    if not isinstance(value, str) or _HEX.fullmatch(value) is None:
        raise SemanticError("invalid_binary32", "expected eight lowercase binary32 hex digits")
    bits = int(value, 16)
    exponent, mantissa = (bits >> 23) & 255, bits & ((1 << 23) - 1)
    if exponent == 255:
        raise SemanticError("nonfinite_input", "non-finite binary32 input is forbidden")
    significand = mantissa if exponent == 0 else (1 << 23) + mantissa
    result = significand * _pow2(-149 if exponent == 0 else exponent - 150)
    return -result if bits >> 31 else result


def encode_bits(value: Fraction) -> str:
    sign = 1 << 31 if value < 0 else 0
    value = abs(value)
    if value == 0:
        return "00000000"
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    if value < _pow2(exponent):
        exponent -= 1
    if exponent < -126:
        bits = round_even(value / _pow2(-149))
    else:
        units = round_even(value / _pow2(exponent - 23))
        if units == 1 << 24:
            units >>= 1
            exponent += 1
        if exponent > 127:
            raise SemanticError("numerical_overflow", "binary32 reconstruction overflow")
        bits = ((exponent + 127) << 23) | (units - (1 << 23))
    return f"{bits | (sign if bits else 0):08x}"


def evaluate(case: object) -> dict[str, list[list[object]]]:
    if not isinstance(case, dict) or set(case) != {"input_bits", "scale_bits", "zero_points"}:
        raise SemanticError("invalid_case", "case requires input_bits, scale_bits and zero_points")
    rows, scales, zeros = case["input_bits"], case["scale_bits"], case["zero_points"]
    if not isinstance(rows, list) or not rows or not all(isinstance(row, list) for row in rows):
        raise SemanticError("invalid_shape", "input must be a nonempty rank-two matrix")
    width = len(rows[0])
    if not width or width % 128 or any(len(row) != width for row in rows):
        raise SemanticError("invalid_shape", "rows must have equal width divisible by 128")
    if len(rows) * width > 65536:
        raise SemanticError("reference_limit_exceeded", "reference is limited to 65536 elements per case")
    groups = width // 128
    for values in (scales, zeros):
        if not isinstance(values, list) or len(values) != len(rows) or any(not isinstance(row, list) or len(row) != groups for row in values):
            raise SemanticError("invalid_qparams", "one scale and zero point per row/group is required")
    codes, reconstructed = [], []
    for row_index, row in enumerate(rows):
        decoded_scales = [decode_bits(s) for s in scales[row_index]]
        if any(s <= 0 for s in decoded_scales):
            raise SemanticError("invalid_scale", "scale must be finite and strictly positive")
        if any(type(z) is not int or not 0 <= z <= 15 for z in zeros[row_index]):
            raise SemanticError("invalid_zero_point", "zero point must be an integer from 0 to 15")
        row_codes, row_bits = [], []
        for column, bits in enumerate(row):
            x = decode_bits(bits)
            scale, z = decoded_scales[column // 128], zeros[row_index][column // 128]
            ratio = x / scale
            # These bounds imply saturation for every supported zero point,
            # including overflowing quotients, so no host infinity is needed.
            if ratio >= 16:
                q = 15
            elif ratio <= -16:
                q = 0
            else:
                quotient = decode_bits(encode_bits(ratio))
                q = min(15, max(0, round_even(quotient) + z))
            row_codes.append(q)
            row_bits.append(encode_bits((q - z) * scale))
        codes.append(row_codes)
        reconstructed.append(row_bits)
    return {"codes": codes, "reconstructed_bits": reconstructed}
