"""The first deliberately narrow implementation-independent numerical profile."""
from __future__ import annotations

from collections.abc import Mapping
from lazybrick.canonical import digest

PROFILE_ID = "lazybrick.affine-u4-g128-f32.v1"


class SemanticError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def descriptor() -> dict[str, object]:
    return {
        "profile": PROFILE_ID, "version": "1", "rank": 2,
        "orientation": "output_by_input", "group_axis": 1, "group_size": 128,
        "partial_groups": "reject", "codes": {"min": 0, "max": 15},
        "input_dtype": "binary32", "scale_dtype": "binary32",
        "output_dtype": "binary32", "scale": "supplied_positive_finite",
        "zero_point": "supplied_integer_0_to_15",
        "division": "binary32_round_nearest_ties_even",
        "integer_rounding": "nearest_ties_even_before_zero_point_addition",
        "dequantization": "binary32_round_nearest_ties_even",
        "division_overflow": "saturate", "reconstruction_overflow": "reject",
        "subnormals": "preserve", "reconstructed_zero": "positive",
        "nonfinite_inputs": "reject", "parameter_selection": "out_of_scope",
    }


def profile_digest() -> str:
    return digest(descriptor())


def validate_semantics(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"profile", "profile_digest"}:
        raise SemanticError("invalid_semantics", "semantics requires exactly profile and profile_digest")
    if value["profile"] != PROFILE_ID:
        raise SemanticError("unknown_semantic_profile", "unsupported semantic profile")
    if value["profile_digest"] != profile_digest():
        raise SemanticError("semantic_digest_mismatch", "semantic profile digest does not match its definition")
    return {"profile": PROFILE_ID, "profile_digest": profile_digest()}
