"""Canonical encoding rules."""

from __future__ import annotations

import pytest

from lazybrick import canonical_json, digest
from lazybrick.errors import CanonicalizationError


class TestOrdering:
    def test_key_order_does_not_matter(self) -> None:
        assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})

    def test_keys_sort_by_code_point(self) -> None:
        assert canonical_json({"b": 1, "A": 2, "a": 3}) == b'{"A":2,"a":3,"b":1}'

    def test_nested_keys_are_sorted(self) -> None:
        assert canonical_json({"x": {"z": 1, "y": 2}}) == b'{"x":{"y":2,"z":1}}'

    def test_list_order_is_preserved(self) -> None:
        # Sequences are ordered data, not sets.
        assert canonical_json([3, 1, 2]) == b"[3,1,2]"


class TestEncoding:
    def test_no_insignificant_whitespace(self) -> None:
        assert canonical_json({"a": [1, 2]}) == b'{"a":[1,2]}'

    def test_unicode_is_utf8_not_escaped(self) -> None:
        assert canonical_json({"n": "café"}) == '{"n":"café"}'.encode("utf-8")

    def test_booleans_and_null_are_allowed(self) -> None:
        assert canonical_json({"a": True, "b": None}) == b'{"a":true,"b":null}'


class TestFloatRejection:
    def test_float_is_rejected(self) -> None:
        with pytest.raises(CanonicalizationError) as caught:
            canonical_json({"samples": 256.0})

        assert caught.value.codes == ("float_not_allowed",)

    def test_rejection_reports_the_path(self) -> None:
        with pytest.raises(CanonicalizationError) as caught:
            canonical_json({"a": {"b": [0, {"c": 1.5}]}})

        assert caught.value.issues[0].path == "a.b[1].c"

    def test_every_float_is_reported_at_once(self) -> None:
        with pytest.raises(CanonicalizationError) as caught:
            canonical_json({"a": 1.0, "b": 2.0})

        assert len(caught.value.issues) == 2

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_floats_are_rejected(self, value: float) -> None:
        with pytest.raises(CanonicalizationError):
            canonical_json({"x": value})

    def test_integers_are_fine(self) -> None:
        assert canonical_json({"samples": 256}) == b'{"samples":256}'

    def test_decimal_strings_are_the_supported_escape_hatch(self) -> None:
        # This is how measurements are carried; see EvidenceRecord.
        assert canonical_json({"score": "8.4213"}) == b'{"score":"8.4213"}'


class TestUnsupportedValues:
    def test_non_string_key_is_rejected(self) -> None:
        with pytest.raises(CanonicalizationError) as caught:
            canonical_json({1: "a"})

        assert caught.value.codes == ("invalid_key",)

    def test_arbitrary_object_is_rejected(self) -> None:
        with pytest.raises(CanonicalizationError) as caught:
            canonical_json({"x": object()})

        assert caught.value.codes == ("not_serializable",)

    def test_bytes_are_rejected(self) -> None:
        with pytest.raises(CanonicalizationError):
            canonical_json({"x": b"raw"})


class TestDigest:
    def test_digest_is_sha256_hex(self) -> None:
        assert len(digest({"a": 1})) == 64

    def test_equal_values_share_a_digest(self) -> None:
        assert digest({"a": 1, "b": 2}) == digest({"b": 2, "a": 1})

    def test_different_values_differ(self) -> None:
        assert digest({"a": 1}) != digest({"a": 2})

    def test_int_and_string_are_not_the_same_value(self) -> None:
        assert digest({"a": 1}) != digest({"a": "1"})
