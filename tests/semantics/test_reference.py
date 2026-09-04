from copy import deepcopy
from fractions import Fraction
import pytest
from lazybrick.semantics.profile import SemanticError, PROFILE_ID, profile_digest, validate_semantics
from lazybrick.semantics.reference import evaluate, decode_bits, encode_bits
from lazybrick.semantics.fixtures import suite


@pytest.mark.parametrize("case", suite(), ids=lambda c:c["id"])
def test_reference_matches_hand_calculated_cases(case):
    assert evaluate(case["input"]) == case["expected"]


@pytest.mark.parametrize("value,bits", [
    (Fraction(1)+Fraction(1,2**24),"3f800000"),
    (Fraction(1)+Fraction(3,2**24),"3f800002"),
    (Fraction(1,2**150),"00000000"),
    (Fraction(3,2**150),"00000002"),
    (Fraction(-3,2**150),"80000002"),
    (Fraction((2**24)-1,2**150),"00800000"),
])
def test_binary32_rounding_boundaries(value,bits):
    assert encode_bits(value) == bits


@pytest.mark.parametrize("bits", ["00000000","00000001","007fffff","00800000","3f800000","bf800000","7f7fffff"])
def test_binary32_finite_roundtrip(bits):
    assert encode_bits(decode_bits(bits)) == bits


@pytest.mark.parametrize("scale", ["00000000","80000000","bf800000","7f800000","7fc00000"])
def test_invalid_scales_rejected(scale):
    case=deepcopy(suite()[0]["input"]);case["scale_bits"]=[[scale]]
    with pytest.raises(SemanticError): evaluate(case)


def test_boolean_zero_point_and_partial_groups_rejected():
    case=deepcopy(suite()[0]["input"]);case["zero_points"]=[[True]]
    with pytest.raises(SemanticError,match="zero point"):evaluate(case)
    case=deepcopy(suite()[0]["input"]);case["input_bits"][0].pop()
    with pytest.raises(SemanticError,match="divisible"):evaluate(case)


def test_division_overflow_saturates_and_reconstruction_overflow_rejects():
    case={"input_bits":[["3f800000"]*128],"scale_bits":[["00000001"]],"zero_points":[[0]]}
    assert evaluate(case) == {"codes":[[15]*128],"reconstructed_bits":[["0000000f"]*128]}
    case={"input_bits":[["7f7fffff"]*128],"scale_bits":[["7e000000"]],"zero_points":[[0]]}
    with pytest.raises(SemanticError,match="overflow"):evaluate(case)


def test_profile_digest_is_required_and_unknown_fields_rejected():
    assert validate_semantics({"profile":PROFILE_ID,"profile_digest":profile_digest()})
    with pytest.raises(SemanticError,match="digest"):
        validate_semantics({"profile":PROFILE_ID,"profile_digest":"0"*64})
    with pytest.raises(SemanticError):
        validate_semantics({"profile":PROFILE_ID,"profile_digest":profile_digest(),"rounding":"whatever"})
