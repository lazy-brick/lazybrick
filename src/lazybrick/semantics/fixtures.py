"""Hand-calculated cases, not outputs regenerated from the evaluator."""
from __future__ import annotations
from lazybrick.canonical import digest


def suite() -> list[dict[str, object]]:
    # s=1, z=7. Halfway quotient rounding precedes adding the odd offset.
    inputs = ["3f000000", "3fc00000", "40200000", "bf000000", "bfc00000", "c0200000", "42c80000", "c2c80000"]
    codes = [7, 9, 9, 7, 5, 5, 15, 0]
    bits = ["00000000", "40000000", "40000000", "00000000", "c0000000", "c0000000", "41000000", "c0e00000"]
    return [
        {"id":"ties-offset-saturation", "input":{"input_bits":[inputs+["00000000"]*120],
            "scale_bits":[["3f800000"]], "zero_points":[[7]]},
         "expected":{"codes":[codes+[7]*120], "reconstructed_bits":[bits+["00000000"]*120]}},
        # Row0: x=2 / (1,2) => codes (2,2), recon (2,2).
        # Row1: x=-2 / (.5,1) + (8,9) => (4,7), recon (-2,-2).
        {"id":"group-and-row-boundaries", "input":{"input_bits":[["40000000"]*256,["c0000000"]*256],
            "scale_bits":[["3f800000","40000000"],["3f000000","3f800000"]], "zero_points":[[0,1],[8,9]]},
         "expected":{"codes":[[2]*256,[4]*128+[7]*128], "reconstructed_bits":[["40000000"]*256,["c0000000"]*256]}},
        {"id":"subnormal-and-signed-zero", "input":{"input_bits":[["00000001","00000002","80000000"]+["00000000"]*125],
            "scale_bits":[["00000001"]],"zero_points":[[0]]},
         "expected":{"codes":[[1,2,0]+[0]*125], "reconstructed_bits":[["00000001","00000002","00000000"]+["00000000"]*125]}},
    ]


def suite_digest() -> str:
    return digest({"suite_version":"1", "cases":suite()})
