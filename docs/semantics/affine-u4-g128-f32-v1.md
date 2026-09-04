# Grouped affine U4, G128, binary32 — profile v1

Status: proposed public numerical contract, implemented by a CPU reference.
This is not a claim of AWQ algorithm equivalence, whole-model reproduction, or
hardware performance. The contract applies to supplied scales and zero points.

## Domain and representation

The profile ID is `lazybrick.affine-u4-g128-f32.v1`. Its canonical descriptor
and digest are exported by `lazybrick.semantics.profile`.

Inputs are a nonempty rank-two matrix `W[out_features][in_features]` of finite
IEEE-754 binary32 values. Column count must be a positive multiple of 128. Groups
are contiguous blocks of 128 columns within each row; never cross rows.
Each row/group has one finite strictly positive binary32 scale and one integer
zero point in [0, 15]. Unknown keys, booleans used as integers, missing groups,
ragged matrices, non-finite values and nonpositive scales are rejected.

Fixture values and scales are eight lowercase hexadecimal digits encoding the
32 bits in most-significant-byte-first order. These are logical bit strings,
not a tensor packing format. Negative zero is accepted as input and treated as
zero; reconstructed zero is always positive zero. No decimal-to-float parsing
is hidden in the contract. Production tensor storage formats are unchanged.

## Arithmetic

For each input x and supplied scale s and zero point z:

1. Divide x by s with IEEE binary32 round-to-nearest, ties-to-even.
2. Round the quotient to an integer, nearest with ties-to-even.
3. Add integer z, then clamp to [0, 15], producing logical code q.
4. Compute (q-z)*s and round to binary32, nearest with ties-to-even.

The integer offset is added after rounding the quotient. Fusing it into
`round(x/s + z)` changes some halfway cases and is not conforming.
Division overflow saturates according to its sign. Reconstruction overflow is
rejected with `numerical_overflow`; outputs must be finite. Subnormals are
preserved; flush-to-zero is not permitted. For a zero-range input group the
caller still supplies a positive scale; zero-range scale selection is outside
this contract. There is no implicit padding, transpose, dtype cast, clipping
search, activation quantization, or parameter-selection algorithm.

The CPU reference uses exact rational representations of binary32 inputs and
explicit rounding, avoiding dependence on the host floating-point environment.
It is deliberately a small test oracle, not a full-model quantization backend.

## Conformance

The packaged suite has independently calculated expected logical codes and
reconstructed bits. It tests halfway ties, offset order, saturation, negative
values, subnormals, and row/group boundaries. Codes and reconstructed bits must
match exactly; there is no quality-based tolerance or fallback. Finite fixture
coverage is bounded evidence, not a proof for every possible tensor.

Reports bind the profile digest, suite digest, implementation identity,
environment, comparison policy and every case result. A supplied candidate's
passing outputs do not become independent execution evidence merely because
verification succeeds. Reference-oracle checks are explicitly labeled as such.
The report digest must come from an external trusted context; a self-hash is
not authentication. Runtime speed and whole-model accuracy remain separate.

## Recipes and migration

Recipe/plan v0.2 may declare `semantics: {profile, profile_digest}` on a stage.
Legacy v0.1 serialized recipes, plans, artifact-input digests and golden fixtures
remain unchanged. Legacy semantic status is `unspecified`. Semantic declaration
and conformance are separate: declared intent is never a passing badge.

The declared contract participates in new plan/build-input identity. Changing
implementations does not change the profile digest but still changes build-input
identity. The current LLM Compressor AWQ path does not have demonstrated mapping
to this profile. Its planner and adapter reject the declaration rather than
silently adopting binary32 arithmetic or changing the algorithm.

Bundle integration requires the independently reviewed complete-bundle integrity
verifier. If unavailable, conformance bundle verification refuses the operation;
it never falls back to checking only weight files or a report self-hash.

The CPU reference limits each case to 65,536 input elements to bound offline
validation cost. This is an evaluator resource limit, not a model size claim.
