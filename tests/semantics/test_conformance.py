from copy import deepcopy
import json
import sys
import pytest
from lazybrick.canonical import digest
from lazybrick.cli import main
from lazybrick.runs import RunIdentity, RunStore
from lazybrick.semantics.conformance import reference_report, verify_report, context_for, run_conformance, verify_bundle_conformance
from lazybrick.semantics.profile import SemanticError


def verify(report, context=None, expected=None):
    return verify_report(report, expected_digest=expected or digest(report),expected_context=context or context_for(report))


def test_reference_evidence_is_scoped_and_verified():
    report=reference_report()
    assert verify(report)["status"] == "passed"
    assert report["execution_kind"] == "reference_evaluator"
    assert report["binding"] is None


@pytest.mark.parametrize("field,value", [("profile_digest","a"*64),("suite_digest","b"*64),("comparison",{"codes":"approximate","reconstructed_bits":"exact"})])
def test_contract_changes_rejected_even_after_rehash(field,value):
    report=reference_report();report[field]=value
    with pytest.raises(SemanticError):verify(report)


@pytest.mark.parametrize("field", ["implementation","environment","binding"])
def test_substituted_context_rejected(field):
    report=reference_report();context=context_for(report)
    if field=="implementation": report[field]["source_digest"]="c"*64
    elif field=="environment":report[field]["python"]="different"
    else:
        report[field]={"identity":{"recipe_digest":"a"*64,"plan_digest":"b"*64,"artifact_id":"c"*64,"run_id":"r","attempt_id":"a"},"stage_id":"q","stage_digest":"d"*64}
    with pytest.raises(SemanticError,match="context"):verify(report,context)


def test_altered_or_missing_case_cannot_keep_passed_status():
    report=reference_report();report["cases"][0]["output"]["codes"][0][0]=1
    with pytest.raises(SemanticError,match="status"):verify(report)
    report=reference_report();report["cases"].pop()
    with pytest.raises(SemanticError,match="missing"):verify(report)


def test_external_digest_catches_complete_replacement():
    report=reference_report();anchor=digest(report)
    report["environment"]["python"]="modified"
    with pytest.raises(SemanticError,match="external digest"):verify(report,expected=anchor)


def test_wrong_candidate_and_exceptions_are_not_passes():
    ref=reference_report()
    for candidate in (lambda case:{"codes":[[True]],"reconstructed_bits":[["00000000"]]},lambda case:1/0):
        report=run_conformance(candidate,implementation=ref["implementation"],environment=ref["environment"])
        assert verify(report)["status"] == "failed"


def test_noncallable_candidate_is_rejected_before_cases():
    ref = reference_report()
    with pytest.raises(SemanticError, match="callable"):
        run_conformance(None, implementation=ref["implementation"], environment=ref["environment"])


def test_bundle_requires_external_integrity_anchor(tmp_path):
    with pytest.raises(SemanticError, match="SHA-256"):
        verify_bundle_conformance(tmp_path, expected_bundle_digest=None,
            expected_report_digest="a"*64, expected_context={})


def test_bundle_integration_refuses_missing_integrity_layer(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "lazybrick.runs.bundle", None)
    with pytest.raises(SemanticError,match="integrity verifier"):
        verify_bundle_conformance(tmp_path,expected_bundle_digest="a"*64,expected_report_digest="b"*64,expected_context={})


def test_cli_reference_and_external_verification(tmp_path,capsys):
    path=tmp_path/"report.json"
    assert main(["conformance","reference","--output",str(path)]) == 0
    anchor=json.loads(capsys.readouterr().out)
    ctx=tmp_path/"context.json";ctx.write_text(json.dumps(anchor["context"]))
    assert main(["conformance","verify",str(path),"--expected-digest",anchor["report_digest"],"--context",str(ctx)]) == 0
    capsys.readouterr()
    assert main(["conformance","reference","--output",str(path)]) == 5


def test_bundle_binding_after_integrity_gate(tmp_path, monkeypatch):
    # Exercise only this module's integration contract. The dependency owns its
    # separate all-file integrity tests; this stub is not integrity evidence.
    import sys
    from types import ModuleType
    from lazybrick.semantics.profile import PROFILE_ID, profile_digest
    from lazybrick.semantics.reference import evaluate
    stage = {"id":"q", "semantics":{"profile":PROFILE_ID,"profile_digest":profile_digest()}}
    identity = {"recipe_digest":"a"*64,"plan_digest":"b"*64,"artifact_id":"c"*64,"run_id":"r","attempt_id":"a"}
    binding = {"identity":identity,"stage_id":"q","stage_digest":digest(stage)}
    ref = reference_report()
    report = run_conformance(evaluate, implementation=ref["implementation"], environment=ref["environment"], binding=binding)
    (tmp_path/"identity.json").write_text(json.dumps(identity))
    (tmp_path/"plan.json").write_text(json.dumps({"plan":{"stages":[stage]}}))
    dependency = ModuleType("lazybrick.runs.bundle")
    calls = []
    dependency.verify_bundle = lambda root, **kwargs: calls.append((root,kwargs))
    monkeypatch.setitem(sys.modules, dependency.__name__, dependency)
    def check():
        (tmp_path/"conformance.json").write_text(json.dumps(report))
        return verify_bundle_conformance(tmp_path, expected_bundle_digest="d"*64,
            expected_report_digest=digest(report), expected_context=context_for(report))
    assert check()["status"] == "passed"
    assert calls == [(tmp_path,{"expected_digest":"d"*64})]
    report["execution_kind"] = "reference_evaluator"
    with pytest.raises(SemanticError, match="self-checks"): check()
    report["execution_kind"] = "candidate_execution"
    report["binding"]["identity"]["attempt_id"] = "different"
    with pytest.raises(SemanticError, match="this attempt"): check()
    report["binding"]["identity"]["attempt_id"] = "a"
    report["binding"]["stage_digest"] = "e"*64
    with pytest.raises(SemanticError, match="resolved plan"): check()
    report["binding"]["stage_digest"] = digest(stage)
    def reject(*args, **kwargs):
        raise ValueError("integrity gate rejected")
    dependency.verify_bundle = reject
    with pytest.raises(ValueError, match="integrity gate rejected"): check()


def test_real_bundle_conformance_integration(tmp_path):
    from lazybrick.semantics.profile import PROFILE_ID, profile_digest
    from lazybrick.semantics.reference import evaluate

    identity = RunIdentity.create(
        recipe_digest="a" * 64,
        plan_digest="b" * 64,
        artifact_id="c" * 64,
    )
    stage = {
        "id": "quantize",
        "semantics": {
            "profile": PROFILE_ID,
            "profile_digest": profile_digest(),
        },
    }
    binding = {
        "identity": identity.to_dict(),
        "stage_id": stage["id"],
        "stage_digest": digest(stage),
    }
    reference = reference_report()
    report = run_conformance(
        evaluate,
        implementation=reference["implementation"],
        environment=reference["environment"],
        binding=binding,
    )

    store = RunStore(tmp_path)
    attempt = store.begin(identity)
    attempt.write_text("recipe.yaml", "schema_version: '0.2'\n")
    attempt.write_json("resolved_recipe.json", {"schema_version": "0.2"})
    attempt.write_json(
        "plan.json",
        {
            "recipe_digest": identity.recipe_digest,
            "plan_digest": identity.plan_digest,
            "artifact_id": identity.artifact_id,
            "plan": {"stages": [stage]},
        },
    )
    attempt.write_json("provenance.json", {"tool": "test"})
    attempt.write_json("results.json", {"quality": {"baseline": 1, "quantized": 1}})
    attempt.write_json("state-history.json", {"state": "SUCCEEDED"})
    attempt.write_json("conformance.json", report)
    (attempt.artifact_dir / "model.safetensors").write_bytes(b"weights")
    bundle = attempt.finalize_success({"format": "compressed-tensors"})

    index_path = next((tmp_path / "artifacts").glob("*/*.json"))
    expected_bundle_digest = json.loads(index_path.read_text())["bundle_digest"]
    verified = verify_bundle_conformance(
        bundle,
        expected_bundle_digest=expected_bundle_digest,
        expected_report_digest=digest(report),
        expected_context=context_for(report),
    )

    assert verified["status"] == "passed"
