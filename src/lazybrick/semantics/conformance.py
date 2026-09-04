"""Scoped conformance execution and integrity verification, never authentication."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import platform
import re
from typing import Callable

from lazybrick.canonical import digest
from lazybrick.semantics.profile import SemanticError, PROFILE_ID, profile_digest
from lazybrick.semantics.fixtures import suite, suite_digest
from lazybrick.semantics.reference import evaluate

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_SCOPE = "supplied-parameter-numerical-conformance"
_POLICY = {"codes":"exact", "reconstructed_bits":"exact"}


def _hash(value: object) -> None:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise SemanticError("invalid_conformance_record", "expected lowercase SHA-256")


def _implementation(value: object) -> None:
    if not isinstance(value, dict) or set(value) != {"name", "source_digest"}:
        raise SemanticError("invalid_conformance_record", "implementation requires name and source_digest")
    if not isinstance(value["name"], str) or not value["name"].strip():
        raise SemanticError("invalid_conformance_record", "implementation name is required")
    _hash(value["source_digest"])


def _environment(value: object) -> None:
    if not isinstance(value, dict) or set(value) != {"python", "platform", "architecture", "dependencies"}:
        raise SemanticError("invalid_conformance_record", "environment identity is incomplete")
    if any(not isinstance(value[k], str) or not value[k] for k in ("python","platform","architecture")):
        raise SemanticError("invalid_conformance_record", "environment strings are required")
    if not isinstance(value["dependencies"], dict) or any(not isinstance(k,str) or not k or not isinstance(v,str) or not v for k,v in value["dependencies"].items()):
        raise SemanticError("invalid_conformance_record", "dependency versions must be explicit")


def _binding(value: object) -> None:
    if value is None:
        return
    if not isinstance(value, dict) or set(value) != {"identity", "stage_id", "stage_digest"}:
        raise SemanticError("invalid_conformance_record", "invalid attempt/stage binding")
    _hash(value["stage_digest"])
    if not isinstance(value["stage_id"],str) or not value["stage_id"]:
        raise SemanticError("invalid_conformance_record", "stage ID is required")
    identity = value["identity"]
    if not isinstance(identity, dict) or set(identity) != {"recipe_digest","plan_digest","artifact_id","run_id","attempt_id"}:
        raise SemanticError("invalid_conformance_record", "identity binding is incomplete")
    for key in ("recipe_digest","plan_digest","artifact_id"):
        _hash(identity[key])
    for key in ("run_id","attempt_id"):
        if not isinstance(identity[key],str) or not identity[key]:
            raise SemanticError("invalid_conformance_record", "run and attempt IDs are required")


def context_for(report: dict[str, object]) -> dict[str, str]:
    return {"profile_digest": report["profile_digest"], "suite_digest": report["suite_digest"],
            "implementation_digest": digest(report["implementation"]),
            "environment_digest": digest(report["environment"]),
            "binding_digest": digest(report["binding"]), "execution_kind": report["execution_kind"]}


def run_conformance(candidate: Callable, *, implementation: dict[str, str],
                    environment: dict[str, object], binding: dict[str, object] | None = None,
                    execution_kind: str = "candidate_execution") -> dict[str, object]:
    _implementation(implementation)
    _environment(environment)
    _binding(binding)
    if not callable(candidate):
        raise SemanticError("invalid_conformance_candidate", "candidate must be callable")
    if execution_kind not in {"candidate_execution", "reference_evaluator"}:
        raise SemanticError("invalid_conformance_record", "unsupported execution kind")
    cases = []
    for case in suite():
        try:
            output = candidate(deepcopy(case["input"]))
            passed = digest(output) == digest(case["expected"])
            cases.append({"id":case["id"], "status":"passed" if passed else "failed", "output":deepcopy(output)})
        except Exception:
            # Do not leak exception strings (which can contain paths or secrets).
            cases.append({"id":case["id"], "status":"error", "output":None})
    return {"report_version":"1", "scope":_SCOPE, "profile":PROFILE_ID,
            "profile_digest":profile_digest(), "suite_digest":suite_digest(),
            "comparison":dict(_POLICY), "implementation":deepcopy(implementation),
            "environment":deepcopy(environment), "binding":deepcopy(binding),
            "execution_kind":execution_kind, "cases":cases,
            "status":"passed" if all(c["status"] == "passed" for c in cases) else "failed"}


def reference_report() -> dict[str, object]:
    root = Path(__file__).parent
    sources = {name:sha256((root/name).read_bytes()).hexdigest()
               for name in ("profile.py","reference.py","fixtures.py","conformance.py")}
    return run_conformance(evaluate,
        implementation={"name":"lazybrick.cpu-reference", "source_digest":digest(sources)},
        environment={"python":platform.python_version(), "platform":platform.system(),
                     "architecture":platform.machine(), "dependencies":{}},
        execution_kind="reference_evaluator")


def verify_report(report: object, *, expected_digest: str,
                  expected_context: dict[str, str]) -> dict[str, object]:
    _hash(expected_digest)
    if not isinstance(report, dict) or set(report) != {"report_version","scope","profile","profile_digest","suite_digest","comparison","implementation","environment","binding","execution_kind","cases","status"}:
        raise SemanticError("invalid_conformance_record", "unknown or missing report fields")
    if digest(report) != expected_digest:
        raise SemanticError("conformance_digest_mismatch", "report differs from external digest")
    _implementation(report["implementation"])
    _environment(report["environment"])
    _binding(report["binding"])
    if report["report_version"] != "1" or report["scope"] != _SCOPE or report["profile"] != PROFILE_ID:
        raise SemanticError("invalid_conformance_record", "unsupported report version, profile, or scope")
    if report["profile_digest"] != profile_digest() or report["suite_digest"] != suite_digest() or report["comparison"] != _POLICY:
        raise SemanticError("conformance_contract_mismatch", "profile, suite or comparison policy differs")
    if report["execution_kind"] not in {"candidate_execution", "reference_evaluator"} or context_for(report) != expected_context:
        raise SemanticError("conformance_binding_mismatch", "report differs from expected implementation/environment/attempt context")
    cases = report["cases"]
    expected = suite()
    if not isinstance(cases,list) or len(cases) != len(expected):
        raise SemanticError("incomplete_conformance", "conformance cases are missing or duplicated")
    statuses = []
    for result, case in zip(cases,expected,strict=True):
        if not isinstance(result,dict) or set(result) != {"id","status","output"} or result["id"] != case["id"]:
            raise SemanticError("invalid_conformance_record", "case identity or fields differ")
        if result["status"] == "error":
            if result["output"] is not None:
                raise SemanticError("invalid_conformance_record", "error case must have no output")
            status = "error"
        else:
            status = "passed" if digest(result["output"]) == digest(case["expected"]) else "failed"
        if result["status"] != status:
            raise SemanticError("conformance_result_mismatch", "case status disagrees with outputs")
        statuses.append(status)
    status = "passed" if all(s == "passed" for s in statuses) else "failed"
    if report["status"] != status:
        raise SemanticError("conformance_result_mismatch", "summary disagrees with case results")
    return deepcopy(report)


def verify_bundle_conformance(bundle: str | Path, *, expected_bundle_digest: str,
                              expected_report_digest: str, expected_context: dict[str,str]):
    _hash(expected_bundle_digest)
    try:
        from lazybrick.runs.bundle import verify_bundle
    except ModuleNotFoundError as error:
        if error.name != "lazybrick.runs.bundle":
            raise
        raise SemanticError("bundle_integrity_unavailable", "complete bundle integrity verifier is required") from error
    root = Path(bundle)
    verify_bundle(root, expected_digest=expected_bundle_digest)
    report = json.loads((root/"conformance.json").read_text())
    verified = verify_report(report, expected_digest=expected_report_digest, expected_context=expected_context)
    if verified["execution_kind"] != "candidate_execution":
        raise SemanticError("conformance_scope_mismatch", "reference self-checks cannot establish artifact conformance")
    identity = json.loads((root/"identity.json").read_text())
    plan_record = json.loads((root/"plan.json").read_text())
    stages = plan_record.get("plan", {}).get("stages", [])
    binding = verified["binding"]
    if binding is None or binding["identity"] != identity:
        raise SemanticError("conformance_binding_mismatch", "report is not bound to this attempt")
    matches = [stage for stage in stages if stage.get("id") == binding["stage_id"]]
    if len(matches) != 1 or digest(matches[0]) != binding["stage_digest"]:
        raise SemanticError("conformance_binding_mismatch", "report stage differs from resolved plan")
    if matches[0].get("semantics") != {"profile":PROFILE_ID,"profile_digest":profile_digest()}:
        raise SemanticError("conformance_contract_mismatch", "stage does not declare the reported profile")
    return verified
