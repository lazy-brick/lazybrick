"""Schema v0.1 tests.

Every negative case asserts the *reason code*, not the English message, so that
wording can be improved without rewriting the suite.
"""

from __future__ import annotations

from typing import Any

import pytest

from lazybrick import RecipeValidationError, is_immutable_revision, validate_recipe
from lazybrick.schema import validate_document


def codes(recipe: dict[str, Any]) -> list[str]:
    return [issue.code for issue in validate_document(recipe)]


def paths(recipe: dict[str, Any]) -> list[str]:
    return [issue.path for issue in validate_document(recipe)]


def test_valid_recipe_passes(valid_recipe) -> None:
    assert validate_document(valid_recipe) == ()
    validate_recipe(valid_recipe)


def test_calibration_and_evaluation_are_optional(valid_recipe) -> None:
    valid_recipe.pop("calibration")
    valid_recipe.pop("evaluation")

    # Whether AWQ *needs* calibration is a capability question, not a shape one.
    assert validate_document(valid_recipe) == ()


class TestSchemaVersion:
    def test_missing_version_is_reported_alone(self, valid_recipe) -> None:
        valid_recipe.pop("schema_version")
        valid_recipe.pop("model")

        # The body is not validated against a schema we cannot identify.
        assert codes(valid_recipe) == ["missing_field"]
        assert paths(valid_recipe) == ["schema_version"]

    def test_unknown_version_is_rejected(self, valid_recipe) -> None:
        valid_recipe["schema_version"] = "0.2"

        assert codes(valid_recipe) == ["unknown_schema_version"]

    def test_unknown_version_suppresses_body_errors(self, valid_recipe) -> None:
        valid_recipe["schema_version"] = "99"
        valid_recipe["stages"] = "not a list"

        assert codes(valid_recipe) == ["unknown_schema_version"]


class TestUnknownFields:
    def test_unknown_top_level_field(self, valid_recipe) -> None:
        valid_recipe["quantise"] = True

        assert codes(valid_recipe) == ["unknown_field"]
        assert paths(valid_recipe) == ["quantise"]

    def test_unknown_nested_field(self, valid_recipe) -> None:
        valid_recipe["target"]["gpu"] = "h100"

        assert paths(valid_recipe) == ["target.gpu"]

    def test_unknown_stage_field(self, valid_recipe) -> None:
        valid_recipe["stages"][0]["algorithm"] = "awq"

        assert paths(valid_recipe) == ["stages[0].algorithm"]


class TestModel:
    def test_uri_is_required(self, valid_recipe) -> None:
        valid_recipe["model"].pop("uri")

        assert codes(valid_recipe) == ["missing_field"]
        assert paths(valid_recipe) == ["model.uri"]

    def test_revision_is_required(self, valid_recipe) -> None:
        valid_recipe["model"].pop("revision")

        assert paths(valid_recipe) == ["model.revision"]

    def test_unknown_scheme_is_rejected(self, valid_recipe) -> None:
        valid_recipe["model"]["uri"] = "s3://bucket/model"

        assert codes(valid_recipe) == ["unsupported_scheme"]

    def test_bare_repo_id_is_not_a_uri(self, valid_recipe) -> None:
        valid_recipe["model"]["uri"] = "Qwen/Qwen3-4B"

        assert codes(valid_recipe) == ["invalid_value"]

    def test_mutable_revision_is_accepted_by_the_schema(self, valid_recipe) -> None:
        # The resolver pins it; execution is refused later if it is still mutable.
        valid_recipe["model"]["revision"] = "main"

        assert validate_document(valid_recipe) == ()

    def test_trust_remote_code_must_be_boolean(self, valid_recipe) -> None:
        valid_recipe["model"]["trust_remote_code"] = "no"

        assert paths(valid_recipe) == ["model.trust_remote_code"]


class TestStages:
    def test_stages_must_not_be_empty(self, valid_recipe) -> None:
        valid_recipe["stages"] = []

        assert codes(valid_recipe) == ["invalid_type"]

    def test_missing_stages_is_reported(self, valid_recipe) -> None:
        valid_recipe.pop("stages")

        assert paths(valid_recipe) == ["stages"]

    def test_duplicate_stage_ids_fail(self, valid_recipe) -> None:
        valid_recipe["stages"].append(dict(valid_recipe["stages"][0]))

        assert codes(valid_recipe) == ["duplicate_id"]
        assert paths(valid_recipe) == ["stages[1].id"]

    def test_plugin_version_is_required(self, valid_recipe) -> None:
        valid_recipe["stages"][0].pop("plugin_version")

        assert paths(valid_recipe) == ["stages[0].plugin_version"]

    def test_implementation_is_required(self, valid_recipe) -> None:
        valid_recipe["stages"][0].pop("implementation")

        assert codes(valid_recipe) == ["missing_field"]
        assert paths(valid_recipe) == ["stages[0].implementation"]

    def test_implementation_must_be_pinned(self, valid_recipe) -> None:
        valid_recipe["stages"][0]["implementation"] = {}

        assert codes(valid_recipe) == ["missing_field"]

    def test_branch_instead_of_commit_is_rejected(self, valid_recipe) -> None:
        valid_recipe["stages"][0]["implementation"]["commit"] = "main"

        assert codes(valid_recipe) == ["mutable_reference"]

    def test_short_commit_is_rejected(self, valid_recipe) -> None:
        valid_recipe["stages"][0]["implementation"]["commit"] = "de46bfd"

        assert codes(valid_recipe) == ["mutable_reference"]

    def test_git_and_container_pin_cannot_be_combined(self, valid_recipe) -> None:
        valid_recipe["stages"][0]["implementation"]["container"] = (
            "ghcr.io/example/awq@sha256:" + "a" * 64
        )

        assert codes(valid_recipe) == ["invalid_value"]

    def test_container_must_be_digest_pinned(self, valid_recipe) -> None:
        valid_recipe["stages"][0]["implementation"] = {
            "container": "ghcr.io/example/awq:latest"
        }

        assert codes(valid_recipe) == ["mutable_reference"]

    def test_digest_pinned_container_is_accepted(self, valid_recipe) -> None:
        valid_recipe["stages"][0]["implementation"] = {
            "container": "ghcr.io/example/awq@sha256:" + "c" * 64
        }

        assert validate_document(valid_recipe) == ()


class TestNumbers:
    def test_fractional_sample_count_is_rejected(self, valid_recipe) -> None:
        valid_recipe["calibration"]["samples"] = 256.0

        assert codes(valid_recipe) == ["invalid_type"]

    def test_boolean_is_not_an_integer(self, valid_recipe) -> None:
        valid_recipe["target"]["device_count"] = True

        assert codes(valid_recipe) == ["invalid_type"]

    def test_zero_devices_is_rejected(self, valid_recipe) -> None:
        valid_recipe["target"]["device_count"] = 0

        assert codes(valid_recipe) == ["invalid_value"]

    def test_compute_capability_must_be_a_string(self, valid_recipe) -> None:
        valid_recipe["target"]["min_compute_capability"] = 8.0

        assert codes(valid_recipe) == ["invalid_type"]

    def test_compute_capability_shape_is_checked(self, valid_recipe) -> None:
        valid_recipe["target"]["min_compute_capability"] = "ampere"

        assert codes(valid_recipe) == ["invalid_value"]


class TestReporting:
    def test_every_issue_is_reported_in_one_pass(self, valid_recipe) -> None:
        valid_recipe["model"].pop("uri")
        valid_recipe["stages"][0].pop("plugin")
        valid_recipe["export"].pop("format")

        assert sorted(paths(valid_recipe)) == [
            "export.format",
            "model.uri",
            "stages[0].plugin",
        ]

    def test_error_exposes_structured_issues(self, valid_recipe) -> None:
        valid_recipe["model"].pop("uri")

        with pytest.raises(RecipeValidationError) as caught:
            validate_recipe(valid_recipe)

        assert caught.value.codes == ("missing_field",)
        assert caught.value.to_json() == [
            {
                "path": "model.uri",
                "code": "missing_field",
                "message": "required field is missing",
            }
        ]

    def test_root_must_be_a_mapping(self) -> None:
        assert codes(["not", "a", "mapping"]) == ["invalid_type"]


@pytest.mark.parametrize(
    ("revision", "expected"),
    [
        ("1cfa9a7208912126459214e8b04321603b3df60c", True),
        ("1CFA9A7208912126459214E8B04321603B3DF60C", False),
        ("main", False),
        ("v1.0", False),
        ("1cfa9a7", False),
        ("", False),
        (None, False),
        (12345, False),
    ],
)
def test_is_immutable_revision(revision: object, expected: bool) -> None:
    assert is_immutable_revision(revision) is expected
