from __future__ import annotations

from lazybrick.runs import collect_provenance, redact_environment


def test_secret_redaction_is_case_insensitive() -> None:
    redacted = redact_environment(
        {
            "PATH": "/bin",
            "HF_TOKEN": "secret",
            "database_password": "secret",
            "MY_API_KEY": "secret",
            "VISIBLE_SETTING": "value",
        }
    )

    assert redacted == {
        "HF_TOKEN": "<redacted>",
        "MY_API_KEY": "<redacted>",
        "PATH": "/bin",
        "VISIBLE_SETTING": "value",
        "database_password": "<redacted>",
    }


def test_provenance_records_required_inventory() -> None:
    record = collect_provenance(
        commands=[["lazybrick", "plan", "recipe.yaml"]],
        seeds={"calibration": 42},
        environment={"HF_TOKEN": "secret"},
        packages=["pytest", "missing-package-for-lazybrick-test"],
    )

    assert record["software"]["python"]
    assert record["software"]["packages"]["pytest"] != "not-installed"
    assert record["software"]["packages"]["missing-package-for-lazybrick-test"] == "not-installed"
    assert record["system"]["os"]
    assert "gpus" in record["nvidia"]
    assert record["commands"] == [["lazybrick", "plan", "recipe.yaml"]]
    assert record["seeds"] == {"calibration": 42}
    assert record["environment"]["HF_TOKEN"] == "<redacted>"
