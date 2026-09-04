from copy import deepcopy
import pytest
from lazybrick.recipe import recipe_digest, validate_recipe
from lazybrick.records import ExecutionPlan
from lazybrick.errors import RecipeValidationError
from lazybrick.semantics.profile import PROFILE_ID, profile_digest
from lazybrick.resolve import Resolver, ResolverCache
from lazybrick.capabilities import check_compatibility, HardwareProfile


def declaration():return {"profile":PROFILE_ID,"profile_digest":profile_digest()}


def test_v01_rejects_new_field_without_migrating(valid_recipe):
    valid_recipe["stages"][0]["semantics"]=declaration()
    with pytest.raises(RecipeValidationError,match="unknown field"):validate_recipe(valid_recipe)


def test_v02_roundtrip_and_identity_separation(valid_recipe):
    legacy=ExecutionPlan.from_recipe(valid_recipe,recipe_digest(valid_recipe))
    assert legacy.semantic_digest is None
    valid_recipe["schema_version"]="0.2";valid_recipe["stages"][0]["semantics"]=declaration()
    plan=ExecutionPlan.from_recipe(valid_recipe,recipe_digest(valid_recipe))
    assert plan.plan_version=="0.2"
    assert ExecutionPlan.from_json(plan.to_json()).to_json()==plan.to_json()
    other=deepcopy(valid_recipe);other["stages"][0]["implementation"]["commit"]="e"*40
    changed=ExecutionPlan.from_recipe(other,recipe_digest(other))
    assert changed.semantic_digest==plan.semantic_digest
    assert changed.artifact_id!=plan.artifact_id
    assert legacy.artifact_id!=plan.artifact_id
    old=plan.to_json();old["plan_version"]="0.1"
    with pytest.raises(RecipeValidationError,match="v0.2"):ExecutionPlan.from_json(old)


def test_planner_does_not_infer_semantics_from_w4_label(valid_recipe,hf_transport,tmp_path):
    valid_recipe["schema_version"]="0.2";valid_recipe["stages"][0]["semantics"]=declaration()
    resolved=Resolver(hf_transport,ResolverCache(tmp_path)).resolve_recipe(valid_recipe,recipe_digest(valid_recipe))
    result=check_compatibility(resolved.plan,resolved.model,[],HardwareProfile.none())
    assert not result.accepted
    assert "unsupported_semantic_profile" in result.codes


def test_adapter_rejects_explicit_semantics_before_importing_optional_stack():
    from lazybrick.adapters.llm_compressor.__main__ import handle
    from lazybrick.adapters.llm_compressor.adapter import AdapterInputError
    with pytest.raises(AdapterInputError,match="not been verified"):
        handle({"operation":"execute","payload":{"semantics":declaration()}})
