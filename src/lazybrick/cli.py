"""Command-line interface.

Exit codes are part of the contract, because scripts read them:

    0  success, or an accepted plan
    2  the recipe is invalid                (RecipeValidationError)
    3  a reference could not be resolved    (ResolutionError)
    4  the plan is incompatible             (rejected by the planner)
    5  usage error, or a refused operation

``plan`` never downloads weights, allocates a GPU, or executes a plugin. It
reads metadata, intersects capabilities, and prints the answer.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys
from typing import Any

from lazybrick.__about__ import __version__
from lazybrick.canonical import canonical_json
from lazybrick.capabilities import HardwareProfile, check_compatibility
from lazybrick.errors import RecipeValidationError
from lazybrick.records import ExecutionPlan, ModelRef, PluginManifest
from lazybrick.recipe import load_recipe
from lazybrick.resolve import ResolutionError, Resolver, ResolverCache

OK = 0
INVALID_RECIPE = 2
UNRESOLVED = 3
INCOMPATIBLE = 4
REFUSED = 5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lazybrick",
        description="Compose, validate, and plan reproducible compression recipes.",
        epilog=(
            "exit codes: 0 ok, 2 invalid recipe, 3 unresolved reference, "
            "4 incompatible plan, 5 refused"
        ),
    )
    parser.add_argument("--version", action="version", version=f"lazybrick {__version__}")

    commands = parser.add_subparsers(dest="command")

    def shared(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "--json", action="store_true", help="emit machine-readable JSON."
        )

    def resolution(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "--offline",
            action="store_true",
            help="resolve from the local cache only; never touch the network.",
        )
        sub.add_argument(
            "--cache-dir",
            metavar="PATH",
            help="where resolved metadata is cached.",
        )

    validate = commands.add_parser("validate", help="Validate a recipe.")
    validate.add_argument("recipe")
    shared(validate)

    digest = commands.add_parser(
        "digest", help="Print a recipe's authored-content digest."
    )
    digest.add_argument("recipe")

    inspect = commands.add_parser(
        "inspect", help="Resolve a model and report what it is, using metadata only."
    )
    inspect.add_argument("model", help="hf://owner/name, or owner/name.")
    inspect.add_argument("--revision", default="main")
    shared(inspect)
    resolution(inspect)

    plan = commands.add_parser(
        "plan", help="Resolve a recipe and check whether it can run."
    )
    plan.add_argument("recipe")
    plan.add_argument(
        "--target",
        metavar="PATH",
        help="JSON description of the machine that will run the build.",
    )
    plan.add_argument(
        "--plugin-manifest",
        metavar="PATH",
        action="append",
        default=[],
        help="a plugin manifest to check against; repeatable.",
    )
    shared(plan)
    resolution(plan)

    build = commands.add_parser("build", help="Build an artifact (M0: dry runs only).")
    build.add_argument("recipe")
    build.add_argument(
        "--dry-run",
        action="store_true",
        help="required: execution is not implemented yet.",
    )
    build.add_argument("--target", metavar="PATH")
    build.add_argument(
        "--plugin-manifest", metavar="PATH", action="append", default=[]
    )
    shared(build)
    resolution(build)

    return parser


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(canonical_json(payload).decode("utf-8") + "\n")


def _fail(message: str, issues: Sequence[Any] = ()) -> None:
    print(message, file=sys.stderr)
    for issue in issues:
        print(f"  {issue}", file=sys.stderr)


def _load_json(path: str, what: str) -> dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as error:
        raise _CliError(f"cannot read {what} {path!r}: {error}") from error
    except json.JSONDecodeError as error:
        raise _CliError(f"{what} {path!r} is not valid JSON: {error}") from error


class _CliError(Exception):
    """A usage problem, reported without a traceback."""


def _resolver(args: argparse.Namespace) -> Resolver:
    cache = ResolverCache(args.cache_dir) if args.cache_dir else ResolverCache()
    return Resolver(cache=cache, offline=args.offline)


def _manifests(paths: Sequence[str]) -> list[PluginManifest]:
    return [
        PluginManifest.from_json(_load_json(path, "plugin manifest")) for path in paths
    ]


def _hardware(path: str | None) -> HardwareProfile | None:
    if path is None:
        return None
    try:
        return HardwareProfile.from_json(_load_json(path, "target"))
    except KeyError as error:
        raise _CliError(f"target {path!r} is missing {error}") from error


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def _cmd_validate(args: argparse.Namespace) -> int:
    recipe = load_recipe(args.recipe)
    if args.json:
        _emit(
            {
                "valid": True,
                "schema_version": recipe.schema_version,
                "recipe_digest": recipe.digest,
            }
        )
    else:
        print(f"Valid LazyBrick recipe (schema v{recipe.schema_version})")
        print(f"recipe_digest: {recipe.digest}")
    return OK


def _cmd_digest(args: argparse.Namespace) -> int:
    print(load_recipe(args.recipe).digest)
    return OK


def _cmd_inspect(args: argparse.Namespace) -> int:
    uri = args.model if "://" in args.model else f"hf://{args.model}"
    model = _resolver(args).resolve_model(ModelRef(uri, args.revision))

    if args.json:
        _emit(model.to_json())
        return OK

    print(f"{model.repo_id}")
    print(f"  revision       {model.revision}")
    if model.requested_revision != model.revision:
        print(f"  requested      {model.requested_revision}")
    print(f"  profile        {model.model_profile}")
    print(f"  components     {', '.join(model.components)}")
    print(f"  architecture   {', '.join(model.architectures) or 'unknown'}")
    print(f"  dtype          {model.dtype or 'unknown'}")
    print(f"  weights        {model.weight_format}")
    if model.parameter_count:
        print(f"  parameters     {model.parameter_count:,}")
    print(f"  license        {model.license or 'unknown'}")
    if model.requires_remote_code:
        print("  remote code    REQUIRED (loading executes code from the repo)")
    if model.gated:
        print("  gated          yes (needs HF_TOKEN)")
    return OK


def _plan_and_check(args: argparse.Namespace):
    document = load_recipe(args.recipe)
    resolver = _resolver(args)
    resolved = resolver.resolve_recipe(document.data, document.digest)
    plan = ExecutionPlan.from_recipe(resolved.recipe, document.digest)
    result = check_compatibility(
        plan, resolved.model, _manifests(args.plugin_manifest), _hardware(args.target)
    )
    return document, resolved, plan, result


def _cmd_plan(args: argparse.Namespace) -> int:
    document, resolved, plan, result = _plan_and_check(args)

    if args.json:
        _emit(
            {
                "recipe_digest": document.digest,
                "plan_digest": plan.plan_digest,
                "artifact_id": plan.artifact_id,
                "resolved_recipe": resolved.to_json(),
                "plan": plan.to_json(),
                "compatibility": result.to_json(),
            }
        )
        return OK if result.accepted else INCOMPATIBLE

    print(f"model          {resolved.model.repo_id}@{resolved.model.revision[:12]}")
    print(f"profile        {resolved.model.model_profile}")
    print(f"stages         {', '.join(stage.id for stage in plan.stages)}")
    print(f"recipe_digest  {document.digest}")
    print(f"plan_digest    {plan.plan_digest}")
    print(f"artifact_id    {plan.artifact_id}")
    print()
    if result.accepted:
        print("ACCEPTED: every capability intersection is non-empty.")
        return OK

    print(f"REJECTED: {len(result.reasons)} problem(s).")
    for reason in result.reasons:
        print(f"  {reason}")
    return INCOMPATIBLE


def _cmd_build(args: argparse.Namespace) -> int:
    if not args.dry_run:
        _fail(
            "lazybrick build: execution is not implemented yet; pass --dry-run to "
            "resolve and check the recipe without running anything."
        )
        return REFUSED

    document, resolved, plan, result = _plan_and_check(args)
    if args.json:
        _emit(
            {
                "dry_run": True,
                "would_build": result.accepted,
                "artifact_id": plan.artifact_id,
                "plan_digest": plan.plan_digest,
                "compatibility": result.to_json(),
            }
        )
        return OK if result.accepted else INCOMPATIBLE

    if not result.accepted:
        print(f"REJECTED: {len(result.reasons)} problem(s).")
        for reason in result.reasons:
            print(f"  {reason}")
        return INCOMPATIBLE

    print("Dry run only; nothing was executed.")
    print(f"would build    {plan.artifact_id}")
    print(f"from           {resolved.model.repo_id}@{resolved.model.revision[:12]}")
    print(f"via            {', '.join(stage.plugin for stage in plan.stages)}")
    print(f"exporting      {plan.export.format} for {plan.export.runtime}")
    return OK


_COMMANDS = {
    "validate": _cmd_validate,
    "digest": _cmd_digest,
    "inspect": _cmd_inspect,
    "plan": _cmd_plan,
    "build": _cmd_build,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return OK

    try:
        return _COMMANDS[args.command](args)
    except RecipeValidationError as error:
        _fail("Invalid LazyBrick recipe:", error.issues)
        return INVALID_RECIPE
    except ResolutionError as error:
        _fail("Cannot resolve recipe references:", error.issues)
        return UNRESOLVED
    except _CliError as error:
        _fail(f"lazybrick {args.command}: {error}")
        return REFUSED
