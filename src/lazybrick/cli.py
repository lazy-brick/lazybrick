"""Command-line interface for the initial LazyBrick package."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import sys

from lazybrick.__about__ import __version__
from lazybrick.recipe import RecipeValidationError, load_recipe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lazybrick",
        description="Validate and digest model-compression recipes.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"lazybrick {__version__}",
    )

    commands = parser.add_subparsers(dest="command")

    validate = commands.add_parser(
        "validate",
        help="Validate a YAML or JSON recipe.",
    )
    validate.add_argument("recipe")

    digest = commands.add_parser(
        "digest",
        help="Print the deterministic SHA-256 digest of a recipe's authored content.",
    )
    digest.add_argument("recipe")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    try:
        recipe = load_recipe(args.recipe)
    except RecipeValidationError as error:
        print("Invalid LazyBrick recipe:", file=sys.stderr)
        for issue in error.issues:
            print(f"  {issue}", file=sys.stderr)
        return 2

    if args.command == "validate":
        print(f"Valid LazyBrick recipe (schema v{recipe.schema_version})")
        print(f"recipe_digest: {recipe.digest}")
        return 0

    if args.command == "digest":
        print(recipe.digest)
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2
