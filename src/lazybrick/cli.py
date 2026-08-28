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
        description="Validate and fingerprint model-compression recipes.",
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

    fingerprint = commands.add_parser(
        "fingerprint",
        help="Print the deterministic SHA-256 fingerprint of a recipe.",
    )
    fingerprint.add_argument("recipe")

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
        print(str(error), file=sys.stderr)
        return 2

    if args.command == "validate":
        print(f"Valid LazyBrick recipe: {recipe.fingerprint}")
        return 0

    if args.command == "fingerprint":
        print(recipe.fingerprint)
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2
