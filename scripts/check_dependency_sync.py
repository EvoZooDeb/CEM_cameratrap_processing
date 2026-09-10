"""Verify that Docker requirement pins match the project metadata."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_requirements(path: Path) -> set[str]:
    """Return non-comment requirement lines from a simple requirements file."""
    return {
        line
        for raw_line in path.read_text(encoding="utf-8").splitlines()
        if (line := raw_line.strip()) and not line.startswith("#")
    }


def report_difference(label: str, expected: set[str], actual: set[str]) -> bool:
    """Print dependency drift and return whether the two sets differ."""
    if expected == actual:
        return False

    print(f"{label} dependencies are out of sync:", file=sys.stderr)
    for requirement in sorted(expected - actual):
        print(f"  missing from requirements file: {requirement}", file=sys.stderr)
    for requirement in sorted(actual - expected):
        print(f"  missing from pyproject.toml: {requirement}", file=sys.stderr)
    return True


def main() -> int:
    """Compare base and two-stage requirement files with pyproject.toml."""
    with (ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]

    base_drift = report_difference(
        "Base",
        set(project["dependencies"]),
        read_requirements(ROOT / "requirements.txt"),
    )
    two_stage_drift = report_difference(
        "Two-stage",
        set(project["optional-dependencies"]["two-stage"]),
        read_requirements(ROOT / "requirements-two-stage.txt")
        | read_requirements(ROOT / "requirements-pytorchwildlife-runtime.txt"),
    )
    return int(base_drift or two_stage_drift)


if __name__ == "__main__":
    raise SystemExit(main())
