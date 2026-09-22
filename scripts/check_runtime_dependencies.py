"""Validate the selected OpenCV runtime while tolerating its alias metadata."""

from __future__ import annotations

import argparse
import importlib.metadata
import re
import subprocess
import sys


OPENCV_PACKAGES = {"opencv-python", "opencv-python-headless"}
MISSING_REQUIREMENT = re.compile(
    r"^.+ requires (opencv-python(?:-headless)?), which is not installed\.$",
    re.IGNORECASE,
)


def installed_opencv_packages() -> set[str]:
    """Return installed distributions that provide the shared ``cv2`` module."""
    return {
        name
        for distribution in importlib.metadata.distributions()
        if (raw_name := distribution.metadata.get("Name"))
        and (name := raw_name.lower()) in OPENCV_PACKAGES
    }


def unexpected_pip_check_lines(output: str, selected_package: str) -> list[str]:
    """Ignore only metadata complaints satisfied by the selected OpenCV variant."""
    replaced_package = (OPENCV_PACKAGES - {selected_package}).pop()
    unexpected = []
    for line in output.splitlines():
        stripped = line.strip()
        if stripped == "No broken requirements found.":
            continue
        match = MISSING_REQUIREMENT.match(stripped)
        if not match or match.group(1).lower() != replaced_package:
            unexpected.append(stripped)
    return [line for line in unexpected if line]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--opencv-variant", choices=("gui", "headless"), required=True)
    args = parser.parse_args()

    selected_package = (
        "opencv-python" if args.opencv_variant == "gui" else "opencv-python-headless"
    )
    installed = installed_opencv_packages()
    if installed != {selected_package}:
        print(
            f"Expected only {selected_package}, found: {sorted(installed)}",
            file=sys.stderr,
        )
        return 1

    import cv2

    gui_line = next(
        (line.strip() for line in cv2.getBuildInformation().splitlines() if "GUI:" in line),
        "",
    )
    is_headless = gui_line.upper().endswith("NONE")
    if is_headless != (args.opencv_variant == "headless"):
        print(f"OpenCV build does not match {args.opencv_variant}: {gui_line}", file=sys.stderr)
        return 1

    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        check=False,
        capture_output=True,
        text=True,
    )
    output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    unexpected = unexpected_pip_check_lines(output, selected_package)
    if unexpected:
        print("Unexpected dependency problems:", file=sys.stderr)
        print("\n".join(unexpected), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
