#!/usr/bin/env python3
import sys

from felis_cli.app import FelisApp


def main(argv: list[str] | None = None) -> int:
    app = FelisApp()
    return app.run(argv or sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())

