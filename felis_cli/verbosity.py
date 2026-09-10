"""Shared verbosity and third-party output controls for the FELIS CLI."""

from __future__ import annotations

import io
import logging
import os
from contextlib import contextmanager, redirect_stdout
from typing import Iterator


_verbosity_level = 1


def _configure_ultralytics_logging() -> None:
    """Keep Ultralytics' independent logger aligned with FELIS verbosity."""
    level = logging.INFO if _verbosity_level >= 2 else logging.WARNING
    logger = logging.getLogger("ultralytics")
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)


def configure(level: int) -> None:
    """Apply Cliff's verbosity level to FELIS and lazily imported backends."""
    global _verbosity_level
    _verbosity_level = max(0, int(level))

    # Ultralytics creates a non-propagating logger during import. The environment
    # variable handles future imports; the logger update handles an existing import.
    os.environ["YOLO_VERBOSE"] = "true" if _verbosity_level >= 2 else "false"
    _configure_ultralytics_logging()

    # TensorFlow reads this before import. Keep warnings and errors visible, while
    # suppressing informational CUDA probing unless backend diagnostics are requested.
    if _verbosity_level < 3:
        try:
            current_level = int(os.environ.get("TF_CPP_MIN_LOG_LEVEL", "0"))
        except ValueError:
            current_level = 0
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(max(current_level, 1))


def backend_verbose() -> bool:
    """Return whether third-party inference libraries should print progress."""
    # Ultralytics may have been imported after configure(), replacing its logger.
    _configure_ultralytics_logging()
    return _verbosity_level >= 2


def status(logger: logging.Logger, message: str, *args: object) -> None:
    """Log normal pipeline and per-media progress."""
    logger.info(message, *args)


def detail(logger: logging.Logger, message: str, *args: object) -> None:
    """Log per-frame, per-label, or per-crop detail for ``-v`` and above."""
    logger.debug(message, *args)


def diagnostic(logger: logging.Logger, message: str, *args: object) -> None:
    """Log backend diagnostics only when ``-vv`` (or higher) is active."""
    if _verbosity_level >= 3:
        logger.debug(message, *args)


@contextmanager
def third_party_stdout() -> Iterator[None]:
    """Hide unsolicited backend stdout except at diagnostic verbosity."""
    if _verbosity_level >= 3:
        yield
    else:
        with redirect_stdout(io.StringIO()):
            yield
