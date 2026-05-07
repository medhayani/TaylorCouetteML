"""Structured logger and tiny dict-flatten helper for TensorBoard."""

from __future__ import annotations

import logging
import sys
from typing import Any, Dict


def get_logger(name: str = "code6", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    return logger


def log_dict(logger: logging.Logger, d: Dict[str, Any], prefix: str = "") -> None:
    for k, v in d.items():
        if isinstance(v, dict):
            log_dict(logger, v, prefix=f"{prefix}{k}.")
        else:
            logger.info(f"{prefix}{k} = {v}")
