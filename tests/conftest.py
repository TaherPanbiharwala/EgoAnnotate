"""Shared fixtures.

`jobs/10_blur_egoblur.py` and `scripts/22_blur_review.py` start with digits,
so they are not importable module names. They are also standalone scripts by
design (PEP 723 / no-CLI), not package members. Loading them by path is what
makes their pure logic testable without a GPU, real EgoBlur weights, or a
rented pod — and that logic is where the privacy-critical bugs live.

Both modules import only stdlib at module scope; numpy, cv2 and torch are
imported inside the functions that need them, so this load is cheap and does
not require the GPU dependency set the job declares in its PEP 723 header.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def blur_job():
    return _load("jobs/10_blur_egoblur.py", "blur_job")


@pytest.fixture(scope="session")
def blur_review():
    return _load("scripts/22_blur_review.py", "blur_review")
