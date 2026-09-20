"""Shared fixtures.

`egoblur/job.py` and `egoblur/review.py` are standalone scripts,
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

# egoblur/ is a real package (has __init__.py): its pose_prior.py/hand_prior.py/
# verify_yunet.py use ordinary relative imports internally, so tests reach them
# as `from egoblur import pose_prior` rather than loading them by path like the
# standalone job.py/review.py above.
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# public-release-tools/ is a flat sibling folder (not a package under egoannote):
# its modules import the installed `egoannote` package normally, and tests reach
# them as `import annotate_redacted`, `import public_release`, `from pack...`.
_PUBLIC_RELEASE_TOOLS = _ROOT / "public-release-tools"
if str(_PUBLIC_RELEASE_TOOLS) not in sys.path:
    sys.path.insert(0, str(_PUBLIC_RELEASE_TOOLS))


def _load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def blur_job():
    return _load("egoblur/job.py", "blur_job")


@pytest.fixture(scope="session")
def blur_review():
    return _load("egoblur/review.py", "blur_review")


@pytest.fixture(scope="session")
def depth_compare_job():
    """The private DA3 GPU job has stdlib-only module imports by design."""

    return _load("jobs/30_depth_da3_metric.py", "depth_compare_job")

