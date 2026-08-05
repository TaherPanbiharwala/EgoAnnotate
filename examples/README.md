# examples/

`synthetic_smoke_test.mp4` — a 12s synthetic ffmpeg test pattern (`testsrc`),
NOT real egocentric footage. Its only job is to let `scripts/demo.py` and CI
exercise the plumbing (probe → frame extraction → MediaPipe → captioning →
storage) with zero setup: no API key, no GPU, no real video, no accounts.

It has no hands in it, so `hands.parquet` for this clip will legitimately
show `hands_present=0` on every frame — that's expected, not a bug.

**This is not the golden-file test the plan calls for.** A real golden-file
test (30s of real footage + human-labeled expected segment boundaries, for
CI regression-testing the segmentation algorithm) needs real footage, which
is blocked on Phase 0 (Google Drive access). Once real footage exists, this
directory should also gain:

- `sample.mp4` — a real ~30s clip
- `sample.expected_boundaries.jsonl` — human-labeled boundaries
- a note on whether those labels are genuine human ground truth or frozen
  pipeline output (if frozen, it's a non-regression test, not ground truth —
  say so explicitly, per the plan's testing-honesty rule)

## Recording a real VLM fixture for `backends.fake.FakeBackend`

Once you have an OpenRouter key, capture a handful of REAL responses (not
synthesized ones) so `scripts/demo.py` can replay genuine model behavior
with zero further API cost:

```python
# one-off script, not part of the package:
import json
from pathlib import Path
from egoannote.backends.openai_compat import OpenAICompatBackend, ModelConfig
# ... construct a real backend, call .caption() a few times on real windows,
# and append {"response_text": resp.text} as JSONL to examples/vlm_fixture.jsonl
```

`FakeBackend(fixture_path=Path("examples/vlm_fixture.jsonl"))` will then
cycle through those real responses instead of the synthetic default.
