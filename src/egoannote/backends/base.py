"""The one contract every VLM backend satisfies. Pipeline code (layers/caption.py)
imports only this — never a provider name — so swapping models is a
models.toml edit, not a code change."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(slots=True)
class VLMResponse:
    text: str
    latency_ms: int
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    provider: str | None = None  # realized upstream provider, for pinning/audit
    model_version: str | None = None


class VLMBackend(Protocol):
    """Implementations resize/preprocess frames themselves, retry at this
    layer (not in the caller), and raise on permanent failure.

    Frames MUST be sent with a per-frame text label immediately preceding
    each image ("Frame 0", "Frame 1", ...). prompts/caption_v3.txt instructs
    the model to return `start_frame`/`end_frame` indices referring to those
    labels — without them the model is guessing at which image it saw an
    action in, and every temporal boundary downstream is unanchored. This is
    the load-bearing detail of the ordered-list design; do not "simplify" it
    away by concatenating images.
    """

    name: str
    model_id: str

    def caption(self, frames_jpeg: list[bytes], prompt: str) -> VLMResponse: ...


def frame_label(idx: int) -> str:
    """The per-frame marker the caption prompt tells the model to expect.
    Shared by every backend so the wire format cannot drift from the prompt."""
    return f"Frame {idx}"


class SpendLimitExceeded(RuntimeError):
    """Raised by the spend guard before a call that would exceed the cap."""


class PermanentBackendError(RuntimeError):
    """A failure retrying cannot fix: bad API key, unknown model id, an
    unenforceable spend cap.

    Part of the BACKEND CONTRACT, not one provider's private detail. Callers
    distinguish permanent from transient to decide whether to abort the run
    or record a per-window error and continue — so every implementation must
    raise this type, or a caller written against the Protocol will treat a
    bad key as 900 individual window failures and report success.

    (This lived in openai_compat.py until review pointed out that
    layers/caption.py had to import it from there, contradicting this
    module's own "never import a provider name" rule.)
    """
