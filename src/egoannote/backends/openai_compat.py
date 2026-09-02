"""One class covers OpenRouter, Ollama, vLLM, Together, Groq, LM Studio,
and OpenAI directly — the OpenAI /chat/completions request shape has become
the de-facto standard multi-image API. Adding a provider is a models.toml
entry, not a new class (see backends/registry.py for the loader).

Eng review 2.6 (agreement contamination): OpenRouter routes a model id to
different upstream providers with different quantizations, and can fall
back silently mid-run. If model A is served by two different providers
across a run, the "agreement" you measure is partly provider variance, not
model disagreement. Pin the provider (`provider.order` + `allow_fallbacks:
false`) and persist whatever the response reports back as `provider` on
every record — see schema.WindowCaption.provider.
"""

from __future__ import annotations

import base64
import threading
import time
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

from .. import config
from .base import (
    PermanentBackendError,
    SpendLimitExceeded,
    VLMResponse,
    frame_label,
)

__all__ = ["ModelConfig", "OpenAICompatBackend", "PermanentBackendError", "SpendTracker"]

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "[::1]"}

# Statuses where retrying is pure waste — the same request will fail the same
# way. 429 and 5xx are deliberately NOT here; those are worth a backoff.
_NON_RETRYABLE_STATUS = frozenset({400, 401, 402, 403, 404, 422})


@dataclass(slots=True)
class ModelConfig:
    model_id: str  # key in models.toml, e.g. "example-a"
    backend_name: str
    base_url: str
    model: str  # provider-facing model string, e.g. "google/gemini-2.5-flash"
    max_images: int
    price_in_per_mtok: float
    price_out_per_mtok: float
    provider_order: list[str] | None = None
    # None preserves a provider's default.  Some models mandate reasoning;
    # use the lowest supported effort and exclude its trace so the response
    # budget is available for the required JSON final answer.
    reasoning_enabled: bool | None = None
    reasoning_effort: str | None = None
    reasoning_exclude: bool | None = None


class OpenAICompatBackend:
    def __init__(
        self,
        cfg: ModelConfig,
        api_key: str,
        *,
        max_spend_usd: float | None = None,
        spend_tracker: SpendTracker | None = None,
        retries: int = 2,
        timeout_seconds: float = 120.0,
    ):
        if len(cfg.model) == 0 or cfg.model.startswith("REPLACE_ME"):
            raise ValueError(
                f"models.toml entry for '{cfg.model_id}' has a placeholder model "
                f"string ('{cfg.model}') — fill in a real model id before running."
            )

        # The bearer token is attached to every request this client makes.
        # models.toml explicitly supports local backends (Ollama, vLLM,
        # LM Studio), so http:// is a legitimate configuration for those —
        # but only for a loopback host. A remote http:// endpoint would put
        # the API key on the wire in cleartext.
        parsed = urlparse(cfg.base_url)
        if parsed.scheme != "https" and parsed.hostname not in _LOCAL_HOSTS:
            raise ValueError(
                f"base_url for '{cfg.model_id}' is {cfg.base_url!r}: a non-loopback "
                f"endpoint must use https, or the API key is sent in cleartext. "
                f"Use https://, or point at localhost for a local model server."
            )

        # A model that cannot accept a whole window is a config error, not a
        # per-call error — fail at construction rather than producing an
        # error row for every window in the run.
        if cfg.max_images < config.VLM_WINDOW_FRAMES:
            raise ValueError(
                f"models.toml entry '{cfg.model_id}' declares max_images="
                f"{cfg.max_images}, but a caption window is "
                f"{config.VLM_WINDOW_FRAMES} frames. Pick a model that accepts "
                f"at least {config.VLM_WINDOW_FRAMES} images."
            )

        # The spend guard multiplies provider-reported token counts by these
        # prices. At 0.0 every call estimates $0.00, total_usd never rises,
        # and max_spend_usd silently never trips — an unlimited budget that
        # looks like a configured one. models.toml ships 0.0 placeholders.
        if max_spend_usd is not None and (
            cfg.price_in_per_mtok <= 0.0 and cfg.price_out_per_mtok <= 0.0
        ):
            raise ValueError(
                f"max_spend_usd was set, but models.toml entry '{cfg.model_id}' has "
                f"price_in_per_mtok=price_out_per_mtok=0.0. Every call would cost an "
                f"estimated $0.00 and the limit would never trigger. Fill in real "
                f"prices from the provider before running with a spend cap."
            )
        self.name = cfg.backend_name
        self.model_id = cfg.model_id
        self._cfg = cfg
        self._client = httpx.Client(
            base_url=cfg.base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout_seconds,
        )
        self._max_spend = max_spend_usd
        self._spend = spend_tracker or SpendTracker()
        self._retries = retries

    def close(self) -> None:
        self._client.close()

    def caption(self, frames_jpeg: list[bytes], prompt: str) -> VLMResponse:
        if len(frames_jpeg) > self._cfg.max_images:
            raise ValueError(
                f"{self._cfg.model} accepts at most {self._cfg.max_images} images, "
                f"got {len(frames_jpeg)}"
            )

        if self._max_spend is not None and self._spend.total_usd >= self._max_spend:
            raise SpendLimitExceeded(
                f"Spend cap of ${self._max_spend:.2f} reached (spent "
                f"${self._spend.total_usd:.2f}); refusing to make another call. "
                f"Raise it via the max_spend_usd argument, or resume this run later "
                f"— completed windows are already stored and will be skipped."
            )

        # Instruction first, then LABEL-then-IMAGE pairs. The per-frame text
        # label is not decoration: the caption prompt asks the model to
        # return start_frame/end_frame referring to these markers. Sending
        # bare images (as an earlier version of this file did) leaves every
        # temporal index the model returns completely unanchored.
        content: list[dict] = [{"type": "text", "text": prompt}]
        for i, f in enumerate(frames_jpeg):
            content.append({"type": "text", "text": frame_label(i)})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64.b64encode(f).decode()}"
                    },
                }
            )

        body: dict = {
            "model": self._cfg.model,
            "messages": [{"role": "user", "content": content}],
            # Both of these were previously declared in config.py and never
            # sent. max_tokens absent meant the provider default applied, so
            # the truncation headroom config documents was inert; temperature
            # absent meant provider-default sampling, which contaminates the
            # two-model agreement metric with sampling variance.
            "max_tokens": config.VLM_MAX_NEW_TOKENS,
            "temperature": config.VLM_TEMPERATURE,
        }
        if self._cfg.provider_order:
            body["provider"] = {"order": self._cfg.provider_order, "allow_fallbacks": False}
        reasoning: dict[str, object] = {}
        if self._cfg.reasoning_enabled is not None:
            reasoning["enabled"] = self._cfg.reasoning_enabled
        if self._cfg.reasoning_effort is not None:
            reasoning["effort"] = self._cfg.reasoning_effort
        if self._cfg.reasoning_exclude is not None:
            reasoning["exclude"] = self._cfg.reasoning_exclude
        if reasoning:
            body["reasoning"] = reasoning

        last_exc: Exception | None = None
        for attempt in range(self._retries + 1):
            t0 = time.monotonic()
            try:
                r = self._client.post("/chat/completions", json=body)
                if r.status_code in _NON_RETRYABLE_STATUS:
                    # 401/403/400 etc. will fail identically on every retry.
                    # Retrying burns 1s + 2s of wall clock per window for
                    # nothing, and on a bad key that is the whole run.
                    raise PermanentBackendError(
                        f"{self._cfg.model_id}: HTTP {r.status_code} from "
                        f"{self._cfg.base_url} — not retryable. "
                        f"{'Check your API key.' if r.status_code in (401, 403) else ''} "
                        f"Body: {r.text[:200]}"
                    )
                r.raise_for_status()
                data = r.json()
                latency_ms = int((time.monotonic() - t0) * 1000)

                # Account for spend BEFORE destructuring the response. A 200
                # whose body is an error object (OpenRouter does this for some
                # upstream failures) still burned billed tokens; charging only
                # on the success path let those escape the budget entirely.
                usage = data.get("usage", {})
                input_tokens = usage.get("prompt_tokens")
                output_tokens = usage.get("completion_tokens")
                cost_usd = self._estimate_cost(input_tokens, output_tokens)
                if cost_usd is None and self._max_spend is not None:
                    # Do not silently treat "provider told us nothing" as $0.
                    raise PermanentBackendError(
                        f"{self._cfg.model_id}: response carried no usage block, so "
                        f"cost cannot be accounted against max_spend_usd. Refusing to "
                        f"continue with an unenforceable spend cap."
                    )
                self._spend.add(cost_usd or 0.0)

                choices = data.get("choices")
                if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                    raise PermanentBackendError(
                        f"{self._cfg.model_id}: response has no usable choices entry"
                    )
                choice = choices[0]
                message = choice.get("message")
                if not isinstance(message, dict):
                    raise PermanentBackendError(
                        f"{self._cfg.model_id}: response choice has no usable message"
                    )
                text = message.get("content")
                if not isinstance(text, str) or not text.strip():
                    reasoning = message.get("reasoning")
                    raise PermanentBackendError(
                        f"{self._cfg.model_id}: provider returned no final text content "
                        f"(finish_reason={choice.get('finish_reason')!r}, "
                        f"reasoning_present={bool(reasoning)}). Refusing to parse reasoning "
                        "as a caption; lower or disable reasoning for this model."
                    )
                return VLMResponse(
                    text=text,
                    latency_ms=latency_ms,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost_usd,
                    provider=data.get("provider"),
                    model_version=data.get("model"),
                )
            except PermanentBackendError:
                raise
            except (httpx.HTTPError, KeyError, ValueError, IndexError) as e:
                last_exc = e
                if attempt < self._retries:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"VLM call failed after {self._retries + 1} attempts: {last_exc}")

    def _estimate_cost(self, input_tokens: int | None, output_tokens: int | None) -> float | None:
        if input_tokens is None or output_tokens is None:
            return None
        return (
            input_tokens * self._cfg.price_in_per_mtok
            + output_tokens * self._cfg.price_out_per_mtok
        ) / 1_000_000


class SpendTracker:
    """Shared across parallel workers hitting the same spend cap.

    `total_usd += amount` is a read-modify-write and is NOT atomic under
    threads, so a lock is required for the "shared across workers" claim in
    this docstring to be true rather than aspirational.

    Note the guard itself remains check-then-act at the call site: N workers
    can each pass the budget check before any of them records its cost, so
    the cap can be overshot by up to (in-flight calls x cost-per-call). That
    is acceptable for a per-window cost measured in fractions of a cent; it
    would not be for a coarse-grained one.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._total_usd = 0.0

    @property
    def total_usd(self) -> float:
        with self._lock:
            return self._total_usd

    def add(self, amount: float) -> None:
        with self._lock:
            self._total_usd += amount
