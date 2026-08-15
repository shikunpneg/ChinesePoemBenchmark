"""LLM-as-judge baseline (方案 §3.2 required reference).

Two implementations:

  1. `LLMJudgeStub` — pure-feature heuristic that mimics typical LLM behavior
     ("is this poem-like?"). Used when no API key is available.
     Documented as NOT a real LLM judge.

  2. `LLMJudgeAPI` — interface placeholder for the real DeepSeek-V4-Flash
     API. Will raise `NotImplementedError` until configured. The interface
     is designed so a real implementation is a drop-in replacement.

Both expose the same `predict_batch(texts) -> list[int]` so they can be
swapped in `eval_round_v3.py` via `LLM_JUDGE = LLMJudgeStub()` or
`LLMJudgeAPI(api_key="...", model="deepseek-v4-flash")`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .features import (
    extract_all_features,
    text_reliability,
)


@dataclass
class LLMJudgeResult:
    label: int          # 0 or 1
    confidence: float   # [0, 1]
    raw: str | None = None


class LLMJudgeBase(ABC):
    """Abstract base for any LLM-as-judge backend."""

    name: str = "base"

    @abstractmethod
    def predict_batch(self, texts: list[str]) -> list[LLMJudgeResult]:
        """Predict (poem / non-poem) for a list of texts."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


class LLMJudgeStub(LLMJudgeBase):
    """Honest placeholder for LLM-as-judge when DEEPSEEK_API_KEY is unavailable.

    This is NOT a real LLM. The default behavior is:

      - `mode="majority"`: predict the **training-set majority class** for every
        sample, with the same short-text gate as our metric. Establishes the
        "no-information" floor that any real LLM should beat.

    Once `DEEPSEEK_API_KEY` is set, `LLMJudgeAPI` will be used instead.
    This stub is here to validate the *pipeline* (prompt, parse, score)
    end-to-end so that adding the real API is a drop-in change.
    """

    name = "stub-majority"

    def __init__(self, mode: str = "majority",
                 default_label: int = 1) -> None:
        if mode not in ("majority", "majority_gated"):
            raise ValueError(f"unknown stub mode: {mode}")
        self.mode = mode
        self.default_label = default_label

    def predict_batch(self, texts: list[str]) -> list[LLMJudgeResult]:
        out = []
        for t in texts:
            rel = text_reliability(t)
            if self.mode == "majority_gated" and rel["is_truncatable"]:
                # mirror our metric's short-text handling
                out.append(LLMJudgeResult(
                    label=0, confidence=0.5, raw="truncatable->0"))
                continue
            out.append(LLMJudgeResult(
                label=self.default_label, confidence=0.5,
                raw=f"majority={self.default_label}"))
        return out


class LLMJudgeAPI(LLMJudgeBase):
    """Real DeepSeek-V4-Flash API judge.

    NOT IMPLEMENTED YET. Requires:
      - DEEPSEEK_API_KEY environment variable (or pass api_key=...)
      - Network access to DeepSeek API endpoint

    The interface is here so a real implementation can be a drop-in:
        LLM_JUDGE = LLMJudgeAPI(api_key="...", model="deepseek-v4-flash")
    """

    name = "deepseek-api"

    def __init__(self,
                 api_key: str | None = None,
                 model: str = "deepseek-v4-flash",
                 base_url: str = "https://api.deepseek.com/v1",
                 system_prompt: str | None = None) -> None:
        import os
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.model = model
        self.base_url = base_url
        self.system_prompt = system_prompt or (
            "你是一位中文诗歌判断专家。请根据文本的格式、语言凝练程度、"
            "意象运用与韵律节奏，判断它是否属于诗歌。"
            "只回答 0（非诗歌）或 1（诗歌），不要输出其他文字。")

    def predict_batch(self, texts: list[str]) -> list[LLMJudgeResult]:
        raise NotImplementedError(
            "LLMJudgeAPI not implemented: requires DEEPSEEK_API_KEY. "
            "Use LLMJudgeStub as a placeholder, or implement against the "
            "OpenAI-compatible DeepSeek endpoint.")


def default_judge() -> LLMJudgeBase:
    """Return the best available judge for the current environment."""
    api = LLMJudgeAPI()
    if api.api_key:
        return api
    return LLMJudgeStub()