import os
import time
from dataclasses import dataclass

from groq import Groq

from src.application.interfaces.llm_client import ChatMessage, LLMClient


@dataclass(frozen=True)
class GroqChatClient(LLMClient):
    api_key: str
    model: str = "llama-3.1-8b-instant"
    base_url: str = "https://api.groq.com"
    max_retries: int = 3
    backoff_seconds: float = 1.0

    @classmethod
    def from_env(cls) -> "GroqChatClient":
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set")

        model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant").strip() or "llama-3.1-8b-instant"
        base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com").strip()
        if not base_url:
            base_url = "https://api.groq.com"

        max_retries_raw = os.getenv("GROQ_MAX_RETRIES", "3").strip()
        try:
            max_retries = max(1, int(max_retries_raw))
        except ValueError:
            max_retries = 3

        backoff_raw = os.getenv("GROQ_BACKOFF_SECONDS", "1").strip()
        try:
            backoff_seconds = max(0.1, float(backoff_raw))
        except ValueError:
            backoff_seconds = 1.0

        return cls(
            api_key=api_key,
            model=model,
            base_url=base_url,
            max_retries=max_retries,
            backoff_seconds=backoff_seconds,
        )

    def complete(self, *, messages: list[ChatMessage], temperature: float = 0.0) -> str:
        client = Groq(api_key=self.api_key, base_url=self.base_url)

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    stream=False,
                )

                choices = response.choices or []
                if not choices:
                    raise RuntimeError("Groq response did not include any choices")

                content = choices[0].message.content
                if not isinstance(content, str) or not content.strip():
                    raise RuntimeError("Groq response did not include assistant content")

                return content
            except Exception as exc:
                last_error = exc
                status = getattr(exc, "status_code", None)
                should_retry = status in {408, 429, 500, 502, 503, 504} or status is None
                if attempt == self.max_retries - 1 or not should_retry:
                    break
                time.sleep(self.backoff_seconds * (2**attempt))

        raise RuntimeError(f"Groq request failed after retries: {last_error}") from last_error