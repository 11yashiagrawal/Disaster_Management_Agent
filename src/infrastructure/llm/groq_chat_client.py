import json
import os
from dataclasses import dataclass
from typing import Any
from urllib import error, request

from src.application.interfaces.llm_client import ChatMessage, LLMClient


@dataclass(frozen=True)
class GroqChatClient(LLMClient):
    api_key: str
    model: str = "llama-3.1-8b-instant"
    base_url: str = "https://api.groq.com/openai/v1/chat/completions"

    @classmethod
    def from_env(cls) -> "GroqChatClient":
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set")

        model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant").strip() or "llama-3.1-8b-instant"
        base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1/chat/completions").strip()
        if not base_url:
            base_url = "https://api.groq.com/openai/v1/chat/completions"

        return cls(api_key=api_key, model=model, base_url=base_url)

    def complete(self, *, messages: list[ChatMessage], temperature: float = 0.0) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }

        req = request.Request(
            self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=60) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Groq request failed: {exc.code} {message}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Groq request failed: {exc.reason}") from exc

        choices = response_payload.get("choices") or []
        if not choices:
            raise RuntimeError("Groq response did not include any choices")

        message = choices[0].get("message") or {}
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Groq response did not include assistant content")

        return content