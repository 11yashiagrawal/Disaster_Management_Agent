from abc import ABC, abstractmethod
from typing import TypedDict


class ChatMessage(TypedDict):
    role: str
    content: str


class LLMClient(ABC):
    @abstractmethod
    def complete(self, *, messages: list[ChatMessage], temperature: float = 0.0) -> str:
        raise NotImplementedError