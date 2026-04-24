try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional convenience dependency
    load_dotenv = None

from src.application.interfaces.nlp_parser import NLPParser
from src.infrastructure.llm.groq_chat_client import GroqChatClient
from src.infrastructure.nlp.regex_parser import RegexParser
from src.infrastructure.nlp.semantic_triage_parser import SemanticTriageParser


def build_triage_parser() -> NLPParser:
    if load_dotenv is not None:
        load_dotenv()

    client = GroqChatClient.from_env()

    return SemanticTriageParser(client=client, fallback_parser=RegexParser())