try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional convenience dependency
    load_dotenv = None

from src.application.interfaces.nlp_parser import NLPParser
from src.infrastructure.llm.groq_chat_client import GroqChatClient
from src.infrastructure.nlp.basic_transcript_enricher import BasicTranscriptEnricher
from src.infrastructure.nlp.hybrid_parser import HybridParser
from src.infrastructure.nlp.regex_parser import RegexParser
from src.infrastructure.nlp.semantic_triage_parser import SemanticTriageParser


try:
    from src.infrastructure.nlp.spacy_transcript_enricher import SpacyTranscriptEnricher
except Exception:  # pragma: no cover - keep factory usable if spaCy is unavailable
    SpacyTranscriptEnricher = None


def build_triage_parser() -> NLPParser:
    if load_dotenv is not None:
        load_dotenv()

    client = GroqChatClient.from_env()
    enricher = SpacyTranscriptEnricher() if SpacyTranscriptEnricher is not None else BasicTranscriptEnricher()
    llm_parser = SemanticTriageParser(client=client, fallback_parser=RegexParser())

    return HybridParser(enricher=enricher, llm_parser=llm_parser, fallback_parser=RegexParser())