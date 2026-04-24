import json
import re
from typing import Optional

from src.application.dto.extraction_result_dto import ExtractionResultDTO
from src.application.interfaces.llm_client import ChatMessage, LLMClient
from src.application.interfaces.nlp_parser import NLPParser
from src.infrastructure.nlp.regex_parser import RegexParser


class SemanticTriageParser(NLPParser):
    """LLM-first triage parser that turns noisy transcripts into strict JSON."""

    def __init__(self, *, client: LLMClient, fallback_parser: Optional[NLPParser] = None) -> None:
        self.client = client
        self.fallback_parser = fallback_parser or RegexParser()

    def parse(
        self,
        *,
        call_id: str,
        timestamp: str,
        transcript: str,
    ) -> ExtractionResultDTO:
        messages = self._build_messages(call_id=call_id, timestamp=timestamp, transcript=transcript)

        try:
            response_text = self.client.complete(messages=messages, temperature=0.0)
            payload = self._extract_json(response_text)
            result = ExtractionResultDTO.model_validate(payload)
            result.call_id = call_id
            result.timestamp = timestamp
            result.raw_transcript = transcript
            return result
        except Exception:
            if self.fallback_parser is not None:
                return self.fallback_parser.parse(
                    call_id=call_id,
                    timestamp=timestamp,
                    transcript=transcript,
                )
            raise

    def _build_messages(self, *, call_id: str, timestamp: str, transcript: str) -> list[ChatMessage]:
        schema = json.dumps(ExtractionResultDTO.model_json_schema(), indent=2)

        system_message = (
            "You are the triage agent for a smart-city emergency dispatch system. "
            "Read one transcript and infer the incident using semantic understanding, context, and intent. "
            "Return only valid JSON that matches the supplied schema. "
            "Do not explain your reasoning, do not add markdown, and do not invent facts. "
            "If a field is missing or unclear, use unknown, null, or an empty list as appropriate. "
            "Treat the transcript as a real emergency call, including contradictory or noisy phrasing."
        )

        user_message = (
            f"CALL METADATA\n"
            f"call_id: {call_id}\n"
            f"timestamp: {timestamp}\n\n"
            f"TRANSCRIPT\n{transcript}\n\n"
            f"JSON SCHEMA\n{schema}"
        )

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

    def _extract_json(self, response_text: str) -> dict:
        text = response_text.strip()

        fenced_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if fenced_match:
            text = fenced_match.group(1).strip()

        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("LLM response did not contain a JSON object")

        return payload