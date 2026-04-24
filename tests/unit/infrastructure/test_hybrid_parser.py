from src.application.dto.enriched_transcript_dto import EnrichedTranscriptDTO
from src.application.interfaces.llm_client import ChatMessage, LLMClient
from src.application.interfaces.transcript_enricher import TranscriptEnricher
from src.infrastructure.nlp.basic_transcript_enricher import BasicTranscriptEnricher
from src.domain.enums.incident_type import IncidentType
from src.infrastructure.nlp.hybrid_parser import HybridParser
from src.infrastructure.nlp.semantic_triage_parser import SemanticTriageParser


class StubEnricher(TranscriptEnricher):
        def enrich(self, *, transcript: str) -> EnrichedTranscriptDTO:
                return EnrichedTranscriptDTO(
                        raw_transcript=transcript,
                        normalized_transcript=transcript,
                        inferred_incident_type=IncidentType.FIRE,
                        inferred_severity=None,
                        possible_locations=["near green park metro station"],
                        possible_hazards=["smoke"],
                        possible_casualty_clues=["trapped"],
                        urgency_clues=["urgent"],
                        extracted_signals=[],
                        enrichment_confidence=0.9,
                )


class CapturingLLMClient(LLMClient):
        def __init__(self) -> None:
                self.messages: list[ChatMessage] = []

        def complete(self, *, messages: list[ChatMessage], temperature: float = 0.0) -> str:
                self.messages = messages
                return """
                {
                    "call_id": "ignored",
                    "timestamp": "ignored",
                    "raw_transcript": "ignored",
                    "incident": {
                        "type": "fire",
                        "subtype": null,
                        "severity": "high",
                        "description": "Fire"
                    },
                    "location": {
                        "raw_text": "near green park metro station",
                        "landmark": null,
                        "address": null,
                        "area": null,
                        "city": null,
                        "coordinates": {"lat": null, "lng": null},
                        "confidence": 0.8
                    },
                    "casualties": null,
                    "hazards": [],
                    "resources_needed": {
                        "ambulance": 0,
                        "fire_truck": 0,
                        "police_unit": 0,
                        "rescue_team": 0
                    },
                    "caller_context": null,
                    "extraction_metadata": {
                        "missing_fields": [],
                        "contradictions_detected": [],
                        "overall_confidence": 0.5
                    },
                    "deduplication": {
                        "event_signature": "fire|near_green_park_metro_station",
                        "possible_duplicate_of": null
                    }
                }
                """


def test_hybrid_parser_extracts_fire_from_messy_transcript() -> None:
    parser = HybridParser(enricher=BasicTranscriptEnricher())

    result = parser.parse(
        call_id="CALL_HYB_001",
        timestamp="2026-04-23T20:00:00Z",
        transcript="Please come fast there is too much smoke near green park metro station people shouting",
    )

    assert result.incident.type == IncidentType.FIRE
    assert result.location.raw_text != "unknown"


def test_hybrid_parser_uses_enrichment_hints_for_gas_leak() -> None:
    parser = HybridParser(enricher=BasicTranscriptEnricher())

    result = parser.parse(
        call_id="CALL_HYB_002",
        timestamp="2026-04-23T20:05:00Z",
        transcript="Some smell in the apartment and people fainted please hurry",
    )

    assert result.incident.type in {IncidentType.GAS_LEAK, IncidentType.MEDICAL_EMERGENCY}


def test_hybrid_parser_preserves_structured_output_shape() -> None:
    parser = HybridParser(enricher=BasicTranscriptEnricher())

    result = parser.parse(
        call_id="CALL_HYB_003",
        timestamp="2026-04-23T20:10:00Z",
        transcript="Bus and car hit badly near city hospital opposite old market send help fast",
    )

    assert result.call_id == "CALL_HYB_003"
    assert result.incident is not None
    assert result.location is not None
    assert result.resources_needed is not None
    assert result.extraction_metadata is not None


def test_hybrid_parser_uses_llm_stage_when_provided() -> None:
    llm_client = CapturingLLMClient()
    llm_parser = SemanticTriageParser(client=llm_client)
    parser = HybridParser(enricher=StubEnricher(), llm_parser=llm_parser)

    result = parser.parse(
        call_id="CALL_HYB_004",
        timestamp="2026-04-23T20:15:00Z",
        transcript="THIS IS A PANIC CALL PLEASE HELP!!",
    )

    assert result.incident.type == IncidentType.FIRE
    assert "panic_score:" in llm_client.messages[1]["content"]