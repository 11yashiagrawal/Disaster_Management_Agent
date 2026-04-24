from src.application.interfaces.llm_client import ChatMessage, LLMClient
from src.domain.enums.incident_type import IncidentType
from src.domain.enums.severity_level import SeverityLevel
from src.infrastructure.nlp.semantic_triage_parser import SemanticTriageParser


class FakeLLMClient(LLMClient):
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.messages: list[ChatMessage] = []

    def complete(self, *, messages: list[ChatMessage], temperature: float = 0.0) -> str:
        self.messages = messages
        return self.response_text


def test_semantic_triage_parser_builds_structured_output_from_llm_json() -> None:
    response_text = """
    {
      "call_id": "ignored",
      "timestamp": "ignored",
      "raw_transcript": "ignored",
      "incident": {
        "type": "fire",
        "subtype": "building_fire",
        "severity": "critical",
        "description": "A building fire with trapped people"
      },
      "location": {
        "raw_text": "near green park metro station",
        "landmark": "green park metro station",
        "address": null,
        "area": null,
        "city": null,
        "coordinates": {"lat": null, "lng": null},
        "confidence": 0.92
      },
      "casualties": {
        "injured_count": 2,
        "dead_count": null,
        "people_trapped": true
      },
      "hazards": ["fire", "smoke"],
      "resources_needed": {
        "ambulance": 2,
        "fire_truck": 3,
        "police_unit": 1,
        "rescue_team": 1
      },
      "caller_context": {
        "caller_role": "bystander",
        "emotional_state": "panicked",
        "callback_number": null
      },
      "extraction_metadata": {
        "missing_fields": [],
        "contradictions_detected": [],
        "overall_confidence": 0.94
      },
      "deduplication": {
        "event_signature": "fire|near_green_park_metro_station",
        "possible_duplicate_of": null
      }
    }
    """

    client = FakeLLMClient(response_text=response_text)
    parser = SemanticTriageParser(client=client)

    result = parser.parse(
        call_id="CALL_LLM_001",
        timestamp="2026-04-24T18:00:00Z",
        transcript="There is a building on fire near Green Park Metro Station. People are trapped.",
    )

    assert result.call_id == "CALL_LLM_001"
    assert result.timestamp == "2026-04-24T18:00:00Z"
    assert result.raw_transcript.startswith("There is a building on fire")
    assert result.incident.type == IncidentType.FIRE
    assert result.incident.severity == SeverityLevel.CRITICAL
    assert result.location.raw_text == "near green park metro station"
    assert result.resources_needed.fire_truck == 3
    assert result.deduplication is not None
    assert result.deduplication.event_signature == "fire|near_green_park_metro_station"

    assert client.messages[0]["role"] == "system"
    assert "JSON SCHEMA" in client.messages[1]["content"]
    assert "panic_score:" in client.messages[1]["content"]


def test_semantic_triage_parser_uses_raw_transcript_for_panic_scoring() -> None:
    response_text = """
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
        "raw_text": "unknown",
        "landmark": null,
        "address": null,
        "area": null,
        "city": null,
        "coordinates": {"lat": null, "lng": null},
        "confidence": 0.2
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
        "overall_confidence": 0.3
      },
      "deduplication": {
        "event_signature": "fire|unknown",
        "possible_duplicate_of": null
      }
    }
    """

    client = FakeLLMClient(response_text=response_text)
    parser = SemanticTriageParser(client=client)

    parser.parse(
        call_id="CALL_LLM_002",
        timestamp="2026-04-24T18:30:00Z",
        transcript="REPAIRED TEXT ONLY",
        raw_transcript="THIS IS A PANIC CALL PLEASE HELP!!",
    )

    assert "THIS IS A PANIC CALL PLEASE HELP!!" in client.messages[1]["content"]