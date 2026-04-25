from domain.enums.incident_type import IncidentType
from infrastructure.nlp.hybrid_parser import HybridParser
from infrastructure.nlp.spacy_transcript_enricher import SpacyTranscriptEnricher


def test_hybrid_parser_extracts_fire_from_messy_transcript() -> None:
    parser = HybridParser(enricher=SpacyTranscriptEnricher())

    result = parser.parse(
        call_id="CALL_HYB_001",
        timestamp="2026-04-23T20:00:00Z",
        transcript="Please come fast there is too much smoke near green park metro station people shouting",
    )

    assert result.incident.type == IncidentType.FIRE
    assert result.location.raw_text != "unknown"


def test_hybrid_parser_uses_enrichment_hints_for_gas_leak() -> None:
    parser = HybridParser(enricher=SpacyTranscriptEnricher())

    result = parser.parse(
        call_id="CALL_HYB_002",
        timestamp="2026-04-23T20:05:00Z",
        transcript="Some smell in the apartment and people fainted please hurry",
    )

    assert result.incident.type in {IncidentType.GAS_LEAK, IncidentType.MEDICAL_EMERGENCY}


def test_hybrid_parser_preserves_structured_output_shape() -> None:
    parser = HybridParser(enricher=SpacyTranscriptEnricher())

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