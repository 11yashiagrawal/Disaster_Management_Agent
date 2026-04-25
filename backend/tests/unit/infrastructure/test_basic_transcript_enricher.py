from domain.enums.incident_type import IncidentType
from domain.enums.severity_level import SeverityLevel
from infrastructure.nlp.basic_transcript_enricher import BasicTranscriptEnricher


def test_enricher_infers_fire_from_messy_transcript() -> None:
    enricher = BasicTranscriptEnricher()

    result = enricher.enrich(
        transcript="Please come fast there is too much smoke and fire near the metro side building people shouting"
    )

    assert result.inferred_incident_type == IncidentType.FIRE
    assert "fire" in result.possible_hazards


def test_enricher_infers_critical_when_people_are_trapped() -> None:
    enricher = BasicTranscriptEnricher()

    result = enricher.enrich(
        transcript="Building burning and children inside, people trapped, come immediately"
    )

    assert result.inferred_severity == SeverityLevel.CRITICAL
    assert "trapped" in result.possible_casualty_clues


def test_enricher_extracts_location_hints() -> None:
    enricher = BasicTranscriptEnricher()

    result = enricher.enrich(
        transcript="One bus and car hit badly near city hospital opposite old market please send help"
    )

    assert any("city hospital" in loc for loc in result.possible_locations)


def test_enricher_infers_gas_leak_from_weak_phrase() -> None:
    enricher = BasicTranscriptEnricher()

    result = enricher.enrich(
        transcript="Some smell of gas in the apartment and people fainted please hurry"
    )

    assert result.inferred_incident_type == IncidentType.GAS_LEAK
    assert "gas" in result.possible_hazards


def test_enricher_collects_urgency_clues() -> None:
    enricher = BasicTranscriptEnricher()

    result = enricher.enrich(
        transcript="Please help fast there is smoke beside the school hurry"
    )

    assert "please" in result.urgency_clues
    assert "help fast" in result.urgency_clues or "hurry" in result.urgency_clues


def test_enricher_returns_normalized_transcript() -> None:
    enricher = BasicTranscriptEnricher()

    result = enricher.enrich(
        transcript="   THERE is   Smoke near   Sector 14 Market   "
    )

    assert result.normalized_transcript == "there is smoke near sector 14 market"