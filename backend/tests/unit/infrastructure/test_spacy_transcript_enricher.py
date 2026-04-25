from domain.enums.incident_type import IncidentType
from domain.enums.severity_level import SeverityLevel
from infrastructure.nlp.spacy_transcript_enricher import SpacyTranscriptEnricher


def test_spacy_enricher_infers_fire() -> None:
    enricher = SpacyTranscriptEnricher()

    result = enricher.enrich(
        transcript="Please come fast there is too much smoke and fire near Green Park Metro Station."
    )

    assert result.inferred_incident_type == IncidentType.FIRE
    assert "fire" in result.possible_hazards
    assert any("green park metro station" in loc for loc in result.possible_locations)


def test_spacy_enricher_infers_critical() -> None:
    enricher = SpacyTranscriptEnricher()

    result = enricher.enrich(
        transcript="Building burning and children inside, people trapped, come immediately."
    )

    assert result.inferred_severity == SeverityLevel.CRITICAL
    assert "trapped" in result.possible_casualty_clues


def test_spacy_enricher_extracts_landmark_location() -> None:
    enricher = SpacyTranscriptEnricher()

    result = enricher.enrich(
        transcript="Bus and car hit badly near City Hospital opposite old market."
    )

    assert any("city hospital" in loc for loc in result.possible_locations)


def test_spacy_enricher_detects_gas_leak() -> None:
    enricher = SpacyTranscriptEnricher()

    result = enricher.enrich(
        transcript="Some smell of gas in the apartment and people fainted please hurry."
    )

    assert result.inferred_incident_type == IncidentType.GAS_LEAK
    assert "gas" in result.possible_hazards