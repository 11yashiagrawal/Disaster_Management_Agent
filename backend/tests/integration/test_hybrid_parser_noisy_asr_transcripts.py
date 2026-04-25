import pytest

from domain.enums.incident_type import IncidentType
from domain.enums.severity_level import SeverityLevel
from infrastructure.nlp.hybrid_parser import HybridParser
from infrastructure.nlp.spacy_transcript_enricher import SpacyTranscriptEnricher


@pytest.fixture(scope="module")
def parser() -> HybridParser:
    return HybridParser(enricher=SpacyTranscriptEnricher())


NOISY_ASR_CASES = [
    {
        "name": "fire_with_gibberish_and_ellipsis",
        "transcript": (
            "aaaaaa please come fast.... smok and fire rhifh near green park metro... "
            "people inside maybe... hurry hurry"
        ),
        "acceptable_incident_types": {IncidentType.FIRE},
        "acceptable_severities": {
            SeverityLevel.HIGH,
            SeverityLevel.CRITICAL,
            SeverityLevel.MEDIUM,
        },
        "should_have_location": True,
    },
    {
        "name": "road_accident_with_asr_typos",
        "transcript": (
            "bus and kar hit badli near sity hospital.... rood blockd.... two injrd please send amblance"
        ),
        "acceptable_incident_types": {
            IncidentType.ROAD_ACCIDENT,
            IncidentType.MEDICAL_EMERGENCY,
        },
        "acceptable_severities": {
            SeverityLevel.HIGH,
            SeverityLevel.CRITICAL,
            SeverityLevel.MEDIUM,
        },
        "should_have_location": True,
    },
    {
        "name": "gas_leak_with_corrupted_words",
        "transcript": (
            "gass smel in apartmant aaaaa people faintng cant breth please hurry..."
        ),
        "acceptable_incident_types": {
            IncidentType.GAS_LEAK,
            IncidentType.MEDICAL_EMERGENCY,
            IncidentType.UNKNOWN,
        },
        "acceptable_severities": {
            SeverityLevel.HIGH,
            SeverityLevel.CRITICAL,
            SeverityLevel.MEDIUM,
            SeverityLevel.UNKNOWN,
        },
        "should_have_location": False,
    },
    {
        "name": "collapse_with_fragmented_phrases",
        "transcript": (
            "bilding colapsd... near river brij... rubbl everywhere... people trap... send rescue now now"
        ),
        "acceptable_incident_types": {
            IncidentType.BUILDING_COLLAPSE,
            IncidentType.UNKNOWN,
        },
        "acceptable_severities": {
            SeverityLevel.CRITICAL,
            SeverityLevel.HIGH,
            SeverityLevel.UNKNOWN,
        },
        "should_have_location": True,
    },
    {
        "name": "flood_with_noise_tokens",
        "transcript": (
            "uhh water rising aaaaa near sector 14 markit.... road submrged... vhicles stuck..."
        ),
        "acceptable_incident_types": {
            IncidentType.FLOOD,
            IncidentType.UNKNOWN,
        },
        "acceptable_severities": {
            SeverityLevel.MEDIUM,
            SeverityLevel.HIGH,
            SeverityLevel.UNKNOWN,
        },
        "should_have_location": True,
    },
    {
        "name": "medical_emergency_with_broken_speech",
        "transcript": (
            "man unconsious.... outside skul.... not breathng properli... please come fast"
        ),
        "acceptable_incident_types": {
            IncidentType.MEDICAL_EMERGENCY,
            IncidentType.UNKNOWN,
        },
        "acceptable_severities": {
            SeverityLevel.CRITICAL,
            SeverityLevel.HIGH,
            SeverityLevel.UNKNOWN,
        },
        "should_have_location": False,
    },
    {
        "name": "unknown_with_gibberish_dominance",
        "transcript": (
            "aaaa rhifh gibbersih.... something bad.... xxxxx people shouting.... come fast....."
        ),
        "acceptable_incident_types": {
            IncidentType.UNKNOWN,
        },
        "acceptable_severities": {
            SeverityLevel.UNKNOWN,
            SeverityLevel.HIGH,
        },
        "should_have_location": False,
    },
    {
        "name": "landmark_heavy_fire_with_misspellings",
        "transcript": (
            "shop burnng opposite old boz stand near citi hospitl.... smok everywhere... help fast"
        ),
        "acceptable_incident_types": {
            IncidentType.FIRE,
            IncidentType.UNKNOWN,
        },
        "acceptable_severities": {
            SeverityLevel.HIGH,
            SeverityLevel.CRITICAL,
            SeverityLevel.MEDIUM,
            SeverityLevel.UNKNOWN,
        },
        "should_have_location": True,
    },
    {
        "name": "stuttered_fire_report",
        "transcript": (
            "fire fire fire please please come quick quick near metro station people trap maybe"
        ),
        "acceptable_incident_types": {
            IncidentType.FIRE,
        },
        "acceptable_severities": {
            SeverityLevel.CRITICAL,
            SeverityLevel.HIGH,
            SeverityLevel.MEDIUM,
        },
        "should_have_location": True,
    },
    {
        "name": "partial_dropout_accident_case",
        "transcript": (
            "car.... bus.... hit.... near flyover.... three people injrd.... road blocked"
        ),
        "acceptable_incident_types": {
            IncidentType.ROAD_ACCIDENT,
            IncidentType.UNKNOWN,
        },
        "acceptable_severities": {
            SeverityLevel.HIGH,
            SeverityLevel.CRITICAL,
            SeverityLevel.MEDIUM,
            SeverityLevel.UNKNOWN,
        },
        "should_have_location": True,
    },
]


@pytest.mark.parametrize("case", NOISY_ASR_CASES, ids=[c["name"] for c in NOISY_ASR_CASES])
def test_hybrid_parser_handles_noisy_asr_like_transcripts(
    parser: HybridParser,
    case: dict,
) -> None:
    result = parser.parse(
        call_id=f"CALL_NOISY_{case['name'].upper()}",
        timestamp="2026-04-23T22:00:00Z",
        transcript=case["transcript"],
    )

    assert result.call_id.startswith("CALL_NOISY_")
    assert result.raw_transcript == case["transcript"]
    assert result.incident is not None
    assert result.location is not None
    assert result.resources_needed is not None
    assert result.extraction_metadata is not None

    assert result.incident.type in case["acceptable_incident_types"]
    assert result.incident.severity in case["acceptable_severities"]

    if case["should_have_location"]:
        assert result.location.raw_text != "unknown"

    assert 0.0 <= result.extraction_metadata.overall_confidence <= 1.0

    assert result.deduplication is not None
    assert "|" in result.deduplication.event_signature


def test_noisy_transcript_suite_never_breaks_output_shape(parser: HybridParser) -> None:
    transcript = "aaaaa.... rhifh.... smoke maybe near metro.... people inside maybe hurry"

    result = parser.parse(
        call_id="CALL_NOISY_SHAPE",
        timestamp="2026-04-23T22:10:00Z",
        transcript=transcript,
    )

    assert isinstance(result.call_id, str)
    assert isinstance(result.raw_transcript, str)
    assert isinstance(result.location.raw_text, str)
    assert isinstance(result.hazards, list)
    assert isinstance(result.extraction_metadata.missing_fields, list)
    assert result.resources_needed.ambulance >= 0
    assert result.resources_needed.fire_truck >= 0
    assert result.resources_needed.police_unit >= 0
    assert result.resources_needed.rescue_team >= 0