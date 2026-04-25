import json
from pathlib import Path

from domain.enums.incident_type import IncidentType
from domain.enums.severity_level import SeverityLevel
from infrastructure.nlp.regex_parser import RegexParser


def load_sample_call() -> dict:
    sample_call_path = Path("src/contracts/examples/sample_call.json")
    with sample_call_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_expected_output() -> dict:
    expected_output_path = Path("src/contracts/examples/sample_extraction_output.json")
    with expected_output_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def test_regex_parser_matches_expected_sample_output() -> None:
    sample_call = load_sample_call()
    expected_output = load_expected_output()

    parser = RegexParser()

    actual_output = parser.parse(
        call_id=sample_call["call_id"],
        timestamp=sample_call["timestamp"],
        transcript=sample_call["transcript"],
    ).model_dump()

    assert actual_output == expected_output


def test_regex_parser_extracts_fire_incident_type() -> None:
    parser = RegexParser()

    result = parser.parse(
        call_id="CALL_TEST_001",
        timestamp="2026-04-23T16:00:00Z",
        transcript="There is a building on fire near the metro station. Send help quickly.",
    )

    assert result.incident.type == IncidentType.FIRE


def test_regex_parser_extracts_flood_incident_type() -> None:
    parser = RegexParser()

    result = parser.parse(
        call_id="CALL_TEST_002",
        timestamp="2026-04-23T16:02:00Z",
        transcript="Water is rising fast and the road is submerged near sector 14 market.",
    )

    assert result.incident.type == IncidentType.FLOOD


def test_regex_parser_extracts_gas_leak_incident_type() -> None:
    parser = RegexParser()

    result = parser.parse(
        call_id="CALL_TEST_003",
        timestamp="2026-04-23T16:04:00Z",
        transcript="There is a strong gas smell in the apartment building. People are fainting.",
    )

    assert result.incident.type == IncidentType.GAS_LEAK


def test_regex_parser_extracts_critical_severity_when_people_are_trapped() -> None:
    parser = RegexParser()

    result = parser.parse(
        call_id="CALL_TEST_004",
        timestamp="2026-04-23T16:05:00Z",
        transcript="There is a fire in the building and people are trapped inside. Help immediately.",
    )

    assert result.incident.severity == SeverityLevel.CRITICAL


def test_regex_parser_extracts_high_severity_for_serious_accident() -> None:
    parser = RegexParser()

    result = parser.parse(
        call_id="CALL_TEST_005",
        timestamp="2026-04-23T16:06:00Z",
        transcript="A serious accident happened near the flyover and the road is blocked.",
    )

    assert result.incident.severity in {SeverityLevel.HIGH, SeverityLevel.CRITICAL}


def test_regex_parser_extracts_location_phrase_near() -> None:
    parser = RegexParser()

    result = parser.parse(
        call_id="CALL_TEST_006",
        timestamp="2026-04-23T16:10:00Z",
        transcript="There has been an accident near City Hospital, please send an ambulance fast.",
    )

    assert result.location.raw_text == "near city hospital"


def test_regex_parser_extracts_location_phrase_beside() -> None:
    parser = RegexParser()

    result = parser.parse(
        call_id="CALL_TEST_007",
        timestamp="2026-04-23T16:12:00Z",
        transcript="There is smoke beside the old bus stand hurry please.",
    )

    assert result.location.raw_text == "beside the old bus stand"


def test_regex_parser_extracts_word_based_injured_count() -> None:
    parser = RegexParser()

    result = parser.parse(
        call_id="CALL_TEST_008",
        timestamp="2026-04-23T16:14:00Z",
        transcript="Two people injured in a bus accident near central chowk.",
    )

    assert result.casualties is not None
    assert result.casualties.injured_count == 2


def test_regex_parser_extracts_digit_based_dead_count() -> None:
    parser = RegexParser()

    result = parser.parse(
        call_id="CALL_TEST_009",
        timestamp="2026-04-23T16:16:00Z",
        transcript="2 people killed in the collapse near river bridge.",
    )

    assert result.casualties is not None
    assert result.casualties.dead_count == 2


def test_regex_parser_generates_event_signature() -> None:
    parser = RegexParser()

    result = parser.parse(
        call_id="CALL_TEST_010",
        timestamp="2026-04-23T16:15:00Z",
        transcript="There is a fire near Green Park Metro Station. Please send help quickly.",
    )

    assert result.deduplication is not None
    assert result.deduplication.event_signature == "fire|near_green_park_metro_station"


def test_regex_parser_marks_missing_location_when_not_found() -> None:
    parser = RegexParser()

    result = parser.parse(
        call_id="CALL_TEST_011",
        timestamp="2026-04-23T16:18:00Z",
        transcript="There is a fire and people are trapped please send help immediately.",
    )

    assert result.location.raw_text == "unknown"
    assert "location.raw_text" in result.extraction_metadata.missing_fields


def test_regex_parser_returns_unknown_for_unclear_incident() -> None:
    parser = RegexParser()

    result = parser.parse(
        call_id="CALL_TEST_012",
        timestamp="2026-04-23T16:20:00Z",
        transcript="Something bad has happened here please come quickly.",
    )

    assert result.incident.type == IncidentType.UNKNOWN