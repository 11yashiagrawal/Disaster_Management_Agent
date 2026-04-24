from src.core.utils.confidence import compute_extraction_confidence
from src.domain.enums.incident_type import IncidentType
from src.domain.enums.severity_level import SeverityLevel


def test_confidence_is_high_when_incident_severity_and_location_are_found() -> None:
    result = compute_extraction_confidence(
        incident_type=IncidentType.FIRE,
        severity=SeverityLevel.CRITICAL,
        location_found=True,
    )

    assert result == 0.8


def test_confidence_drops_when_location_is_missing() -> None:
    result = compute_extraction_confidence(
        incident_type=IncidentType.FIRE,
        severity=SeverityLevel.HIGH,
        location_found=False,
    )

    assert result == 0.6


def test_confidence_is_lower_when_severity_is_unknown() -> None:
    result = compute_extraction_confidence(
        incident_type=IncidentType.ROAD_ACCIDENT,
        severity=SeverityLevel.UNKNOWN,
        location_found=True,
    )

    assert result == 0.6


def test_confidence_is_minimum_when_everything_is_unknown_or_missing() -> None:
    result = compute_extraction_confidence(
        incident_type=IncidentType.UNKNOWN,
        severity=SeverityLevel.UNKNOWN,
        location_found=False,
    )

    assert result == 0.15


def test_confidence_uses_casualties_and_resources_signals() -> None:
    result = compute_extraction_confidence(
        incident_type=IncidentType.FIRE,
        severity=SeverityLevel.CRITICAL,
        location_found=True,
        has_casualties=True,
        has_resources=True,
    )

    assert result == 0.95