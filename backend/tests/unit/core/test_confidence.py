from core.utils.confidence import compute_extraction_confidence
from domain.enums.incident_type import IncidentType
from domain.enums.severity_level import SeverityLevel


def test_confidence_is_high_when_incident_severity_and_location_are_found() -> None:
    result = compute_extraction_confidence(
        incident_type=IncidentType.FIRE,
        severity=SeverityLevel.CRITICAL,
        location_found=True,
    )

    assert result == 0.95


def test_confidence_drops_when_location_is_missing() -> None:
    result = compute_extraction_confidence(
        incident_type=IncidentType.FIRE,
        severity=SeverityLevel.HIGH,
        location_found=False,
    )

    assert result == 0.75


def test_confidence_is_lower_when_severity_is_unknown() -> None:
    result = compute_extraction_confidence(
        incident_type=IncidentType.ROAD_ACCIDENT,
        severity=SeverityLevel.UNKNOWN,
        location_found=True,
    )

    assert result == 0.7


def test_confidence_is_minimum_when_everything_is_unknown_or_missing() -> None:
    result = compute_extraction_confidence(
        incident_type=IncidentType.UNKNOWN,
        severity=SeverityLevel.UNKNOWN,
        location_found=False,
    )

    assert result == 0.2