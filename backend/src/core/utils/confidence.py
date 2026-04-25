from domain.enums.incident_type import IncidentType
from domain.enums.severity_level import SeverityLevel


def compute_extraction_confidence(
    *,
    incident_type: IncidentType,
    severity: SeverityLevel,
    location_found: bool,
) -> float:
    """
    Compute a simple deterministic confidence score for structured extraction.

    This is version 1 baseline logic.
    Later it can become weighted or model-driven.
    """

    score = 0.20

    if incident_type != IncidentType.UNKNOWN:
        score += 0.30

    if severity != SeverityLevel.UNKNOWN:
        score += 0.25

    if location_found:
        score += 0.20

    return round(min(score, 0.95), 2)