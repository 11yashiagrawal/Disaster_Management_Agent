from src.domain.enums.incident_type import IncidentType
from src.domain.enums.severity_level import SeverityLevel


def compute_extraction_confidence(
    *,
    incident_type: IncidentType,
    severity: SeverityLevel,
    location_found: bool,
    has_casualties: bool = False,
    has_resources: bool = False,
) -> float:
    """
    Compute a simple deterministic confidence score for structured extraction.

    This version incorporates more of the structured DTO signal.
    """

    score = 0.15

    if incident_type != IncidentType.UNKNOWN:
        score += 0.25

    if severity != SeverityLevel.UNKNOWN:
        score += 0.20

    if location_found:
        score += 0.20

    if has_casualties:
        score += 0.10

    if has_resources:
        score += 0.10

    return round(min(score, 0.95), 2)