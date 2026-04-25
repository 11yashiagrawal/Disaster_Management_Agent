from typing import List

from application.dto.extraction_result_dto import ExtractionResultDTO
from domain.enums.incident_type import IncidentType
from domain.enums.severity_level import SeverityLevel


class SafetyOverrideService:
    """
    Deterministic safety layer.

    Even if model probability is moderate, life-threatening conditions
    must upgrade priority.
    """

    def get_override_reasons(self, extraction: ExtractionResultDTO) -> List[str]:
        reasons: List[str] = []

        if extraction.incident.severity == SeverityLevel.CRITICAL:
            reasons.append("Critical severity reported")

        if extraction.casualties and extraction.casualties.people_trapped:
            reasons.append("People may be trapped")

        if extraction.incident.type in {
            IncidentType.BUILDING_COLLAPSE,
            IncidentType.GAS_LEAK,
            IncidentType.EARTHQUAKE_DAMAGE,
        }:
            reasons.append(f"High-risk incident category: {extraction.incident.type.value}")

        if "gas" in extraction.hazards:
            reasons.append("Gas hazard detected")

        if "smoke" in extraction.hazards and extraction.incident.type == IncidentType.FIRE:
            reasons.append("Smoke present during fire incident")

        return reasons

    def requires_escalation(self, extraction: ExtractionResultDTO) -> bool:
        return len(self.get_override_reasons(extraction)) > 0