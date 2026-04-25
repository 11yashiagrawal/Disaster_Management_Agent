import math

from application.dto.extraction_result_dto import ExtractionResultDTO
from domain.enums.incident_type import IncidentType
from domain.enums.severity_level import SeverityLevel


class RiskScoringService:
    """
    Model-style risk scorer.

    This is a transparent logistic risk model:
    - converts extracted incident facts into numeric features
    - applies weighted scoring
    - converts score into probability using sigmoid

    Later this can be replaced by a trained ML classifier.
    """

    INCIDENT_WEIGHTS = {
        IncidentType.FIRE: 1.4,
        IncidentType.BUILDING_COLLAPSE: 1.8,
        IncidentType.GAS_LEAK: 1.5,
        IncidentType.ROAD_ACCIDENT: 1.1,
        IncidentType.FLOOD: 1.0,
        IncidentType.MEDICAL_EMERGENCY: 1.2,
        IncidentType.EARTHQUAKE_DAMAGE: 1.7,
        IncidentType.LANDSLIDE: 1.5,
        IncidentType.CIVIL_DISTURBANCE: 1.0,
        IncidentType.POWER_FAILURE: 0.4,
        IncidentType.UNKNOWN: 0.0,
    }

    SEVERITY_WEIGHTS = {
        SeverityLevel.CRITICAL: 2.2,
        SeverityLevel.HIGH: 1.5,
        SeverityLevel.MEDIUM: 0.8,
        SeverityLevel.LOW: 0.3,
        SeverityLevel.UNKNOWN: 0.0,
    }

    HAZARD_WEIGHTS = {
        "fire": 0.7,
        "smoke": 0.4,
        "gas": 0.9,
        "debris": 0.6,
        "flood_water": 0.5,
        "injury_risk": 0.8,
        "traffic_block": 0.4,
    }

    def score(self, extraction: ExtractionResultDTO) -> float:
        raw_score = -1.2

        raw_score += self.INCIDENT_WEIGHTS.get(extraction.incident.type, 0.0)
        raw_score += self.SEVERITY_WEIGHTS.get(extraction.incident.severity, 0.0)

        for hazard in extraction.hazards:
            raw_score += self.HAZARD_WEIGHTS.get(hazard, 0.0)

        if extraction.casualties:
            if extraction.casualties.people_trapped:
                raw_score += 1.2

            if extraction.casualties.injured_count:
                raw_score += min(extraction.casualties.injured_count * 0.25, 1.0)

            if extraction.casualties.dead_count:
                raw_score += min(extraction.casualties.dead_count * 0.35, 1.2)

        raw_score += extraction.extraction_metadata.overall_confidence * 0.5

        return self._sigmoid(raw_score)

    def _sigmoid(self, value: float) -> float:
        probability = 1 / (1 + math.exp(-value))
        return round(probability, 4)