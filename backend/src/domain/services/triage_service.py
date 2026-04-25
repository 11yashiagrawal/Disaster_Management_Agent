from typing import List

from application.dto.extraction_result_dto import ExtractionResultDTO
from application.dto.triage_result_dto import TriageResultDTO
from domain.services.risk_scoring_service import RiskScoringService
from domain.services.safety_override_service import SafetyOverrideService


class TriageService:
    """
    Combines model-based risk scoring with deterministic safety overrides.
    """

    def __init__(
        self,
        risk_scoring_service: RiskScoringService | None = None,
        safety_override_service: SafetyOverrideService | None = None,
    ) -> None:
        self.risk_scoring_service = risk_scoring_service or RiskScoringService()
        self.safety_override_service = safety_override_service or SafetyOverrideService()

    def triage(self, extraction: ExtractionResultDTO) -> TriageResultDTO:
        risk_probability = self.risk_scoring_service.score(extraction)
        override_reasons = self.safety_override_service.get_override_reasons(extraction)

        priority_score = int(round(risk_probability * 100))

        if override_reasons:
            priority_score = max(priority_score, 80)

        priority_level = self._priority_from_score(priority_score)

        reasons = self._build_reasons(
            extraction=extraction,
            risk_probability=risk_probability,
            override_reasons=override_reasons,
        )

        actions = self._build_actions(priority_level)

        return TriageResultDTO(
            priority_level=priority_level,
            priority_score=priority_score,
            risk_probability=risk_probability,
            escalation_required=priority_level == "P1",
            triage_reason="; ".join(reasons),
            recommended_actions=actions,
        )

    def _priority_from_score(self, score: int) -> str:
        if score >= 80:
            return "P1"
        if score >= 60:
            return "P2"
        if score >= 35:
            return "P3"
        return "P4"

    def _build_reasons(
        self,
        *,
        extraction: ExtractionResultDTO,
        risk_probability: float,
        override_reasons: List[str],
    ) -> List[str]:
        reasons = [
            f"Model risk probability: {risk_probability}",
            f"Incident type: {extraction.incident.type.value}",
            f"Severity: {extraction.incident.severity.value}",
        ]

        if extraction.hazards:
            reasons.append(f"Hazards detected: {', '.join(extraction.hazards)}")

        if override_reasons:
            reasons.append("Safety overrides applied: " + ", ".join(override_reasons))

        return reasons

    def _build_actions(self, priority_level: str) -> List[str]:
        if priority_level == "P1":
            return [
                "Dispatch nearest available emergency units immediately",
                "Notify command center supervisor",
                "Keep incident in active escalation queue",
            ]

        if priority_level == "P2":
            return [
                "Prioritize dispatch assignment",
                "Monitor for additional duplicate reports",
            ]

        if priority_level == "P3":
            return [
                "Assign standard response when units are available",
                "Request clarification if location is weak",
            ]

        return [
            "Log incident for review",
            "Request more information before dispatch escalation",
        ]