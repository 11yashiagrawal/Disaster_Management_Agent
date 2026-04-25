from application.dto.extraction_result_dto import (
    CasualtiesDTO,
    ExtractionMetadataDTO,
    ExtractionResultDTO,
    IncidentDTO,
    LocationDTO,
    ResourcesNeededDTO,
)
from domain.enums.incident_type import IncidentType
from domain.enums.severity_level import SeverityLevel
from domain.services.triage_service import TriageService


def build_extraction(
    *,
    incident_type: IncidentType,
    severity: SeverityLevel,
    hazards: list[str],
    people_trapped: bool | None = None,
    injured_count: int | None = None,
    confidence: float = 0.85,
) -> ExtractionResultDTO:
    return ExtractionResultDTO(
        call_id="CALL_TEST",
        timestamp="2026-04-25T16:00:00Z",
        raw_transcript="test transcript",
        incident=IncidentDTO(
            type=incident_type,
            subtype=None,
            severity=severity,
            description="test incident",
        ),
        location=LocationDTO(
            raw_text="near city hospital",
            confidence=0.8,
        ),
        casualties=CasualtiesDTO(
            injured_count=injured_count,
            dead_count=None,
            people_trapped=people_trapped,
        ),
        hazards=hazards,
        resources_needed=ResourcesNeededDTO(),
        extraction_metadata=ExtractionMetadataDTO(
            missing_fields=[],
            contradictions_detected=[],
            overall_confidence=confidence,
        ),
    )


def test_triage_assigns_p1_for_fire_with_trapped_people() -> None:
    extraction = build_extraction(
        incident_type=IncidentType.FIRE,
        severity=SeverityLevel.CRITICAL,
        hazards=["fire", "smoke"],
        people_trapped=True,
    )

    result = TriageService().triage(extraction)

    assert result.priority_level == "P1"
    assert result.priority_score >= 80
    assert result.escalation_required is True


def test_triage_assigns_high_priority_for_gas_hazard() -> None:
    extraction = build_extraction(
        incident_type=IncidentType.GAS_LEAK,
        severity=SeverityLevel.HIGH,
        hazards=["gas"],
    )

    result = TriageService().triage(extraction)

    assert result.priority_level in {"P1", "P2"}
    assert result.priority_score >= 60


def test_triage_assigns_lower_priority_for_unknown_uncertain_case() -> None:
    extraction = build_extraction(
        incident_type=IncidentType.UNKNOWN,
        severity=SeverityLevel.UNKNOWN,
        hazards=[],
        confidence=0.3,
    )

    result = TriageService().triage(extraction)

    assert result.priority_level in {"P3", "P4"}
    assert result.escalation_required is False