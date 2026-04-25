from application.dto.extraction_result_dto import (
    ExtractionMetadataDTO,
    ExtractionResultDTO,
    IncidentDTO,
    LocationDTO,
    ResourcesNeededDTO,
)
from domain.enums.incident_type import IncidentType
from domain.enums.severity_level import SeverityLevel
from domain.services.dispatch_service import DispatchService


def build_extraction_with_resources() -> ExtractionResultDTO:
    return ExtractionResultDTO(
        call_id="CALL_DISPATCH",
        timestamp="2026-04-25T17:00:00Z",
        raw_transcript="fire near metro station",
        incident=IncidentDTO(
            type=IncidentType.FIRE,
            subtype="building_fire",
            severity=SeverityLevel.CRITICAL,
            description="Fire incident",
        ),
        location=LocationDTO(
            raw_text="near metro station",
            confidence=0.85,
        ),
        resources_needed=ResourcesNeededDTO(
            ambulance=1,
            fire_truck=2,
            police_unit=1,
            rescue_team=1,
        ),
        hazards=["fire", "smoke"],
        extraction_metadata=ExtractionMetadataDTO(
            missing_fields=[],
            contradictions_detected=[],
            overall_confidence=0.95,
        ),
    )


def test_dispatch_assigns_available_resources() -> None:
    extraction = build_extraction_with_resources()

    result = DispatchService().dispatch(extraction)

    assert result.dispatch_status == "fully_assigned"
    assert len(result.assigned_resources) == 5
    assert result.unfulfilled_resources == []


def test_dispatch_result_contains_eta_values() -> None:
    extraction = build_extraction_with_resources()

    result = DispatchService().dispatch(extraction)

    assert all(unit.eta_minutes > 0 for unit in result.assigned_resources)


def test_dispatch_handles_resource_shortage() -> None:
    extraction = build_extraction_with_resources()
    extraction.resources_needed.fire_truck = 5

    result = DispatchService().dispatch(extraction)

    assert result.dispatch_status == "partially_assigned"
    assert "fire_truck" in result.unfulfilled_resources