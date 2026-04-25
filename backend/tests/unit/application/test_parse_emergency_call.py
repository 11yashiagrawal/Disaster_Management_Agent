from application.dto.extraction_result_dto import (
    ExtractionMetadataDTO,
    ExtractionResultDTO,
    IncidentDTO,
    LocationDTO,
    ResourcesNeededDTO,
)
from application.interfaces.nlp_parser import NLPParser
from application.use_cases.parse_emergency_call import ParseEmergencyCallUseCase
from domain.enums.incident_type import IncidentType
from domain.enums.severity_level import SeverityLevel


class FakeParser(NLPParser):
    def parse(
        self,
        *,
        call_id: str,
        timestamp: str,
        transcript: str,
    ) -> ExtractionResultDTO:
        return ExtractionResultDTO(
            call_id=call_id,
            timestamp=timestamp,
            raw_transcript=transcript,
            incident=IncidentDTO(
                type=IncidentType.FIRE,
                subtype="building_fire",
                severity=SeverityLevel.HIGH,
                description="Mock parsed fire incident",
            ),
            location=LocationDTO(
                raw_text="near city hospital",
                confidence=0.8,
            ),
            resources_needed=ResourcesNeededDTO(
                ambulance=1,
                fire_truck=2,
                police_unit=1,
                rescue_team=0,
            ),
            extraction_metadata=ExtractionMetadataDTO(
                missing_fields=[],
                contradictions_detected=[],
                overall_confidence=0.8,
            ),
        )


def test_parse_emergency_call_use_case_returns_parser_output() -> None:
    parser = FakeParser()
    use_case = ParseEmergencyCallUseCase(parser=parser)

    result = use_case.execute(
        call_id="CALL_TEST_100",
        timestamp="2026-04-23T18:00:00Z",
        transcript=" There is a fire near City Hospital. Send help fast. ",
    )

    assert result.call_id == "CALL_TEST_100"
    assert result.timestamp == "2026-04-23T18:00:00Z"
    assert result.raw_transcript == "There is a fire near City Hospital. Send help fast."
    assert result.incident.type == IncidentType.FIRE
    assert result.incident.subtype == "building_fire"
    assert result.incident.severity == SeverityLevel.HIGH
    assert result.location.raw_text == "near city hospital"
    assert result.resources_needed.fire_truck == 2
    assert result.extraction_metadata.overall_confidence == 0.8