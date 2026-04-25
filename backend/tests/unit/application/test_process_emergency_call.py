from application.use_cases.process_emergency_call import ProcessEmergencyCallUseCase
from domain.enums.incident_type import IncidentType
from domain.services.dispatch_service import DispatchService
from domain.services.triage_service import TriageService
from infrastructure.nlp.hybrid_parser import HybridParser
from infrastructure.nlp.spacy_transcript_enricher import SpacyTranscriptEnricher


def build_use_case() -> ProcessEmergencyCallUseCase:
    return ProcessEmergencyCallUseCase(
        parser=HybridParser(enricher=SpacyTranscriptEnricher()),
        triage_service=TriageService(),
        dispatch_service=DispatchService(),
    )


def test_process_emergency_call_extracts_triages_and_dispatches_fire_case() -> None:
    use_case = build_use_case()

    result = use_case.execute(
        call_id="CALL_PROCESS_001",
        timestamp="2026-04-25T16:30:00Z",
        transcript="aaaa please hurry smok and fire near green park metro people trapped inside",
    )

    assert result.extraction.call_id == "CALL_PROCESS_001"
    assert result.extraction.incident.type == IncidentType.FIRE
    assert result.triage.priority_level in {"P1", "P2"}
    assert result.triage.priority_score >= 60
    assert result.dispatch.dispatch_status in {
        "fully_assigned",
        "partially_assigned",
        "not_assigned",
    }


def test_process_emergency_call_returns_complete_result_shape() -> None:
    use_case = build_use_case()

    result = use_case.execute(
        call_id="CALL_PROCESS_002",
        timestamp="2026-04-25T16:35:00Z",
        transcript="bus and car hit badly near city hospital two injured road blocked",
    )

    assert result.extraction is not None
    assert result.triage is not None
    assert result.dispatch is not None

    assert result.extraction.resources_needed is not None
    assert result.triage.recommended_actions
    assert result.dispatch.assigned_resources is not None

    assert 0 <= result.triage.priority_score <= 100