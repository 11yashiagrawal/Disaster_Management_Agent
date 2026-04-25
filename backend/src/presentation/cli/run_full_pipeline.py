from datetime import datetime, timezone

from application.use_cases.process_emergency_call import ProcessEmergencyCallUseCase
from domain.services.dispatch_service import DispatchService
from domain.services.triage_service import TriageService
from infrastructure.nlp.hybrid_parser import HybridParser
from infrastructure.nlp.spacy_transcript_enricher import SpacyTranscriptEnricher


def main() -> None:
    transcript = (
        "aaaa please hurry.... smok and fire near green park metro... "
        "people trap inside maybe... send help fast"
    )

    use_case = ProcessEmergencyCallUseCase(
        parser=HybridParser(enricher=SpacyTranscriptEnricher()),
        triage_service=TriageService(),
        dispatch_service=DispatchService(),
    )

    result = use_case.execute(
        call_id="CALL_DEMO_001",
        timestamp=datetime.now(timezone.utc).isoformat(),
        transcript=transcript,
    )

    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()