from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from application.use_cases.process_emergency_call import ProcessEmergencyCallUseCase
from domain.services.dispatch_service import DispatchService
from domain.services.triage_service import TriageService
from infrastructure.nlp.hybrid_parser import HybridParser
from infrastructure.nlp.spacy_transcript_enricher import SpacyTranscriptEnricher


router = APIRouter(prefix="/api/emergency-calls", tags=["Emergency Calls"])


class EmergencyCallRequest(BaseModel):
    call_id: Optional[str] = None
    timestamp: Optional[str] = None
    transcript: str = Field(min_length=1)


@router.post("/process")
def process_emergency_call(payload: EmergencyCallRequest):
    use_case = ProcessEmergencyCallUseCase(
        parser=HybridParser(enricher=SpacyTranscriptEnricher()),
        triage_service=TriageService(),
        dispatch_service=DispatchService(),
    )

    result = use_case.execute(
        call_id=payload.call_id or "CALL_API_001",
        timestamp=payload.timestamp or datetime.now(timezone.utc).isoformat(),
        transcript=payload.transcript,
    )

    return result.model_dump()