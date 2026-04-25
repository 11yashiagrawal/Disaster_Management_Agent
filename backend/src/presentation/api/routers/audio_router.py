from datetime import datetime, timezone

from fastapi import APIRouter, File, UploadFile

from application.use_cases.process_emergency_call import ProcessEmergencyCallUseCase
from domain.services.dispatch_service import DispatchService
from domain.services.triage_service import TriageService
from infrastructure.nlp.hybrid_parser import HybridParser
from infrastructure.nlp.spacy_transcript_enricher import SpacyTranscriptEnricher
from infrastructure.speech.whisper_transcriber import WhisperTranscriber


router = APIRouter(prefix="/api/audio", tags=["Audio Intake"])

transcriber = WhisperTranscriber(model_name="base")


@router.post("/process")
async def process_audio_call(file: UploadFile = File(...)):
    audio_bytes = await file.read()

    transcript = transcriber.transcribe(
        audio_bytes=audio_bytes,
        filename=file.filename or "audio.wav",
    )

    use_case = ProcessEmergencyCallUseCase(
        parser=HybridParser(enricher=SpacyTranscriptEnricher()),
        triage_service=TriageService(),
        dispatch_service=DispatchService(),
    )

    result = use_case.execute(
        call_id=f"CALL_AUDIO_{int(datetime.now().timestamp())}",
        timestamp=datetime.now(timezone.utc).isoformat(),
        transcript=transcript,
    )

    return {
        "transcript": transcript,
        "result": result.model_dump(),
    }