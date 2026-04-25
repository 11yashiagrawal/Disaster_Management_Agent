from pydantic import BaseModel

from application.dto.dispatch_result_dto import DispatchResultDTO
from application.dto.extraction_result_dto import ExtractionResultDTO
from application.dto.triage_result_dto import TriageResultDTO
from application.interfaces.nlp_parser import NLPParser
from domain.services.dispatch_service import DispatchService
from domain.services.triage_service import TriageService


class EmergencyCallProcessingResultDTO(BaseModel):
    extraction: ExtractionResultDTO
    triage: TriageResultDTO
    dispatch: DispatchResultDTO


class ProcessEmergencyCallUseCase:
    """
    End-to-end application use case.

    Flow:
    raw transcript -> parser extraction -> model-based triage -> resource dispatch
    """

    def __init__(
        self,
        *,
        parser: NLPParser,
        triage_service: TriageService,
        dispatch_service: DispatchService,
    ) -> None:
        self.parser = parser
        self.triage_service = triage_service
        self.dispatch_service = dispatch_service

    def execute(
        self,
        *,
        call_id: str,
        timestamp: str,
        transcript: str,
    ) -> EmergencyCallProcessingResultDTO:
        extraction = self.parser.parse(
            call_id=call_id,
            timestamp=timestamp,
            transcript=transcript,
        )

        triage = self.triage_service.triage(extraction)
        dispatch = self.dispatch_service.dispatch(extraction)

        return EmergencyCallProcessingResultDTO(
            extraction=extraction,
            triage=triage,
            dispatch=dispatch,
        )