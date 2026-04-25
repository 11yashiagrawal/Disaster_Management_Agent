from application.dto.extraction_result_dto import ExtractionResultDTO
from application.interfaces.nlp_parser import NLPParser


class ParseEmergencyCallUseCase:
    """
    Application use case for converting a raw emergency call transcript
    into structured extraction output.
    """

    def __init__(self, parser: NLPParser) -> None:
        self.parser = parser

    def execute(
        self,
        *,
        call_id: str,
        timestamp: str,
        transcript: str,
    ) -> ExtractionResultDTO:
        transcript = transcript.strip()

        return self.parser.parse(
            call_id=call_id,
            timestamp=timestamp,
            transcript=transcript,
        )