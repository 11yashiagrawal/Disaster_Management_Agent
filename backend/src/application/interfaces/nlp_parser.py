from abc import ABC, abstractmethod

from application.dto.extraction_result_dto import ExtractionResultDTO


class NLPParser(ABC):
    """
    Contract for all transcript parsers.

    Any implementation (regex, spaCy, LLM, hybrid) must accept a raw
    emergency transcript and return a validated ExtractionResultDTO.
    """

    @abstractmethod
    def parse(
        self,
        *,
        call_id: str,
        timestamp: str,
        transcript: str,
    ) -> ExtractionResultDTO:
        """
        Convert raw emergency call transcript into structured output.

        Args:
            call_id: Unique incoming call identifier.
            timestamp: Call received timestamp.
            transcript: Raw spoken text converted to transcript.

        Returns:
            ExtractionResultDTO
        """
        raise NotImplementedError