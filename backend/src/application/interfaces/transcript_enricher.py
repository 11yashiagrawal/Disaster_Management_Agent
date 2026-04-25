from abc import ABC, abstractmethod

from application.dto.enriched_transcript_dto import EnrichedTranscriptDTO


class TranscriptEnricher(ABC):
    """
    Contract for transcript enrichment components.

    The enricher interprets noisy emergency transcripts and returns
    intermediate signals that can later improve structured extraction.
    """

    @abstractmethod
    def enrich(self, *, transcript: str) -> EnrichedTranscriptDTO:
        raise NotImplementedError