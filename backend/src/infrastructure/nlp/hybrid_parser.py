from typing import Optional

from application.dto.enriched_transcript_dto import EnrichedTranscriptDTO
from application.dto.extraction_result_dto import ExtractionResultDTO
from application.interfaces.nlp_parser import NLPParser
from application.interfaces.transcript_enricher import TranscriptEnricher
from core.utils.transcript_correction import repair_transcript
from infrastructure.nlp.regex_parser import RegexParser


class HybridParser(NLPParser):
    """
    Hybrid parser:
    1. Repairs noisy transcript
    2. Enriches it using NLP
    3. Builds an augmented transcript using enrichment hints
    4. Sends the augmented transcript through the deterministic regex parser
    """

    def __init__(
        self,
        *,
        enricher: TranscriptEnricher,
        fallback_parser: Optional[RegexParser] = None,
    ) -> None:
        self.enricher = enricher
        self.fallback_parser = fallback_parser or RegexParser()

    def parse(
        self,
        *,
        call_id: str,
        timestamp: str,
        transcript: str,
    ) -> ExtractionResultDTO:
        repaired_transcript = repair_transcript(transcript)

        enriched = self.enricher.enrich(transcript=repaired_transcript)
        augmented_transcript = self._build_augmented_transcript(
            raw_transcript=repaired_transcript,
            enriched=enriched,
        )

        result = self.fallback_parser.parse(
            call_id=call_id,
            timestamp=timestamp,
            transcript=augmented_transcript,
        )

        result.raw_transcript = transcript
        return result

    def _build_augmented_transcript(
        self,
        *,
        raw_transcript: str,
        enriched: EnrichedTranscriptDTO,
    ) -> str:
        additions: list[str] = []

        if enriched.inferred_incident_type is not None:
            additions.append(f"incident {enriched.inferred_incident_type.value}")

        if enriched.inferred_severity is not None:
            additions.append(f"severity {enriched.inferred_severity.value}")

        for location in enriched.possible_locations[:2]:
            additions.append(location)

        for hazard in enriched.possible_hazards[:3]:
            additions.append(hazard)

        for clue in enriched.possible_casualty_clues[:3]:
            additions.append(clue)

        if not additions:
            return raw_transcript

        return raw_transcript.strip() + ". " + ". ".join(additions)