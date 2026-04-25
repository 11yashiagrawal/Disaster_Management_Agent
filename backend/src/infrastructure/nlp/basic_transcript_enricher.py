import re
from typing import List, Optional

from application.dto.enriched_transcript_dto import (
    EnrichedTranscriptDTO,
    TranscriptSignalDTO,
)
from application.interfaces.transcript_enricher import TranscriptEnricher
from core.utils.text_cleaning import normalize_transcript
from domain.enums.incident_type import IncidentType
from domain.enums.severity_level import SeverityLevel


class BasicTranscriptEnricher(TranscriptEnricher):
    """
    Lightweight NLP-style enricher using soft pattern detection.

    This is phase 2 baseline logic:
    - detects weak semantic cues
    - gathers possible locations
    - gathers urgency / casualty clues
    - infers likely incident + severity even for messy transcripts
    """

    def enrich(self, *, transcript: str) -> EnrichedTranscriptDTO:
        normalized = normalize_transcript(transcript)

        incident_type = self._infer_incident_type(normalized)
        severity = self._infer_severity(normalized)
        locations = self._extract_possible_locations(normalized)
        hazards = self._extract_possible_hazards(normalized)
        casualty_clues = self._extract_casualty_clues(normalized)
        urgency_clues = self._extract_urgency_clues(normalized)
        signals = self._build_signals(
            incident_type=incident_type,
            severity=severity,
            locations=locations,
            hazards=hazards,
            casualty_clues=casualty_clues,
            urgency_clues=urgency_clues,
        )
        confidence = self._compute_enrichment_confidence(
            incident_type=incident_type,
            severity=severity,
            locations=locations,
            hazards=hazards,
            urgency_clues=urgency_clues,
        )

        return EnrichedTranscriptDTO(
            raw_transcript=transcript,
            normalized_transcript=normalized,
            inferred_incident_type=incident_type,
            inferred_severity=severity,
            possible_locations=locations,
            possible_hazards=hazards,
            possible_casualty_clues=casualty_clues,
            urgency_clues=urgency_clues,
            extracted_signals=signals,
            enrichment_confidence=confidence,
        )

    def _infer_incident_type(self, text: str) -> Optional[IncidentType]:
        if re.search(r"\b(fire|smoke|burning|flames|blaze)\b", text):
            return IncidentType.FIRE
        if re.search(r"\b(flood|waterlogging|submerged|water rising|overflowing)\b", text):
            return IncidentType.FLOOD
        if re.search(r"\b(accident|collision|crash|hit badly|vehicle hit|pileup)\b", text):
            return IncidentType.ROAD_ACCIDENT
        if re.search(r"\b(collapse|collapsed|rubble|building fell)\b", text):
            return IncidentType.BUILDING_COLLAPSE
        if re.search(r"\b(gas smell|gas leak|smell of gas|leakage)\b", text):
            return IncidentType.GAS_LEAK
        if re.search(r"\b(unconscious|bleeding|heart attack|not breathing|fainted)\b", text):
            return IncidentType.MEDICAL_EMERGENCY
        return None

    def _infer_severity(self, text: str) -> Optional[SeverityLevel]:
        if re.search(
            r"\b(trapped|not breathing|unconscious|bleeding heavily|explosion|many injured|multiple injured|children inside|cannot breathe)\b",
            text,
        ):
            return SeverityLevel.CRITICAL

        if re.search(
            r"\b(serious|help fast|urgent|immediately|road blocked|smoke everywhere|flames)\b",
            text,
        ):
            return SeverityLevel.HIGH

        if re.search(r"\b(injured|blocked|leak|smoke)\b", text):
            return SeverityLevel.MEDIUM

        return None

    def _extract_possible_locations(self, text: str) -> List[str]:
        patterns = [
            r"\bnear ([a-z0-9\s\-]+)",
            r"\bat ([a-z0-9\s\-]+)",
            r"\bon ([a-z0-9\s\-]+)",
            r"\bin ([a-z0-9\s\-]+)",
            r"\bbeside ([a-z0-9\s\-]+)",
            r"\bbehind ([a-z0-9\s\-]+)",
            r"\bopposite ([a-z0-9\s\-]+)",
            r"\bby the ([a-z0-9\s\-]+)",
        ]

        found: List[str] = []
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                phrase = match.group(0).strip(" ,.")
                phrase = re.split(
                    r"\b(please|send|help|quickly|fast|immediately|urgent|hurry)\b",
                    phrase,
                )[0].strip(" ,.")
                if phrase and phrase not in found:
                    found.append(phrase)

        return found

    def _extract_possible_hazards(self, text: str) -> List[str]:
        hazard_map = {
            "fire": r"\b(fire|burning|blaze|flames)\b",
            "smoke": r"\bsmoke\b",
            "flood_water": r"\b(flood|waterlogging|submerged|overflowing)\b",
            "gas": r"\b(gas smell|gas leak|smell of gas)\b",
            "debris": r"\b(rubble|debris)\b",
            "injury_risk": r"\b(injured|bleeding|unconscious|not breathing)\b",
            "traffic_block": r"\b(road blocked|blocked road|traffic jam|blocked)\b",
        }

        hazards: List[str] = []
        for label, pattern in hazard_map.items():
            if re.search(pattern, text):
                hazards.append(label)

        return hazards

    def _extract_casualty_clues(self, text: str) -> List[str]:
        patterns = [
            r"\binjured\b",
            r"\bkilled\b",
            r"\bdead\b",
            r"\bunconscious\b",
            r"\bnot breathing\b",
            r"\btrapped\b",
            r"\bbleeding\b",
            r"\bfainted\b",
        ]

        clues: List[str] = []
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                clues.append(match.group(0))

        return clues

    def _extract_urgency_clues(self, text: str) -> List[str]:
        patterns = [
            r"\burgent\b",
            r"\bimmediately\b",
            r"\bhelp fast\b",
            r"\bquickly\b",
            r"\bhurry\b",
            r"\bsend help\b",
            r"\bplease\b",
        ]

        clues: List[str] = []
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                clues.append(match.group(0))

        return clues

    def _build_signals(
        self,
        *,
        incident_type: Optional[IncidentType],
        severity: Optional[SeverityLevel],
        locations: List[str],
        hazards: List[str],
        casualty_clues: List[str],
        urgency_clues: List[str],
    ) -> List[TranscriptSignalDTO]:
        signals: List[TranscriptSignalDTO] = []

        if incident_type is not None:
            signals.append(
                TranscriptSignalDTO(
                    label="incident_type",
                    value=incident_type.value,
                    confidence=0.85,
                )
            )

        if severity is not None:
            signals.append(
                TranscriptSignalDTO(
                    label="severity",
                    value=severity.value,
                    confidence=0.80,
                )
            )

        for location in locations:
            signals.append(
                TranscriptSignalDTO(
                    label="location_hint",
                    value=location,
                    confidence=0.75,
                )
            )

        for hazard in hazards:
            signals.append(
                TranscriptSignalDTO(
                    label="hazard",
                    value=hazard,
                    confidence=0.78,
                )
            )

        for clue in casualty_clues:
            signals.append(
                TranscriptSignalDTO(
                    label="casualty_clue",
                    value=clue,
                    confidence=0.72,
                )
            )

        for clue in urgency_clues:
            signals.append(
                TranscriptSignalDTO(
                    label="urgency_clue",
                    value=clue,
                    confidence=0.68,
                )
            )

        return signals

    def _compute_enrichment_confidence(
        self,
        *,
        incident_type: Optional[IncidentType],
        severity: Optional[SeverityLevel],
        locations: List[str],
        hazards: List[str],
        urgency_clues: List[str],
    ) -> float:
        score = 0.20

        if incident_type is not None:
            score += 0.25
        if severity is not None:
            score += 0.20
        if locations:
            score += 0.15
        if hazards:
            score += 0.10
        if urgency_clues:
            score += 0.10

        return round(min(score, 0.95), 2)