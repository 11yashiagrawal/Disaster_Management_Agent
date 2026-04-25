from typing import List, Optional

from pydantic import BaseModel, Field

from domain.enums.incident_type import IncidentType
from domain.enums.severity_level import SeverityLevel


class TranscriptSignalDTO(BaseModel):
    label: str
    value: str
    confidence: float = Field(ge=0.0, le=1.0)


class EnrichedTranscriptDTO(BaseModel):
    raw_transcript: str
    normalized_transcript: str

    inferred_incident_type: Optional[IncidentType] = None
    inferred_severity: Optional[SeverityLevel] = None

    possible_locations: List[str] = Field(default_factory=list)
    possible_hazards: List[str] = Field(default_factory=list)

    possible_casualty_clues: List[str] = Field(default_factory=list)
    urgency_clues: List[str] = Field(default_factory=list)

    extracted_signals: List[TranscriptSignalDTO] = Field(default_factory=list)

    enrichment_confidence: float = Field(ge=0.0, le=1.0)