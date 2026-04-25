from typing import List, Optional

from pydantic import BaseModel, Field

from domain.enums.incident_type import IncidentType
from domain.enums.severity_level import SeverityLevel
from domain.enums.caller_role import CallerRole


class CoordinatesDTO(BaseModel):
    lat: Optional[float] = None
    lng: Optional[float] = None


class IncidentDTO(BaseModel):
    type: IncidentType
    subtype: Optional[str] = None
    severity: SeverityLevel
    description: str


class LocationDTO(BaseModel):
    raw_text: str
    landmark: Optional[str] = None
    address: Optional[str] = None
    area: Optional[str] = None
    city: Optional[str] = None
    coordinates: CoordinatesDTO = Field(default_factory=CoordinatesDTO)
    confidence: float = Field(ge=0.0, le=1.0)


class CasualtiesDTO(BaseModel):
    injured_count: Optional[int] = Field(default=None, ge=0)
    dead_count: Optional[int] = Field(default=None, ge=0)
    people_trapped: Optional[bool] = None


class ResourcesNeededDTO(BaseModel):
    ambulance: int = Field(default=0, ge=0)
    fire_truck: int = Field(default=0, ge=0)
    police_unit: int = Field(default=0, ge=0)
    rescue_team: int = Field(default=0, ge=0)


class CallerContextDTO(BaseModel):
    caller_role: CallerRole = CallerRole.UNKNOWN
    emotional_state: Optional[str] = None
    callback_number: Optional[str] = None


class ExtractionMetadataDTO(BaseModel):
    missing_fields: List[str] = Field(default_factory=list)
    contradictions_detected: List[str] = Field(default_factory=list)
    overall_confidence: float = Field(ge=0.0, le=1.0)


class DeduplicationDTO(BaseModel):
    event_signature: str
    possible_duplicate_of: Optional[str] = None


class ExtractionResultDTO(BaseModel):
    call_id: str
    timestamp: str
    raw_transcript: str

    incident: IncidentDTO
    location: LocationDTO
    casualties: Optional[CasualtiesDTO] = None

    hazards: List[str] = Field(default_factory=list)

    resources_needed: ResourcesNeededDTO
    caller_context: Optional[CallerContextDTO] = None

    extraction_metadata: ExtractionMetadataDTO
    deduplication: Optional[DeduplicationDTO] = None