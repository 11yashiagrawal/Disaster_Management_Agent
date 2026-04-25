from typing import List

from pydantic import BaseModel


class CoordinateDTO(BaseModel):
    lat: float
    lng: float


class AssignedResourceDTO(BaseModel):
    id: str
    type: str
    current_location: str
    current_coordinates: CoordinateDTO
    target_coordinates: CoordinateDTO
    route: List[CoordinateDTO]
    eta_minutes: int


class DispatchResultDTO(BaseModel):
    dispatch_status: str
    incident_coordinates: CoordinateDTO
    assigned_resources: List[AssignedResourceDTO]
    unfulfilled_resources: List[str]
    dispatch_summary: str