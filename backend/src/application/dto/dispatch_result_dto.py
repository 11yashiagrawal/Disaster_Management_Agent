from typing import List

from pydantic import BaseModel


class AssignedResourceDTO(BaseModel):
    id: str
    type: str
    current_location: str
    eta_minutes: int


class DispatchResultDTO(BaseModel):
    dispatch_status: str
    assigned_resources: List[AssignedResourceDTO]
    unfulfilled_resources: List[str]
    dispatch_summary: str