from typing import List

from pydantic import BaseModel, Field


class TriageResultDTO(BaseModel):
    priority_level: str
    priority_score: int = Field(ge=0, le=100)
    risk_probability: float = Field(ge=0.0, le=1.0)
    escalation_required: bool
    triage_reason: str
    recommended_actions: List[str]