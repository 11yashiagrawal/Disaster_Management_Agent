from pydantic import BaseModel


class ResourceUnit(BaseModel):
    id: str
    type: str
    status: str
    current_location: str