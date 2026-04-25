import re
from typing import Optional

from domain.enums.incident_type import IncidentType


def build_event_signature(
    *,
    incident_type: IncidentType,
    location_raw: Optional[str],
) -> str:
    """
    Build a normalized event signature used for rough deduplication.

    Example:
        fire + "near green park metro station"
        -> "fire|near_green_park_metro_station"
    """
    normalized_location = (location_raw or "unknown").strip().lower()
    normalized_location = normalized_location.replace(" ", "_")
    normalized_location = re.sub(r"[^a-z0-9_]", "", normalized_location)

    return f"{incident_type.value}|{normalized_location}"