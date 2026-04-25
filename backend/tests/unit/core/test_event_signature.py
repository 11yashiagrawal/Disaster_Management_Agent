from core.utils.event_signature import build_event_signature
from domain.enums.incident_type import IncidentType


def test_build_event_signature_with_normal_location() -> None:
    result = build_event_signature(
        incident_type=IncidentType.FIRE,
        location_raw="near green park metro station",
    )

    assert result == "fire|near_green_park_metro_station"


def test_build_event_signature_uses_unknown_when_location_missing() -> None:
    result = build_event_signature(
        incident_type=IncidentType.FLOOD,
        location_raw=None,
    )

    assert result == "flood|unknown"


def test_build_event_signature_removes_punctuation() -> None:
    result = build_event_signature(
        incident_type=IncidentType.ROAD_ACCIDENT,
        location_raw="near City Hospital, Block-A!",
    )

    assert result == "road_accident|near_city_hospital_blocka"


def test_build_event_signature_lowercases_and_normalizes_spaces() -> None:
    result = build_event_signature(
        incident_type=IncidentType.GAS_LEAK,
        location_raw="  Near   Sector 17 Market  ",
    )

    assert result == "gas_leak|near___sector_17_market"