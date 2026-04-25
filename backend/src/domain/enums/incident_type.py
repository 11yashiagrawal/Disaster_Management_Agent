from enum import Enum


class IncidentType(str, Enum):
    """
    Standardized top-level incident categories used across
    NLP extraction, triage, dispatch, analytics, and reporting.
    """

    FIRE = "fire"
    FLOOD = "flood"
    ROAD_ACCIDENT = "road_accident"
    MEDICAL_EMERGENCY = "medical_emergency"
    BUILDING_COLLAPSE = "building_collapse"
    GAS_LEAK = "gas_leak"
    EARTHQUAKE_DAMAGE = "earthquake_damage"
    LANDSLIDE = "landslide"
    CIVIL_DISTURBANCE = "civil_disturbance"
    POWER_FAILURE = "power_failure"
    UNKNOWN = "unknown"