from enum import Enum


class CallerRole(str, Enum):
    """
    Standardized caller identities describing the caller's relation
    to the incident. Useful for confidence scoring and follow-up logic.
    """

    VICTIM = "victim"
    BYSTANDER = "bystander"
    RESIDENT = "resident"
    DRIVER = "driver"
    STAFF = "staff"
    SECURITY = "security"
    FIRST_RESPONDER = "first_responder"
    DISPATCHER_RELAY = "dispatcher_relay"
    AUTOMATED_ALERT = "automated_alert"
    UNKNOWN = "unknown"