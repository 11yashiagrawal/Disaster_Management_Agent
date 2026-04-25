from enum import Enum


class SeverityLevel(str, Enum):
    """
    Standardized severity levels used for triage prioritization.

    LOW:
        Minor incident, no immediate threat to life, limited response needed.

    MEDIUM:
        Manageable incident, moderate urgency, response required soon.

    HIGH:
        Serious incident, risk of escalation, urgent coordinated response needed.

    CRITICAL:
        Immediate danger to life, mass risk, highest dispatch priority.

    UNKNOWN:
        Used when severity cannot yet be determined confidently.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    UNKNOWN = "unknown"