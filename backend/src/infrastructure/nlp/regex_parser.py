import re
from typing import List, Optional

from application.dto.extraction_result_dto import (
    CallerContextDTO,
    CasualtiesDTO,
    DeduplicationDTO,
    ExtractionMetadataDTO,
    ExtractionResultDTO,
    IncidentDTO,
    LocationDTO,
    ResourcesNeededDTO,
)
from application.interfaces.nlp_parser import NLPParser
from core.utils.confidence import compute_extraction_confidence
from core.utils.event_signature import build_event_signature
from core.utils.number_parsing import extract_leading_number_token
from core.utils.text_cleaning import normalize_transcript
from domain.enums.caller_role import CallerRole
from domain.enums.incident_type import IncidentType
from domain.enums.severity_level import SeverityLevel


class RegexParser(NLPParser):
    """
    Rule-based baseline parser for emergency call transcripts.

    This parser is intentionally simple and deterministic.
    It is useful for:
    - MVP extraction
    - testing contracts
    - validating end-to-end workflow
    - serving as a fallback parser later
    """

    LOCATION_STOP_WORDS_PATTERN = (
        r"\b(please|send|help|quickly|fast|immediately|urgent|hurry|come fast)\b"
    )

    def parse(
        self,
        *,
        call_id: str,
        timestamp: str,
        transcript: str,
    ) -> ExtractionResultDTO:
        cleaned_transcript = normalize_transcript(transcript)

        incident_type = self._extract_incident_type(cleaned_transcript)
        severity = self._extract_severity(cleaned_transcript)
        hazards = self._extract_hazards(cleaned_transcript, incident_type)
        location_raw = self._extract_location_raw(cleaned_transcript)
        people_trapped = self._extract_people_trapped(cleaned_transcript)
        injured_count = self._extract_injured_count(cleaned_transcript)
        dead_count = self._extract_dead_count(cleaned_transcript)
        resources_needed = self._estimate_resources(incident_type, severity)
        event_signature = build_event_signature(
            incident_type=incident_type,
            location_raw=location_raw,
        )

        missing_fields = []
        if not location_raw:
            missing_fields.append("location.raw_text")

        incident_description = self._build_incident_description(
            incident_type=incident_type,
            severity=severity,
            people_trapped=people_trapped,
            hazards=hazards,
        )

        location_confidence = 0.85 if location_raw else 0.20
        overall_confidence = compute_extraction_confidence(
            incident_type=incident_type,
            severity=severity,
            location_found=bool(location_raw),
        )

        return ExtractionResultDTO(
            call_id=call_id,
            timestamp=timestamp,
            raw_transcript=transcript,
            incident=IncidentDTO(
                type=incident_type,
                subtype=self._extract_subtype(incident_type, cleaned_transcript),
                severity=severity,
                description=incident_description,
            ),
            location=LocationDTO(
                raw_text=location_raw or "unknown",
                landmark=None,
                address=None,
                area=None,
                city=None,
                confidence=location_confidence,
            ),
            casualties=CasualtiesDTO(
                injured_count=injured_count,
                dead_count=dead_count,
                people_trapped=people_trapped,
            ),
            hazards=hazards,
            resources_needed=resources_needed,
            caller_context=CallerContextDTO(
                caller_role=CallerRole.UNKNOWN,
                emotional_state=self._extract_emotional_state(cleaned_transcript),
                callback_number=None,
            ),
            extraction_metadata=ExtractionMetadataDTO(
                missing_fields=missing_fields,
                contradictions_detected=[],
                overall_confidence=overall_confidence,
            ),
            deduplication=DeduplicationDTO(
                event_signature=event_signature,
                possible_duplicate_of=None,
            ),
        )

    def _extract_incident_type(self, text: str) -> IncidentType:
        if re.search(r"\b(fire|smoke|burning|flames|blaze|sparked)\b", text):
            return IncidentType.FIRE
        if re.search(r"\b(flood|waterlogging|water rising|submerged|overflowing)\b", text):
            return IncidentType.FLOOD
        if re.search(r"\b(accident|collision|crash|hit|vehicle hit|pileup)\b", text):
            return IncidentType.ROAD_ACCIDENT
        if re.search(r"\b(collapse|collapsed|building fell|structure fell|rubble)\b", text):
            return IncidentType.BUILDING_COLLAPSE
        if re.search(r"\b(gas leak|gas smell|leakage|smell of gas)\b", text):
            return IncidentType.GAS_LEAK
        if re.search(r"\b(injured|unconscious|heart attack|medical|bleeding|fainted|not breathing)\b", text):
            return IncidentType.MEDICAL_EMERGENCY
        if re.search(r"\b(earthquake|tremor|quake)\b", text):
            return IncidentType.EARTHQUAKE_DAMAGE
        if re.search(r"\b(landslide|mudslide)\b", text):
            return IncidentType.LANDSLIDE
        if re.search(r"\b(riot|violence|mob|stampede)\b", text):
            return IncidentType.CIVIL_DISTURBANCE
        if re.search(r"\b(power outage|power failure|transformer blast)\b", text):
            return IncidentType.POWER_FAILURE
        return IncidentType.UNKNOWN

    def _extract_subtype(self, incident_type: IncidentType, text: str) -> Optional[str]:
        if incident_type == IncidentType.FIRE:
            if re.search(r"\bbuilding\b", text):
                return "building_fire"
            if re.search(r"\b(vehicle|car|bus|truck)\b", text):
                return "vehicle_fire"
            if re.search(r"\b(shop|market|store)\b", text):
                return "commercial_fire"

        if incident_type == IncidentType.ROAD_ACCIDENT:
            if re.search(r"\btruck\b", text):
                return "truck_accident"
            if re.search(r"\bbus\b", text):
                return "bus_accident"
            if re.search(r"\bcar\b", text):
                return "car_accident"

        return None

    def _extract_severity(self, text: str) -> SeverityLevel:
        critical_patterns = [
            r"\btrapped\b",
            r"\bpeople inside\b",
            r"\bchildren inside\b",
            r"\bnot breathing\b",
            r"\bunconscious\b",
            r"\bmajor fire\b",
            r"\bbuilding collapse\b",
            r"\bmultiple injured\b",
            r"\bmany injured\b",
            r"\bbleeding heavily\b",
            r"\bexplosion\b",
            r"\bcannot breathe\b",
            r"\burgent\b",
            r"\bimmediately\b",
        ]
        high_patterns = [
            r"\bhelp fast\b",
            r"\bserious\b",
            r"\bbleeding\b",
            r"\bsmoke everywhere\b",
            r"\bflames\b",
            r"\bsevere\b",
            r"\broad blocked\b",
            r"\broad is blocked\b",
            r"\broad blocked completely\b",
            r"\broad accident\b",
        ]
        medium_patterns = [
            r"\binjured\b",
            r"\bblocked\b",
            r"\bleak\b",
            r"\baccident\b",
            r"\bsmoke\b",
        ]

        if any(re.search(pattern, text) for pattern in critical_patterns):
            return SeverityLevel.CRITICAL
        if any(re.search(pattern, text) for pattern in high_patterns):
            return SeverityLevel.HIGH
        if any(re.search(pattern, text) for pattern in medium_patterns):
            return SeverityLevel.MEDIUM
        return SeverityLevel.UNKNOWN

    def _extract_hazards(self, text: str, incident_type: IncidentType) -> List[str]:
        hazards = set()

        keyword_map = {
            "fire": r"\b(fire|flames|burning|blaze)\b",
            "smoke": r"\bsmoke\b",
            "flood_water": r"\b(flood|waterlogging|submerged|overflowing)\b",
            "gas": r"\b(gas leak|gas smell|smell of gas)\b",
            "debris": r"\b(debris|rubble)\b",
            "injury_risk": r"\b(injured|bleeding|unconscious|not breathing)\b",
            "traffic_block": r"\b(blocked|traffic jam|road blocked)\b",
        }

        for hazard, pattern in keyword_map.items():
            if re.search(pattern, text):
                hazards.add(hazard)

        if incident_type == IncidentType.FIRE:
            hazards.add("fire")
        elif incident_type == IncidentType.FLOOD:
            hazards.add("flood_water")

        return sorted(hazards)

    def _extract_location_raw(self, text: str) -> Optional[str]:
        patterns = [
            r"\bnear ([a-z0-9\s\-]+)",
            r"\bat ([a-z0-9\s\-]+)",
            r"\bon ([a-z0-9\s\-]+)",
            r"\bin ([a-z0-9\s\-]+)",
            r"\bin front of ([a-z0-9\s\-]+)",
            r"\bopposite ([a-z0-9\s\-]+)",
            r"\bbeside ([a-z0-9\s\-]+)",
            r"\bbehind ([a-z0-9\s\-]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                location = match.group(0).strip()
                location = re.split(self.LOCATION_STOP_WORDS_PATTERN, location)[0].strip(" ,.")
                return location if location else None

        return None

    def _extract_people_trapped(self, text: str) -> Optional[bool]:
        if re.search(r"\b(trapped|stuck inside|cannot get out|people inside|children inside)\b", text):
            return True
        return None

    def _extract_injured_count(self, text: str) -> Optional[int]:
        patterns = [
            r"\b((?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|a|an))\s+"
            r"(?:people|persons|men|women|children|child|person)?\s*injured\b",
            r"\b((?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|a|an))\s+"
            r"(?:people|persons|men|women|children|child|person)\b",
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                value = extract_leading_number_token(match.group(1))
                if value is not None:
                    return value

        if re.search(r"\binjured\b", text):
            return None
        return None

    def _extract_dead_count(self, text: str) -> Optional[int]:
        pattern = (
            r"\b((?:\d+|one|two|three|four|five|six|seven|eight|nine|ten))\s+"
            r"(?:people|persons|men|women|children|child|person)?\s*(dead|killed)\b"
        )
        match = re.search(pattern, text)
        if match:
            return extract_leading_number_token(match.group(1))
        return None

    def _estimate_resources(
        self,
        incident_type: IncidentType,
        severity: SeverityLevel,
    ) -> ResourcesNeededDTO:
        ambulance = 0
        fire_truck = 0
        police_unit = 0
        rescue_team = 0

        if incident_type == IncidentType.FIRE:
            fire_truck = 2
            ambulance = 1
            police_unit = 1
            rescue_team = 1 if severity in {SeverityLevel.HIGH, SeverityLevel.CRITICAL} else 0

        elif incident_type == IncidentType.ROAD_ACCIDENT:
            ambulance = 2
            police_unit = 1
            rescue_team = 1 if severity == SeverityLevel.CRITICAL else 0

        elif incident_type == IncidentType.MEDICAL_EMERGENCY:
            ambulance = 1

        elif incident_type == IncidentType.BUILDING_COLLAPSE:
            ambulance = 2
            fire_truck = 1
            police_unit = 1
            rescue_team = 2

        elif incident_type == IncidentType.GAS_LEAK:
            fire_truck = 1
            police_unit = 1
            rescue_team = 1

        elif incident_type == IncidentType.FLOOD:
            rescue_team = 2
            ambulance = 1
            police_unit = 1

        if severity == SeverityLevel.CRITICAL:
            ambulance += 1
            fire_truck += 1 if fire_truck > 0 else 0
            police_unit += 1 if police_unit > 0 else 0

        return ResourcesNeededDTO(
            ambulance=ambulance,
            fire_truck=fire_truck,
            police_unit=police_unit,
            rescue_team=rescue_team,
        )

    def _extract_emotional_state(self, text: str) -> Optional[str]:
        if re.search(r"\b(help|please|quickly|fast|urgent|immediately|hurry)\b", text):
            return "panicked"
        return "unknown"

    def _build_incident_description(
        self,
        *,
        incident_type: IncidentType,
        severity: SeverityLevel,
        people_trapped: Optional[bool],
        hazards: List[str],
    ) -> str:
        parts = [incident_type.value.replace("_", " ").title()]

        if hazards:
            parts.append(f"with hazards: {', '.join(hazards)}")

        if people_trapped:
            parts.append("possible trapped persons reported")

        parts.append(f"severity assessed as {severity.value}")

        return "; ".join(parts)