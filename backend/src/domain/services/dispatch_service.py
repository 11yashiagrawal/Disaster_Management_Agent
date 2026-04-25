from application.dto.dispatch_result_dto import (
    AssignedResourceDTO,
    CoordinateDTO,
    DispatchResultDTO,
)
from application.dto.extraction_result_dto import ExtractionResultDTO
from domain.entities.resource_unit import ResourceUnit
from infrastructure.dispatch.mock_resource_registry import MockResourceRegistry


class DispatchService:
    """
    Assigns available emergency resources and returns mock map coordinates/routes.
    """

    LOCATION_COORDINATES = {
        "green_park_metro": CoordinateDTO(lat=28.5582, lng=77.2066),
        "metro_station": CoordinateDTO(lat=28.5582, lng=77.2066),
        "city_hospital": CoordinateDTO(lat=28.5672, lng=77.2105),
        "central_chowk": CoordinateDTO(lat=28.5524, lng=77.1989),
        "fire_station_a": CoordinateDTO(lat=28.5708, lng=77.1995),
        "sector_14": CoordinateDTO(lat=28.5451, lng=77.2147),
        "district_hq": CoordinateDTO(lat=28.5746, lng=77.2162),
        "river_bridge": CoordinateDTO(lat=28.5488, lng=77.2248),
        "unknown": CoordinateDTO(lat=28.5582, lng=77.2066),
    }

    MOCK_ETA_BY_LOCATION = {
        "city_hospital": 4,
        "central_chowk": 7,
        "fire_station_a": 6,
        "sector_14": 9,
        "metro_station": 5,
        "district_hq": 12,
    }

    def __init__(self, registry: MockResourceRegistry | None = None) -> None:
        self.registry = registry or MockResourceRegistry()

    def dispatch(self, extraction: ExtractionResultDTO) -> DispatchResultDTO:
        target_coordinates = self._resolve_incident_coordinates(
            extraction.location.raw_text
        )

        requirements = {
            "ambulance": extraction.resources_needed.ambulance,
            "fire_truck": extraction.resources_needed.fire_truck,
            "police_unit": extraction.resources_needed.police_unit,
            "rescue_team": extraction.resources_needed.rescue_team,
        }

        assigned: list[AssignedResourceDTO] = []
        unfulfilled: list[str] = []

        for resource_type, required_count in requirements.items():
            if required_count <= 0:
                continue

            available_units = self.registry.get_available_by_type(resource_type)
            selected_units = self._select_nearest_units(available_units, required_count)

            for unit in selected_units:
                current_coordinates = self._get_location_coordinates(
                    unit.current_location
                )

                assigned.append(
                    AssignedResourceDTO(
                        id=unit.id,
                        type=unit.type,
                        current_location=unit.current_location,
                        current_coordinates=current_coordinates,
                        target_coordinates=target_coordinates,
                        route=self._build_mock_route(
                            start=current_coordinates,
                            end=target_coordinates,
                        ),
                        eta_minutes=self._estimate_eta(unit),
                    )
                )
                self.registry.mark_assigned(unit.id)

            shortage = required_count - len(selected_units)
            if shortage > 0:
                unfulfilled.extend([resource_type] * shortage)

        if assigned and not unfulfilled:
            status = "fully_assigned"
        elif assigned and unfulfilled:
            status = "partially_assigned"
        else:
            status = "not_assigned"

        return DispatchResultDTO(
            dispatch_status=status,
            incident_coordinates=target_coordinates,
            assigned_resources=assigned,
            unfulfilled_resources=unfulfilled,
            dispatch_summary=self._build_summary(status, assigned, unfulfilled),
        )

    def _resolve_incident_coordinates(self, location_raw: str) -> CoordinateDTO:
        normalized = location_raw.lower().replace(" ", "_")

        if "green_park" in normalized or "metro" in normalized:
            return self.LOCATION_COORDINATES["green_park_metro"]

        if "city_hospital" in normalized or "hospital" in normalized:
            return self.LOCATION_COORDINATES["city_hospital"]

        if "sector_14" in normalized or "market" in normalized:
            return self.LOCATION_COORDINATES["sector_14"]

        if "river_bridge" in normalized or "bridge" in normalized:
            return self.LOCATION_COORDINATES["river_bridge"]

        return self.LOCATION_COORDINATES["unknown"]

    def _get_location_coordinates(self, location: str) -> CoordinateDTO:
        return self.LOCATION_COORDINATES.get(
            location,
            self.LOCATION_COORDINATES["unknown"],
        )

    def _build_mock_route(
        self,
        *,
        start: CoordinateDTO,
        end: CoordinateDTO,
    ) -> list[CoordinateDTO]:
        midpoint = CoordinateDTO(
            lat=(start.lat + end.lat) / 2 + 0.002,
            lng=(start.lng + end.lng) / 2 - 0.002,
        )

        return [start, midpoint, end]

    def _select_nearest_units(
        self,
        units: list[ResourceUnit],
        required_count: int,
    ) -> list[ResourceUnit]:
        sorted_units = sorted(units, key=self._estimate_eta)
        return sorted_units[:required_count]

    def _estimate_eta(self, unit: ResourceUnit) -> int:
        return self.MOCK_ETA_BY_LOCATION.get(unit.current_location, 15)

    def _build_summary(
        self,
        status: str,
        assigned: list[AssignedResourceDTO],
        unfulfilled: list[str],
    ) -> str:
        if status == "fully_assigned":
            return f"All required resources assigned. Total units dispatched: {len(assigned)}."

        if status == "partially_assigned":
            return (
                f"{len(assigned)} units assigned, but shortages remain for: "
                f"{', '.join(unfulfilled)}."
            )

        return "No resources could be assigned from the available registry."