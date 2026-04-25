from application.dto.dispatch_result_dto import AssignedResourceDTO, DispatchResultDTO
from application.dto.extraction_result_dto import ExtractionResultDTO
from domain.entities.resource_unit import ResourceUnit
from infrastructure.dispatch.mock_resource_registry import MockResourceRegistry


class DispatchService:
    """
    Assigns available emergency resources based on extracted resource requirements.

    This is a deterministic dispatch allocator for hackathon/demo usage.
    Later it can be replaced with graph routing + live availability.
    """

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
                assigned.append(
                    AssignedResourceDTO(
                        id=unit.id,
                        type=unit.type,
                        current_location=unit.current_location,
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
            assigned_resources=assigned,
            unfulfilled_resources=unfulfilled,
            dispatch_summary=self._build_summary(status, assigned, unfulfilled),
        )

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