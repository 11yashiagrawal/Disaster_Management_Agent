from domain.entities.resource_unit import ResourceUnit


class MockResourceRegistry:
    def __init__(self) -> None:
        self.resources = [
            ResourceUnit(id="AMB_001", type="ambulance", status="available", current_location="city_hospital"),
            ResourceUnit(id="AMB_002", type="ambulance", status="available", current_location="central_chowk"),
            ResourceUnit(id="FIRE_001", type="fire_truck", status="available", current_location="fire_station_a"),
            ResourceUnit(id="FIRE_002", type="fire_truck", status="available", current_location="sector_14"),
            ResourceUnit(id="POL_001", type="police_unit", status="available", current_location="metro_station"),
            ResourceUnit(id="RES_001", type="rescue_team", status="available", current_location="district_hq"),
        ]

    def get_available_by_type(self, resource_type: str) -> list[ResourceUnit]:
        return [
            resource
            for resource in self.resources
            if resource.type == resource_type and resource.status == "available"
        ]

    def mark_assigned(self, resource_id: str) -> None:
        for resource in self.resources:
            if resource.id == resource_id:
                resource.status = "assigned"
                return