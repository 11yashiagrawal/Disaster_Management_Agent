import { useEffect, useMemo, useState } from "react";
import { MapContainer, Marker, Polyline, Popup, TileLayer } from "react-leaflet";
import L from "leaflet";

function createVehicleIcon(unitType) {
  const emoji =
    unitType === "ambulance"
      ? "🚑"
      : unitType === "fire_truck"
      ? "🚒"
      : unitType === "police_unit"
      ? "🚓"
      : "🚨";

  return L.divIcon({
    className: "vehicle-div-icon",
    html: `<div class="vehicle-marker">${emoji}</div>`,
    iconSize: [42, 42],
    iconAnchor: [21, 21],
  });
}

const incidentIcon = L.divIcon({
  className: "incident-div-icon",
  html: `<div class="incident-marker">⚠️</div>`,
  iconSize: [46, 46],
  iconAnchor: [23, 23],
});

function interpolate(start, end, progress) {
  return [
    start[0] + (end[0] - start[0]) * progress,
    start[1] + (end[1] - start[1]) * progress,
  ];
}

function getAnimatedPosition(route, progress) {
  if (!route || route.length === 0) return null;

  const points = route.map((point) => [point.lat, point.lng]);

  if (points.length === 1) return points[0];

  const segmentProgress = progress * (points.length - 1);
  const segmentIndex = Math.min(Math.floor(segmentProgress), points.length - 2);
  const localProgress = segmentProgress - segmentIndex;

  return interpolate(points[segmentIndex], points[segmentIndex + 1], localProgress);
}

function DispatchMap({ result }) {
  const [progress, setProgress] = useState(0);

  const incident = result?.dispatch?.incident_coordinates;
  const assignedResources = result?.dispatch?.assigned_resources || [];

  useEffect(() => {
    setProgress(0);

    const interval = setInterval(() => {
      setProgress((value) => {
        if (value >= 1) return 1;

        // One full route takes around 18 seconds.
        // Since route has 3 points, one visual step/segment takes around 9 seconds.
        // Slow enough for judge demo.
        return Math.min(value + 0.01, 1);
      });
    }, 90);

    return () => clearInterval(interval);
  }, [result]);

  const center = useMemo(() => {
    if (!incident) return [28.5582, 77.2066];
    return [incident.lat, incident.lng];
  }, [incident]);

  if (!incident) return null;

  return (
    <div className="dispatch-map-shell">
      <MapContainer center={center} zoom={14} className="dispatch-map">
        <TileLayer
          attribution="&copy; OpenStreetMap"
          url="https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png"
        />

        <Marker position={[incident.lat, incident.lng]} icon={incidentIcon}>
          <Popup>
            <strong>Incident Location</strong>
            <br />
            Target emergency site
          </Popup>
        </Marker>

        {assignedResources.map((unit) => {
          const route = unit.route || [];
          const routePoints = route.map((point) => [point.lat, point.lng]);
          const animatedPosition = getAnimatedPosition(route, progress);

          if (!animatedPosition) return null;

          return (
            <div key={unit.id}>
              <Polyline
                positions={routePoints}
                pathOptions={{
                  weight: 5,
                  opacity: 0.75,
                  dashArray: "10 10",
                }}
              />

              <Marker
                position={animatedPosition}
                icon={createVehicleIcon(unit.type)}
              >
                <Popup>
                  <strong>{unit.id}</strong>
                  <br />
                  {unit.type.replaceAll("_", " ")}
                  <br />
                  ETA: {unit.eta_minutes} min
                </Popup>
              </Marker>
            </div>
          );
        })}
      </MapContainer>

      <div className="map-floating-card">
        <strong>Live Dispatch Tracking</strong>
        <span>{assignedResources.length} active units moving to target</span>
      </div>

      <div className="map-progress">
        <span>Vehicle movement simulation</span>
        <div>
          <i style={{ width: `${progress * 100}%` }} />
        </div>
      </div>
    </div>
  );
}

export default DispatchMap;