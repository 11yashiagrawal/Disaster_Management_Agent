import { Ambulance, Flame, Hospital, Shield } from "lucide-react";
import "./ResourcesPanel.css";

function formatLabel(value) {
  if (!value) return "Unknown";
  return value
    .replaceAll("_", " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function ResourceIcon({ type }) {
  if (type === "ambulance") return <Ambulance size={20} />;
  if (type === "fire_truck") return <Flame size={20} />;
  if (type === "police_unit") return <Shield size={20} />;
  if (type === "rescue_team") return <Hospital size={20} />;
  return <Shield size={20} />;
}

function ResourcesPanel({ assignedResources }) {
  return (
    <div className="panel resources-panel">
      <h3>4. Assigned Resources</h3>

      <div className="resource-list">
        {assignedResources.length === 0 ? (
          <p className="empty-state">No resources assigned yet.</p>
        ) : (
          assignedResources.map((unit) => (
            <div className="resource-row" key={unit.id}>
              <div className={`resource-icon ${unit.type}`}>
                <ResourceIcon type={unit.type} />
              </div>

              <div className="resource-info">
                <strong>{unit.id}</strong>
                <span>{formatLabel(unit.type)}</span>
              </div>

              <div className="enroute">
                <i />
                En Route
              </div>

              <span className="eta">ETA {unit.eta_minutes} min</span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

export default ResourcesPanel;