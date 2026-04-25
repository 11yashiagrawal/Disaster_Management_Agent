import { Clock3, Flame, Gauge, MapPin } from "lucide-react";
import "./TriagePanel.css";

function TriagePanel({
  priorityLevel,
  priorityScore,
  incidentType,
  location,
  dispatchStatus,
}) {
  return (
    <div className="panel triage-panel">
      <h3>2. AI Triage Result</h3>

      <div className="metric-grid">
        <div className="metric-card priority-metric">
          <Flame size={24} />
          <span>Priority Level</span>
          <strong>{priorityLevel}</strong>
          <small>{priorityLevel === "P1" ? "Critical" : "Risk Assigned"}</small>
        </div>

        <div className="metric-card">
          <Flame size={24} />
          <span>Incident Type</span>
          <strong>{incidentType}</strong>
        </div>

        <div className="metric-card">
          <Gauge size={24} />
          <span>Severity Score</span>
          <strong>
            {priorityScore} <em>/ 100</em>
          </strong>
        </div>

        <div className="metric-card">
          <MapPin size={24} />
          <span>Location Detected</span>
          <strong>{location}</strong>
        </div>

        <div className="metric-card">
          <Clock3 size={24} />
          <span>Detected At</span>
          <strong>
            Today,
            <br />
            04:23 PM
          </strong>
        </div>

        <div className="metric-card">
          <Clock3 size={24} />
          <span>Dispatch Status</span>
          <strong className="orange-text">{dispatchStatus}</strong>
        </div>
      </div>
    </div>
  );
}

export default TriagePanel;