import "./DispatchTimeline.css";

const TIMELINE_ITEMS = [
  {
    label: "Incident Received",
    detail: "Transcript captured from emergency call",
    time: "04:23 PM",
    color: "red",
  },
  {
    label: "AI Processing",
    detail: "Transcript analyzed and classified",
    time: "04:23 PM",
    color: "blue",
  },
  {
    label: "Triage Completed",
    detail: "Priority assigned by risk engine",
    time: "04:24 PM",
    color: "purple",
  },
  {
    label: "Resources Allocated",
    detail: "Units assigned and notified",
    time: "04:24 PM",
    color: "orange",
  },
  {
    label: "Units En Route",
    detail: "Responding to incident location",
    time: "04:25 PM",
    color: "green",
  },
];

function DispatchTimeline() {
  return (
    <div className="panel timeline-panel">
      <h3>7. Dispatch Timeline</h3>

      <div className="timeline">
        {TIMELINE_ITEMS.map((item) => (
          <div className="timeline-item" key={item.label}>
            <span className={`timeline-dot ${item.color}`} />
            <time>{item.time}</time>
            <div>
              <strong>{item.label}</strong>
              <p>{item.detail}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default DispatchTimeline;