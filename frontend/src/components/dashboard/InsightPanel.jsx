import { CheckCircle2 } from "lucide-react";
import "./InsightPanel.css";

function InsightPanel({ priorityLevel, reasoningItems, actions }) {
  return (
    <div className="panel insight-panel">
      <h3>3. AI Reasoning & Insights</h3>

      <div className="insight-box">
        <div>
          <h4>Why {priorityLevel} Priority?</h4>
          <ul>
            {reasoningItems.map((reason) => (
              <li key={reason}>
                <CheckCircle2 size={15} />
                <span>{reason}</span>
              </li>
            ))}
          </ul>
        </div>

        <div>
          <h4>Recommended Actions</h4>
          <ul>
            {actions.map((action) => (
              <li key={action}>
                <span className="red-dot" />
                <span>{action}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}

export default InsightPanel;