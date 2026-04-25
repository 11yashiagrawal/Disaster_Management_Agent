import { Activity } from "lucide-react";
import "./ComingSoonPage.css";

function ComingSoonPage({ pageName, icon: Icon = Activity }) {
  return (
    <section className="dashboard-grid">
      <div className="coming-soon-dashboard-card panel">
        <div className="coming-soon-dashboard-inner">
          <div className="coming-soon-dashboard-icon">
            <Icon size={30} />
          </div>

          <h3>{pageName}</h3>
          <p>
            This module is currently in progress. The page is routable and
            interactive, but the complete operational workflow will be added in
            the next iteration.
          </p>

          <div className="coming-soon-status-grid">
            {["UI Shell Ready", "Backend Hook Pending", "Feature In Progress"].map(
              (item) => (
                <div key={item}>{item}</div>
              )
            )}
          </div>
        </div>
      </div>
    </section>
  );
}

export default ComingSoonPage;