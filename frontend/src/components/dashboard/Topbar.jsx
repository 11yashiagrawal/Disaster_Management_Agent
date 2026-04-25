import { Bell, ChevronDown, User } from "lucide-react";
import "./Topbar.css";

const TOP_NAV = ["Platform", "Solutions", "Resources", "Company", "Support"];

function Topbar({ onNavigate, onTopSectionChange }) {
  return (
    <header className="navbar">
      <div className="nav-links">
        {TOP_NAV.map((item) => (
          <button
            key={item}
            type="button"
            className="top-nav-btn"
            onClick={() => {
              onTopSectionChange(item);
              onNavigate(item);
            }}
          >
            {item} <ChevronDown size={14} />
          </button>
        ))}
      </div>

      <div className="nav-actions">
        <button
          type="button"
          className="status-pill"
          onClick={() => onNavigate("System Status")}
        >
          <span />
          System Status: <strong>Operational</strong>
        </button>

        <button
          type="button"
          className="bell-wrap"
          onClick={() => onNavigate("Notifications")}
        >
          <Bell size={22} />
          <b>12</b>
        </button>

        <button
          type="button"
          className="profile"
          onClick={() => onNavigate("User Profile")}
        >
          <div className="avatar">
            <User size={22} />
          </div>
          <div>
            <strong>John Doe</strong>
            <span>Administrator</span>
          </div>
          <ChevronDown size={16} />
        </button>
      </div>
    </header>
  );
}

export default Topbar;