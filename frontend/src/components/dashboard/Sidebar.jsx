import {
  Activity,
  BarChart3,
  Building2,
  ChevronDown,
  Gauge,
  Grid3X3,
  LayoutDashboard,
  RadioTower,
  Settings,
  Shield,
  Siren,
} from "lucide-react";
import "./Sidebar.css";

const NAV_ITEMS = [
  { label: "Dashboard", icon: LayoutDashboard },
  { label: "Emergency Intake", icon: Activity },
  { label: "Triage Engine", icon: Siren },
  { label: "Dispatch Center", icon: RadioTower },
  { label: "Incident Feed", icon: Grid3X3 },
  { label: "Analytics", icon: BarChart3 },
  { label: "Resources", icon: Building2 },
  { label: "Reports", icon: Gauge },
  { label: "Settings", icon: Settings },
];

function Sidebar({ activePage, onNavigate }) {
  return (
    <aside className="sidebar">
      <button
        type="button"
        className="brand-block brand-button"
        onClick={() => onNavigate("Dashboard")}
      >
        <div className="brand-icon">
          <Shield size={28} />
        </div>
        <div>
          <h1>SHIELD</h1>
        </div>
      </button>

      <button
        type="button"
        className="system-card"
        onClick={() => onNavigate("System Status")}
      >
        <div>
          <span>System Status</span>
          <strong>Operational</strong>
        </div>
        <i />
      </button>

      <nav className="nav-card">
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;

          return (
            <button
              key={item.label}
              type="button"
              className={`nav-item ${activePage === item.label ? "active" : ""}`}
              onClick={() => onNavigate(item.label)}
            >
              <Icon size={18} />
              <span>{item.label}</span>
            </button>
          );
        })}
      </nav>

      <button
        type="button"
        className="control-room-card"
        onClick={() => onNavigate("Control Room")}
      >
        <div className="control-icon">
          <Shield size={24} />
        </div>
        <div>
          <strong>Control Room 01</strong>
          <span>Smart City Ops Center</span>
        </div>
        <ChevronDown size={18} />
      </button>
    </aside>
  );
}

export default Sidebar;