import { useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  Ambulance,
  BarChart3,
  Bell,
  Building2,
  CalendarDays,
  CheckCircle2,
  ChevronDown,
  Clock3,
  Copy,
  Flame,
  Gauge,
  Grid3X3,
  HeartPulse,
  Home,
  Hospital,
  LayoutDashboard,
  Loader2,
  MapPin,
  RadioTower,
  Send,
  Settings,
  Shield,
  Siren,
  Truck,
  User,
  Zap,
} from "lucide-react";
import "./App.css";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

const SAMPLE_TRANSCRIPTS = [
  {
    label: "Gas Leak - Apartment",
    time: "04:23 PM",
    icon: Flame,
    variant: "danger",
    text: "gass smel in apartmant people faintng cant breth please hurry its near green park metro.. 4th floor... need help fast",
  },
  {
    label: "Road Traffic Accident",
    time: "04:18 PM",
    icon: Truck,
    variant: "amber",
    text: "bus and kar hit badli near sity hospital rood blockd two injrd please send amblance",
  },
  {
    label: "Building Fire",
    time: "04:12 PM",
    icon: Flame,
    variant: "danger",
    text: "aaaa please hurry smok and fire near green park metro people trapped inside",
  },
  {
    label: "Medical Emergency",
    time: "04:08 PM",
    icon: HeartPulse,
    variant: "green",
    text: "man unconsious outside skul not breathng properli please come fast",
  },
  {
    label: "Electricity Down",
    time: "04:02 PM",
    icon: Zap,
    variant: "blue",
    text: "transformer blast near sector 14 market power failure smoke around area",
  },
];

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

const TOP_NAV = ["Platform", "Solutions", "Resources", "Company", "Support"];

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

function formatLabel(value) {
  if (!value) return "Unknown";
  return value
    .replaceAll("_", " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function ComingSoonPage({ pageName, icon: Icon = Activity }) {
  return (
    <section className="dashboard-grid">
      <div className="panel" style={{ gridColumn: "1 / -1", padding: "48px" }}>
        <div style={{ maxWidth: "720px" }}>
          <div
            style={{
              width: 58,
              height: 58,
              borderRadius: 18,
              background: "#fff1f2",
              color: "#ff313d",
              display: "grid",
              placeItems: "center",
              marginBottom: 18,
            }}
          >
            <Icon size={30} />
          </div>

          <h3 style={{ fontSize: 28, marginBottom: 10 }}>{pageName}</h3>
          <p style={{ color: "#697386", lineHeight: 1.7, margin: 0 }}>
            This module is currently in progress. The page is routable and
            interactive, but the complete operational workflow will be added in
            the next iteration.
          </p>

          <div
            style={{
              marginTop: 28,
              display: "grid",
              gridTemplateColumns: "repeat(3, minmax(0, 1fr))",
              gap: 14,
            }}
          >
            {["UI Shell Ready", "Backend Hook Pending", "Feature In Progress"].map(
              (item) => (
                <div
                  key={item}
                  style={{
                    border: "1px solid #dfe5ee",
                    borderRadius: 12,
                    padding: 18,
                    background: "#fbfdff",
                    fontWeight: 700,
                    color: "#111827",
                  }}
                >
                  {item}
                </div>
              )
            )}
          </div>
        </div>
      </div>
    </section>
  );
}

function Dashboard() {
  const [activePage, setActivePage] = useState("Emergency Intake");
  const [topSection, setTopSection] = useState("");
  const [transcript, setTranscript] = useState(SAMPLE_TRANSCRIPTS[0].text);
  const [selectedSample, setSelectedSample] = useState(SAMPLE_TRANSCRIPTS[0].label);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const now = useMemo(() => {
    return new Intl.DateTimeFormat("en-IN", {
      weekday: "long",
      month: "long",
      day: "numeric",
      year: "numeric",
    }).format(new Date());
  }, []);

  const activeNavItem =
    NAV_ITEMS.find((item) => item.label === activePage) || NAV_ITEMS[1];

  const processCall = async () => {
    try {
      setLoading(true);

      const res = await fetch(`${API_BASE_URL}/api/emergency-calls/process`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          call_id: `CALL_UI_${Date.now()}`,
          transcript,
        }),
      });

      const data = await res.json();
      setResult(data);
    } catch (error) {
      console.error(error);
      alert("Backend not running.");
    } finally {
      setLoading(false);
    }
  };

  const copyJson = async () => {
    if (!result) return;
    await navigator.clipboard.writeText(JSON.stringify(result, null, 2));
  };

  const priorityLevel = result?.triage?.priority_level || "—";
  const priorityScore = result?.triage?.priority_score ?? "—";
  const incidentType = formatLabel(result?.extraction?.incident?.type);
  const location = result?.extraction?.location?.raw_text || "Awaiting processing";
  const dispatchStatus = formatLabel(result?.dispatch?.dispatch_status);
  const assignedResources = result?.dispatch?.assigned_resources || [];
  const reasoningItems = result?.triage?.triage_reason
    ? result.triage.triage_reason
        .split(";")
        .map((item) => item.trim())
        .filter(Boolean)
    : [];

  const renderPage = () => {
    if (activePage !== "Emergency Intake") {
      return (
        <ComingSoonPage pageName={activePage} icon={activeNavItem.icon} />
      );
    }

    return (
      <section className="dashboard-grid">
        <div className="panel intake-panel">
          <div className="panel-title">
            <div>
              <h3>1. Incoming Transcripts</h3>
              <p>Paste or select a sample transcript to process</p>
            </div>
          </div>

          <div className="intake-body">
            <div className="sample-list-wrap">
              <div className="sample-head">
                <span>Sample Transcripts</span>
                <button
                  type="button"
                  className="new-btn"
                  onClick={() => {
                    setTranscript("");
                    setSelectedSample("New");
                    setResult(null);
                  }}
                >
                  + New
                </button>
              </div>

              <div className="sample-list">
                {SAMPLE_TRANSCRIPTS.map((sample) => {
                  const Icon = sample.icon;
                  return (
                    <button
                      key={sample.label}
                      type="button"
                      className={`sample-card ${
                        selectedSample === sample.label ? "selected" : ""
                      }`}
                      onClick={() => {
                        setSelectedSample(sample.label);
                        setTranscript(sample.text);
                        setResult(null);
                      }}
                    >
                      <Icon
                        size={22}
                        className={`sample-icon ${sample.variant}`}
                      />
                      <div>
                        <strong>{sample.label}</strong>
                        <span>{sample.time}</span>
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>

            <div className="transcript-editor">
              <div className="editor-head">
                <span>Selected Transcript</span>
                <small>{transcript.length}/5000</small>
              </div>

              <textarea
                value={transcript}
                onChange={(e) => setTranscript(e.target.value)}
              />

              {loading && (
                <div className="scan-overlay">
                  <div className="scan-line" />
                  <span>Analyzing transcript with NLP pipeline...</span>
                </div>
              )}

              <div className="editor-actions">
                <button
                  className="process-btn"
                  onClick={processCall}
                  disabled={loading || transcript.trim().length === 0}
                >
                  {loading ? (
                    <>
                      <Loader2 size={17} className="spin" />
                      Processing...
                    </>
                  ) : (
                    <>
                      <Send size={17} />
                      Process Transcript
                    </>
                  )}
                </button>

                <button
                  className="clear-btn"
                  type="button"
                  onClick={() => {
                    setTranscript("");
                    setResult(null);
                    setSelectedSample("");
                  }}
                >
                  Clear
                </button>
              </div>
            </div>
          </div>
        </div>

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

        <div className="panel resources-panel">
          <h3>4. Assigned Resources</h3>

          <div className="resource-list">
            {assignedResources.length === 0 ? (
              <p className="empty-state">No resources assigned yet.</p>
            ) : (
              assignedResources.map((unit) => (
                <div className="resource-row" key={unit.id}>
                  <div className={`resource-icon ${unit.type}`}>
                    {unit.type === "ambulance" ? <Ambulance size={20} /> : null}
                    {unit.type === "fire_truck" ? <Flame size={20} /> : null}
                    {unit.type === "police_unit" ? <Shield size={20} /> : null}
                    {unit.type === "rescue_team" ? <Hospital size={20} /> : null}
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

        {result && (
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
                  {result.triage.recommended_actions.map((action) => (
                    <li key={action}>
                      <span className="red-dot" />
                      <span>{action}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        )}

        {result && (
          <div className="panel json-panel">
            <div className="json-head">
              <h3>5. Full System Output (Raw Data)</h3>
              <button type="button" onClick={copyJson}>
                <Copy size={16} />
                Copy JSON
              </button>
            </div>

            <pre>{JSON.stringify(result, null, 2)}</pre>
          </div>
        )}

        <div className="panel timeline-panel">
          <h3>6. Dispatch Timeline</h3>

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
      </section>
    );
  };

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <button
          type="button"
          className="brand-block brand-button"
          onClick={() => setActivePage("Dashboard")}
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
          onClick={() => setActivePage("System Status")}
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
                className={`nav-item ${
                  activePage === item.label ? "active" : ""
                }`}
                onClick={() => setActivePage(item.label)}
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
          onClick={() => setActivePage("Control Room")}
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

      <main className="workspace">
        <header className="navbar">
          <div className="nav-links">
            {TOP_NAV.map((item) => (
              <button
                key={item}
                type="button"
                className="top-nav-btn"
                onClick={() => {
                  setTopSection(item);
                  setActivePage(item);
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
              onClick={() => setActivePage("System Status")}
            >
              <span />
              System Status: <strong>Operational</strong>
            </button>

            <button
              type="button"
              className="bell-wrap"
              onClick={() => setActivePage("Notifications")}
            >
              <Bell size={22} />
              <b>12</b>
            </button>

            <button
              type="button"
              className="profile"
              onClick={() => setActivePage("User Profile")}
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

        <section className="page-head">
          <div>
            <h2>{activePage}</h2>
            <p>
              {activePage === "Emergency Intake"
                ? "AI-powered transcript processing and incident triage"
                : topSection
                ? `${topSection} section is currently in progress`
                : "Module is currently in progress"}
            </p>
          </div>

          <button
            type="button"
            className="time-card"
            onClick={() => setActivePage("System Clock")}
          >
            <CalendarDays size={20} />
            <div>
              <span>{now}</span>
              <strong>04:25:36 PM</strong>
            </div>
          </button>
        </section>

        {renderPage()}
      </main>
    </div>
  );
}

export default Dashboard;