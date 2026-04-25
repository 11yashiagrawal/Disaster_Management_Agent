import {
  ArrowRight,
  Brain,
  ChevronDown,
  Play,
  Shield,
  Truck,
  Radio,
  BarChart3,
  Landmark,
  Building2,
  BadgePlus,
} from "lucide-react";
import "./LandingPage.css";

const TRUSTED = [
  { label: "Government\nof India", icon: Landmark },
  { label: "Smart City\nMission", icon: Building2 },
  { label: "NITI Aayog", icon: Landmark },
  { label: "Ministry of Home Affairs\nGovernment of India", icon: Landmark },
  { label: "Disaster Management\nAuthorities", icon: Shield },
  { label: "Police\nDepartments", icon: BadgePlus },
  { label: "+120 More\nOrganizations", icon: Building2 },
];

const FEATURES = [
  {
    title: "NLP Transcript Intake",
    desc: "AI reads and understands messy emergency calls in real-time.",
    icon: Radio,
    color: "red",
  },
  {
    title: "Smart Triage Engine",
    desc: "Automatically detects incident type, severity, and priority using advanced AI.",
    icon: Brain,
    color: "blue",
  },
  {
    title: "Real-Time Dispatch",
    desc: "Instantly assign the nearest available resources with optimal routing.",
    icon: Truck,
    color: "green",
  },
  {
    title: "Resource Analytics",
    desc: "Get deep insights into performance, response times, and resource utilization.",
    icon: BarChart3,
    color: "purple",
  },
];

const NAV_ITEMS = ["Platform", "Solutions", "Resources", "Company", "Support"];

function LandingPage({ onLaunchDashboard, onComingSoon }) {
  const launch = () => {
    if (onLaunchDashboard) onLaunchDashboard();
  };

  const openComingSoon = (title) => {
    if (onComingSoon) onComingSoon(title);
  };

  return (
    <div className="landing-page">
      <nav className="landing-nav">
        <button className="landing-brand" type="button" onClick={launch}>
          <div className="landing-brand-icon">
            <Shield size={34} />
          </div>
          <div>
            <h1>S . H . I . E . L . D</h1>
          </div>
        </button>

        <div className="landing-links">
          {NAV_ITEMS.map((item) => (
            <button
              key={item}
              type="button"
              onClick={() => openComingSoon(item)}
            >
              {item}
              <ChevronDown size={15} />
            </button>
          ))}
        </div>

        <div className="landing-actions">
          <button
            className="landing-status"
            type="button"
            onClick={() => openComingSoon("System Status")}
          >
            <i />
            <span>
              System Status: <strong>Operational</strong>
            </span>
          </button>

          <button className="launch-btn" type="button" onClick={launch}>
            Launch Dashboard
            <ArrowRight size={18} />
          </button>
        </div>
      </nav>

      <section className="landing-hero">
        <div className="hero-copy">
          <p className="hero-kicker">
            <Shield size={18} />
            AI-POWERED DISASTER MANAGEMENT PLATFORM
          </p>

          <h2 className="hero-title">
            AI-Powered
            <span>Emergency Response</span>
            for Smart Cities
          </h2>

          <p className="hero-desc">
            ResQ Grid uses advanced AI and real-time data to understand,
            prioritize, and respond to emergencies faster than ever.
            <br />
            Smarter triage. Faster dispatch. Safer cities.
          </p>

          <div className="hero-buttons">
            <button className="primary-cta" type="button" onClick={launch}>
              Launch Dashboard
              <ArrowRight size={18} />
            </button>

            <button
              className="secondary-cta"
              type="button"
              onClick={() => openComingSoon("Watch Demo")}
            >
              <span className="play-icon">
                <Play size={14} fill="currentColor" />
              </span>
              Watch Demo
            </button>
          </div>
        </div>

        <button
          className="hero-visual"
          type="button"
          aria-label="Open dashboard preview"
          onClick={launch}
        />
      </section>

      <section className="trust-strip">
        <h3>TRUSTED BY GOVERNMENTS & LEADING ORGANIZATIONS</h3>

        <div className="trust-logos">
          {TRUSTED.map((item) => {
            const Icon = item.icon;
            return (
              <button
                className="trust-item"
                type="button"
                key={item.label}
                onClick={() => openComingSoon(item.label.replaceAll("\n", " "))}
              >
                <Icon className="trust-mark" />
                <span>{item.label}</span>
              </button>
            );
          })}
        </div>
      </section>

      <section className="feature-section">
        <div className="feature-grid">
          {FEATURES.map((feature) => {
            const Icon = feature.icon;
            return (
              <article className="feature-card" key={feature.title}>
                <button
                  className={`feature-icon ${feature.color}`}
                  type="button"
                  onClick={() => openComingSoon(feature.title)}
                  aria-label={feature.title}
                >
                  <Icon size={34} />
                </button>

                <div className="feature-copy">
                  <h4>{feature.title}</h4>
                  <p>{feature.desc}</p>

                  <button
                    className={`learn-link ${feature.color}`}
                    type="button"
                    onClick={() => openComingSoon(feature.title)}
                  >
                    Learn more
                    <ArrowRight size={16} />
                  </button>
                </div>
              </article>
            );
          })}
        </div>
      </section>
    </div>
  );
}

export default LandingPage;