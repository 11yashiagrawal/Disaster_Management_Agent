import { CalendarDays } from "lucide-react";
import "./PageHeader.css";

function PageHeader({ activePage, topSection, now, onNavigate }) {
  return (
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
        onClick={() => onNavigate("System Clock")}
      >
        <CalendarDays size={20} />
        <div>
          <span>{now}</span>
          <strong>04:25:36 PM</strong>
        </div>
      </button>
    </section>
  );
}

export default PageHeader;