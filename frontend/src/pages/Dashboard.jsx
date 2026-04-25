import { useMemo, useState } from "react";
import "./Dashboard.css";

import Sidebar from "../components/dashboard/Sidebar";
import Topbar from "../components/dashboard/Topbar";
import PageHeader from "../components/dashboard/PageHeader";
import ComingSoonPage from "../components/dashboard/ComingSoonPage";
import IntakePanel from "../components/dashboard/IntakePanel";
import TriagePanel from "../components/dashboard/TriagePanel";
import ResourcesPanel from "../components/dashboard/ResourcesPanel";
import InsightPanel from "../components/dashboard/InsightPanel";
import RawJsonPanel from "../components/dashboard/RawJsonPanel";
import DispatchTimeline from "../components/dashboard/DispatchTimeline";
import DispatchMap from "../components/DispatchMap";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

function formatLabel(value) {
  if (!value) return "Unknown";
  return value
    .replaceAll("_", " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function Dashboard() {
  const [activePage, setActivePage] = useState("Emergency Intake");
  const [topSection, setTopSection] = useState("");
  const [transcript, setTranscript] = useState(
    "gass smel in apartmant people faintng cant breth please hurry its near green park metro.. 4th floor... need help fast"
  );
  const [selectedSample, setSelectedSample] = useState("Gas Leak - Apartment");
  const [loading, setLoading] = useState(false);
  const [audioLoading, setAudioLoading] = useState(false);
  const [transcribedText, setTranscribedText] = useState("");
  const [result, setResult] = useState(null);

  const now = useMemo(() => {
    return new Intl.DateTimeFormat("en-IN", {
      weekday: "long",
      month: "long",
      day: "numeric",
      year: "numeric",
    }).format(new Date());
  }, []);

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

  const processAudioFile = async (file) => {
    if (!file) return;

    try {
      setAudioLoading(true);

      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch(`${API_BASE_URL}/api/audio/process`, {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      setTranscribedText(data.transcript || "");
      setTranscript(data.transcript || "");
      setResult(data.result || null);
      setSelectedSample("");
    } catch (error) {
      console.error(error);
      alert("Audio processing failed. Make sure backend is running.");
    } finally {
      setAudioLoading(false);
    }
  };

  const processSampleAudio = async (sample) => {
    try {
      setAudioLoading(true);

      const response = await fetch(sample.path);
      const blob = await response.blob();

      const file = new File([blob], `${sample.label}.mp3`, {
        type: "audio/mpeg",
      });

      await processAudioFile(file);
    } catch (error) {
      console.error(error);
      alert("Sample audio processing failed.");
    } finally {
      setAudioLoading(false);
    }
  };

  const copyJson = async () => {
    if (!result) return;
    await navigator.clipboard.writeText(JSON.stringify(result, null, 2));
  };

  const priorityLevel = result?.triage?.priority_level || "—";
  const priorityScore = result?.triage?.priority_score ?? "—";
  const incidentType = formatLabel(result?.extraction?.incident?.type);
  const location =
    result?.extraction?.location?.raw_text || "Awaiting processing";
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
      return <ComingSoonPage pageName={activePage} />;
    }

    return (
      <section className="dashboard-grid">
        <IntakePanel
          transcript={transcript}
          selectedSample={selectedSample}
          loading={loading}
          audioLoading={audioLoading}
          transcribedText={transcribedText}
          onTranscriptChange={setTranscript}
          onSelectedSampleChange={setSelectedSample}
          onResultChange={setResult}
          onTranscribedTextChange={setTranscribedText}
          onProcessCall={processCall}
          onUploadAudio={processAudioFile}
          onProcessSampleAudio={processSampleAudio}
        />

        <TriagePanel
          priorityLevel={priorityLevel}
          priorityScore={priorityScore}
          incidentType={incidentType}
          location={location}
          dispatchStatus={dispatchStatus}
        />

        <ResourcesPanel assignedResources={assignedResources} />

        {result && (
          <InsightPanel
            priorityLevel={priorityLevel}
            reasoningItems={reasoningItems}
            actions={result.triage.recommended_actions}
          />
        )}

        {result && (
          <div className="panel map-panel">
            <h3>6. Live Dispatch Tracker</h3>
            <DispatchMap result={result} />
          </div>
        )}

        {result && <RawJsonPanel result={result} onCopyJson={copyJson} />}

        <DispatchTimeline />
      </section>
    );
  };

  return (
    <div className="app-shell">
      <Sidebar activePage={activePage} onNavigate={setActivePage} />

      <main className="workspace">
        <Topbar
          onNavigate={setActivePage}
          onTopSectionChange={setTopSection}
        />

        <PageHeader
          activePage={activePage}
          topSection={topSection}
          now={now}
          onNavigate={setActivePage}
        />

        {renderPage()}
      </main>
    </div>
  );
}

export default Dashboard;