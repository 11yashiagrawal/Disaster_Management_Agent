import { useState } from "react";
import LandingPage from "./LandingPage";
import Dashboard from "./pages/Dashboard";

import "./App.css";
import "./LandingPage.css";

function ComingSoon({ title, goBack }) {
  return (
    <div className="coming-soon-page">
      <div className="coming-soon-card">
        <h1>{title}</h1>
        <p>This section is currently in progress.</p>

        <button type="button" onClick={goBack}>
          Back to Home
        </button>
      </div>
    </div>
  );
}

function App() {
  const [route, setRoute] = useState("landing");
  const [comingSoonTitle, setComingSoonTitle] = useState("");

  const goDashboard = () => {
    setRoute("dashboard");
  };

  const goComingSoon = (title) => {
    setComingSoonTitle(title);
    setRoute("coming-soon");
  };

  if (route === "dashboard") {
    return <Dashboard />;
  }

  if (route === "coming-soon") {
    return (
      <ComingSoon
        title={comingSoonTitle}
        goBack={() => setRoute("landing")}
      />
    );
  }

  return (
    <LandingPage
      onLaunchDashboard={goDashboard}
      onComingSoon={goComingSoon}
    />
  );
}

export default App;