import { Copy, Download } from "lucide-react";
import "./RawJsonPanel.css";

function RawJsonPanel({ result, onCopyJson }) {
  const exportReport = () => {
    const blob = new Blob([JSON.stringify(result, null, 2)], {
      type: "application/json",
    });

    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");

    link.href = url;
    link.download = `incident-report-${Date.now()}.json`;
    link.click();

    URL.revokeObjectURL(url);
  };

  return (
    <div className="panel json-panel">
      <div className="json-head">
        <h3>5. Full System Output (Raw Data)</h3>

        <div className="json-actions">
          <button type="button" onClick={onCopyJson}>
            <Copy size={16} />
            Copy JSON
          </button>

          <button type="button" onClick={exportReport}>
            <Download size={16} />
            Export Report
          </button>
        </div>
      </div>

      <pre>{JSON.stringify(result, null, 2)}</pre>
    </div>
  );
}

export default RawJsonPanel;