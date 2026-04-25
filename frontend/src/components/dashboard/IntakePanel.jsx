import { Flame, HeartPulse, Loader2, Send, Truck, Zap } from "lucide-react";
import VoiceIntakePanel from "../VoiceIntakePanel";
import "./IntakePanel.css";

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

function IntakePanel({
  transcript,
  selectedSample,
  loading,
  audioLoading,
  transcribedText,
  onTranscriptChange,
  onSelectedSampleChange,
  onResultChange,
  onTranscribedTextChange,
  onProcessCall,
  onUploadAudio,
  onProcessSampleAudio,
}) {
  const clearInput = () => {
    onTranscriptChange("");
    onResultChange(null);
    onSelectedSampleChange("");
    onTranscribedTextChange("");
  };

  return (
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
                onTranscriptChange("");
                onSelectedSampleChange("New");
                onResultChange(null);
                onTranscribedTextChange("");
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
                    onSelectedSampleChange(sample.label);
                    onTranscriptChange(sample.text);
                    onResultChange(null);
                    onTranscribedTextChange("");
                  }}
                >
                  <Icon size={22} className={`sample-icon ${sample.variant}`} />
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
            onChange={(e) => {
              onTranscriptChange(e.target.value);
              onSelectedSampleChange("");
            }}
          />

          {loading && (
            <div className="scan-overlay">
              <div className="scan-line" />
              <span>Analyzing transcript with NLP pipeline...</span>
            </div>
          )}

          <VoiceIntakePanel
            audioLoading={audioLoading}
            transcribedText={transcribedText}
            onUploadAudio={onUploadAudio}
            onProcessSampleAudio={onProcessSampleAudio}
          />

          <div className="editor-actions">
            <button
              className="process-btn"
              onClick={onProcessCall}
              disabled={loading || audioLoading || transcript.trim().length === 0}
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

            <button className="clear-btn" type="button" onClick={clearInput}>
              Clear
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default IntakePanel;