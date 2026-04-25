import { useRef, useState } from "react";
import "./VoiceIntakePanel.css";

const SAMPLE_AUDIO_FILES = [
  { label: "Fire", path: "/audio/fire.mp3" },
  { label: "Accident", path: "/audio/accident.mp3" },
  { label: "Gas Leak", path: "/audio/gas.mp3" },
  { label: "Collapse", path: "/audio/collapse.mp3" },
];

function VoiceIntakePanel({
  audioLoading,
  transcribedText,
  onUploadAudio,
  onProcessSampleAudio,
}) {
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  const [recording, setRecording] = useState(false);
  const [recordedAudioUrl, setRecordedAudioUrl] = useState("");

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      audioChunksRef.current = [];

      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, {
          type: "audio/webm",
        });

        const audioUrl = URL.createObjectURL(audioBlob);
        setRecordedAudioUrl(audioUrl);

        const file = new File([audioBlob], `recorded-call-${Date.now()}.webm`, {
          type: "audio/webm",
        });

        await onUploadAudio(file);

        stream.getTracks().forEach((track) => track.stop());
      };

      mediaRecorder.start();
      setRecording(true);
    } catch (error) {
      console.error(error);
      alert("Microphone permission denied or unavailable.");
    }
  };

  const stopRecording = () => {
    if (!mediaRecorderRef.current) return;

    mediaRecorderRef.current.stop();
    setRecording(false);
  };

  return (
    <div className="voice-panel">
      <div className="voice-head">
        <div>
          <strong>Voice Intake</strong>
          <span>Listen, record, upload, or process audio directly into the pipeline</span>
        </div>

        <div className="voice-actions">
          <button
            type="button"
            className={`voice-record-btn ${recording ? "recording" : ""}`}
            onClick={recording ? stopRecording : startRecording}
            disabled={audioLoading}
          >
            {recording ? "Stop Recording" : "Record Call"}
          </button>

          <label className="voice-upload-btn">
            {audioLoading ? "Transcribing..." : "Upload Audio"}
            <input
              hidden
              type="file"
              accept="audio/*"
              onChange={(e) => onUploadAudio(e.target.files?.[0])}
            />
          </label>
        </div>
      </div>

      {recordedAudioUrl && (
        <div className="recorded-audio-box">
          <strong>Recorded Audio</strong>
          <audio controls src={recordedAudioUrl} />
        </div>
      )}

      <div className="voice-horizontal-list">
        {SAMPLE_AUDIO_FILES.map((sample) => (
          <div className="voice-horizontal-card" key={sample.label}>
            <div className="voice-meta">
              <strong>{sample.label}</strong>
              <span>Sample call audio</span>
            </div>

            <audio controls src={sample.path} />

            <button
              type="button"
              onClick={() => onProcessSampleAudio(sample)}
              disabled={audioLoading}
            >
              Use Audio
            </button>
          </div>
        ))}
      </div>

      {transcribedText && (
        <div className="voice-transcript">
          <strong>Generated Transcript</strong>
          <p>{transcribedText}</p>
        </div>
      )}
    </div>
  );
}

export default VoiceIntakePanel;