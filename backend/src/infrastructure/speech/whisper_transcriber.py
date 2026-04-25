import tempfile
from pathlib import Path

import whisper


class WhisperTranscriber:
    def __init__(self, model_name: str = "base") -> None:
        self.model = whisper.load_model(model_name)

    def transcribe(self, *, audio_bytes: bytes, filename: str) -> str:
        suffix = Path(filename).suffix or ".wav"

        with tempfile.NamedTemporaryFile(delete=True, suffix=suffix) as temp_file:
            temp_file.write(audio_bytes)
            temp_file.flush()

            result = self.model.transcribe(temp_file.name)

        return str(result.get("text", "")).strip()