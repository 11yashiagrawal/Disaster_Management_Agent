import json
from pathlib import Path

from infrastructure.nlp.hybrid_parser import HybridParser
from infrastructure.nlp.spacy_transcript_enricher import SpacyTranscriptEnricher


def main() -> None:
    sample_call_path = Path("src/contracts/examples/sample_call.json")

    with sample_call_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    parser = HybridParser(enricher=SpacyTranscriptEnricher())

    result = parser.parse(
        call_id=payload["call_id"],
        timestamp=payload["timestamp"],
        transcript=payload["transcript"],
    )

    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()