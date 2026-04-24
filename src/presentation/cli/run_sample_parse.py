import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

from src.application.use_cases.parse_emergency_call import ParseEmergencyCallUseCase
from src.infrastructure.nlp.parser_factory import build_triage_parser


def main() -> None:
    sample_call_path = PROJECT_ROOT / "src" / "contracts" / "examples" / "sample_call.json"

    with sample_call_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    parser = build_triage_parser()
    use_case = ParseEmergencyCallUseCase(parser=parser)

    result = use_case.execute(
        call_id=payload["call_id"],
        timestamp=payload["timestamp"],
        transcript=payload["transcript"],
    )

    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()