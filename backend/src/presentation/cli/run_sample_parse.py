import json
from pathlib import Path

from application.use_cases.parse_emergency_call import ParseEmergencyCallUseCase
from infrastructure.nlp.regex_parser import RegexParser


def main() -> None:
    sample_call_path = Path("src/contracts/examples/sample_call.json")

    with sample_call_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    parser = RegexParser()
    use_case = ParseEmergencyCallUseCase(parser=parser)

    result = use_case.execute(
        call_id=payload["call_id"],
        timestamp=payload["timestamp"],
        transcript=payload["transcript"],
    )

    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()