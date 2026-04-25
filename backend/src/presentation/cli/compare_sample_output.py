import json
import sys
from pathlib import Path

from application.use_cases.parse_emergency_call import ParseEmergencyCallUseCase
from infrastructure.nlp.regex_parser import RegexParser


def main() -> None:
    sample_call_path = Path("src/contracts/examples/sample_call.json")
    expected_output_path = Path("src/contracts/examples/sample_extraction_output.json")

    with sample_call_path.open("r", encoding="utf-8") as file:
        sample_call = json.load(file)

    with expected_output_path.open("r", encoding="utf-8") as file:
        expected_output = json.load(file)

    parser = RegexParser()
    use_case = ParseEmergencyCallUseCase(parser=parser)

    actual_output = use_case.execute(
        call_id=sample_call["call_id"],
        timestamp=sample_call["timestamp"],
        transcript=sample_call["transcript"],
    ).model_dump()

    if actual_output == expected_output:
        print("PASS: Actual parser output matches expected output.")
        return

    print("FAIL: Actual parser output does not match expected output.\n")

    print("Expected:")
    print(json.dumps(expected_output, indent=2))

    print("\nActual:")
    print(json.dumps(actual_output, indent=2))

    sys.exit(1)


if __name__ == "__main__":
    main()