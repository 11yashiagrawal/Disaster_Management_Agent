# Disaster_Management_Agent

## Team Members

- Yashi Agrawal
- Isha Tomar
- Pratyush Chouksey
- Kavya Katal

## Local Setup

1. Use Python 3.11 or 3.12 (Python 3.14 is not supported by spaCy build deps yet).
2. Create and activate a Python virtual environment.
3. Upgrade packaging tools:
	- `python3 -m pip install --upgrade pip setuptools wheel`
4. Install dependencies:
	- `pip install -r requirements.txt`
5. Create `.env` from `.env.example` and set `GROQ_API_KEY`.
6. Run the sample parser:
	- `python -m src.presentation.cli.run_sample_parse`