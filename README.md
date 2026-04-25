# Disaster Management Agent Frontend

React-based operator dashboard for the Smart City Dynamic Dispatch Grid.

The frontend allows an operator or judge to submit messy emergency call transcripts and view the full AI-assisted response pipeline:

```text
Transcript
→ Incident Extraction
→ Triage Priority
→ Dispatch Assignment
→ Full JSON Output
```

## Overview

This frontend is designed for a hackathon demo and product-style presentation. It connects to the FastAPI backend and displays:

- incoming transcript input
- sample noisy emergency calls
- priority level and score
- extracted incident type
- detected location
- dispatch status
- assigned emergency resources
- ETA values
- triage reasoning
- recommended actions
- full backend JSON output

## Tech Stack

- React
- Vite
- CSS
- lucide-react icons
- FastAPI backend integration

## Project Structure

```text
frontend/
├── public/
├── src/
│   ├── assets/
│   ├── App.jsx
│   ├── App.css
│   ├── index.css
│   └── main.jsx
├── .env
├── package.json
├── vite.config.js
└── README.md
```

## Setup

From the frontend directory:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

Install icons if not already installed:

```bash
npm install lucide-react
```

## Environment Variables

Create a `.env` file in `frontend/`:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

This tells the frontend where the backend API is running.

## Run Frontend

```bash
npm run dev
```

Open the local Vite URL, usually:

```text
http://localhost:5173
```

## Backend Requirement

The backend must be running separately.

From the backend directory:

```bash
cd backend
npm run dev:api
```

Backend URL:

```text
http://127.0.0.1:8000
```

## Demo Flow

1. Start backend:

```bash
cd backend
npm run dev:api
```

2. Start frontend:

```bash
cd frontend
npm run dev
```

3. Open frontend in browser.

4. Select a sample transcript or type your own.

5. Click:

```text
Process Call
```

6. View:

- priority level
- incident type
- extracted location
- dispatch status
- assigned resources
- triage reasoning
- full JSON output

## Sample Transcripts

The UI includes sample noisy transcripts such as:

### Noisy Fire

```text
aaaa please hurry smok and fire near green park metro people trapped inside
```

### Road Accident

```text
bus and kar hit badli near sity hospital rood blockd two injrd please send amblance
```

### Gas Leak

```text
gass smel in apartmant people faintng cant breth please hurry
```

### Building Collapse

```text
bilding colapsd near river brij rubbl everywhere people trap send rescue now
```

These examples show the system handling ASR-like spelling mistakes and noisy emergency speech.

## Main UI Sections

### Sidebar

Displays:

- product name
- system status
- navigation-style modules

### Transcript Panel

Allows user to:

- paste emergency call transcript
- select sample cases
- submit transcript to backend

### Result Cards

Displays:

- priority level
- incident type
- location
- dispatch status

### Dispatch List

Shows assigned units:

- resource ID
- resource type
- ETA

### Insight Panel

Displays:

- triage reasoning
- recommended actions

### JSON Panel

Shows the complete backend response for technical evaluation.

## API Used

The frontend calls:

```http
POST /api/emergency-calls/process
```

Full URL:

```text
http://127.0.0.1:8000/api/emergency-calls/process
```

Request:

```json
{
  "call_id": "CALL_UI_001",
  "transcript": "aaaa please hurry smok and fire near green park metro people trapped inside"
}
```

## CORS Note

Backend must allow frontend origin.

For local hackathon development, backend currently allows broad CORS access.

Production should restrict origins to the deployed frontend domain.

## Build

```bash
npm run build
```

Preview production build:

```bash
npm run preview
```

## Hackathon Pitch

This frontend demonstrates the product in an operator-friendly way:

1. Paste or select noisy emergency transcript
2. Backend repairs and understands the transcript
3. Incident is extracted into structured JSON
4. Triage engine assigns priority
5. Dispatch engine assigns emergency units
6. Operator sees actions and ETA immediately

## Current Limitations

- No login/authentication
- No persistent incident history
- No real-time WebSocket updates yet
- No map visualization yet
- No role-based operator dashboard yet

## Future Improvements

- Live incident feed
- Map-based dispatch visualization
- Resource availability dashboard
- Incident history table
- Duplicate incident clustering UI
- Audio upload and transcription
- Operator notes and manual override controls

# Disaster Management Agent Backend

AI-assisted emergency call processing backend for a Smart City Dynamic Dispatch Grid.

This backend converts noisy emergency call transcripts into structured incident data, performs intelligent triage, and assigns emergency resources for dispatch.

## Overview

The backend pipeline handles realistic, messy ASR-style emergency transcripts such as:

```text
aaaa please hurry smok and fire near green park metro people trapped inside
```

and converts them into a structured response containing:

- corrected transcript understanding
- extracted incident details
- location hints
- severity classification
- casualty signals
- required emergency resources
- model-style triage score
- dispatch assignment with ETA

## Core Pipeline

```text
Raw Transcript
→ Transcript Correction
→ spaCy NLP Enrichment
→ Hybrid Parser
→ Structured Extraction JSON
→ Risk-Based Triage
→ Resource Dispatch
→ API Response
```

## Key Features

- Noisy transcript correction for ASR-style mistakes
- spaCy-based NLP enrichment using `en_core_web_sm`
- Hybrid extraction pipeline combining NLP signals and deterministic parsing
- Structured JSON output using Pydantic DTOs
- Model-style risk scoring for triage priority
- Safety override layer for life-threatening conditions
- Mock emergency resource registry
- Dispatch assignment with ETA
- FastAPI endpoint for frontend integration
- Unit and integration tests, including noisy transcript regression tests

## Tech Stack

- Python
- FastAPI
- spaCy
- Pydantic
- Pytest
- Uvicorn

## Project Structure

```text
backend/
├── src/
│   ├── application/
│   │   ├── dto/
│   │   ├── interfaces/
│   │   └── use_cases/
│   ├── core/
│   │   └── utils/
│   ├── domain/
│   │   ├── entities/
│   │   ├── enums/
│   │   └── services/
│   ├── infrastructure/
│   │   ├── dispatch/
│   │   └── nlp/
│   └── presentation/
│       ├── api/
│       └── cli/
├── tests/
│   ├── unit/
│   └── integration/
├── package.json
├── requirements.txt
└── README.md
```

## Important Modules

### Transcript Correction

```text
src/core/utils/transcript_correction.py
```

Repairs noisy ASR-style input.

Examples:

```text
smok → smoke
gass smel → gas smell
hospitl → hospital
injrd → injured
colapsd → collapsed
```

### spaCy NLP Enrichment

```text
src/infrastructure/nlp/spacy_transcript_enricher.py
```

Uses spaCy `en_core_web_sm` and domain-specific entity patterns to identify:

- incident clues
- severity clues
- location hints
- hazard signals
- casualty clues
- urgency clues

### Hybrid Parser

```text
src/infrastructure/nlp/hybrid_parser.py
```

Combines transcript repair, spaCy enrichment, and deterministic parsing to produce structured incident data.

### Triage Engine

```text
src/domain/services/triage_service.py
```

Combines:

- risk scoring
- incident severity
- hazards
- casualty signals
- safety overrides

to assign a priority level:

```text
P1 = Critical
P2 = High
P3 = Medium
P4 = Low / Needs clarification
```

### Dispatch Engine

```text
src/domain/services/dispatch_service.py
```

Assigns available emergency resources:

- ambulance
- fire truck
- police unit
- rescue team

and returns ETA values.

## Setup

From the backend directory:

```bash
cd backend
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Install the spaCy English model:

```bash
python -m spacy download en_core_web_sm
```

## Run Backend API

```bash
npm run dev:api
```

The API will run at:

```text
http://127.0.0.1:8000
```

Health check:

```text
http://127.0.0.1:8000/health
```

API docs:

```text
http://127.0.0.1:8000/docs
```

## API Endpoint

### Process Emergency Call

```http
POST /api/emergency-calls/process
```

Request:

```json
{
  "call_id": "CALL_001",
  "transcript": "aaaa please hurry smok and fire near green park metro people trapped inside"
}
```

Response shape:

```json
{
  "extraction": {
    "call_id": "CALL_001",
    "incident": {},
    "location": {},
    "casualties": {},
    "hazards": [],
    "resources_needed": {},
    "extraction_metadata": {},
    "deduplication": {}
  },
  "triage": {
    "priority_level": "P1",
    "priority_score": 92,
    "risk_probability": 0.91,
    "escalation_required": true,
    "triage_reason": "...",
    "recommended_actions": []
  },
  "dispatch": {
    "dispatch_status": "fully_assigned",
    "assigned_resources": [],
    "unfulfilled_resources": [],
    "dispatch_summary": "..."
  }
}
```

## Run Full CLI Demo

```bash
npm run demo:full
```

This runs the full backend pipeline from transcript to extraction, triage, and dispatch.

## Run Tests

Run all tests:

```bash
npm run test:all
```

Run noisy ASR transcript regression suite:

```bash
npm run test:noisy
```

Expected current result:

```text
66 passed
```

## Example Noisy Transcripts

```text
aaaa please hurry smok and fire near green park metro people trapped inside
```

```text
bus and kar hit badli near sity hospital rood blockd two injrd please send amblance
```

```text
gass smel in apartmant people faintng cant breth please hurry
```

```text
bilding colapsd near river brij rubbl everywhere people trap send rescue now
```

## Hackathon Value

This backend demonstrates an end-to-end AI-assisted civic emergency response system:

1. Accepts noisy emergency transcripts
2. Repairs ASR-like errors
3. Uses spaCy NLP for transcript enrichment
4. Extracts structured incident data
5. Scores risk and triage priority
6. Applies safety overrides
7. Dispatches available emergency resources
8. Returns complete operational JSON for frontend display

## Current Limitations

- Dispatch uses mock resources, not live city inventory
- ETA is based on mock location mapping
- No database persistence yet
- No real audio transcription module yet
- No duplicate incident clustering yet

## Future Improvements

- Add Whisper for audio-to-text transcription
- Add database for incident history
- Add duplicate detection across multiple calls
- Add graph-based routing for ETA
- Add live resource availability
- Add admin dashboard for command center operators
- Add multilingual transcript support


## Team Members

- Yashi Agrawal
- Isha Tomar
- Pratyush Chouksey
- Kavya Katal