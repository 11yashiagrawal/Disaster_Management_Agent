# Disaster Management Agent - README

## Installation & Local Setup

## Prerequisites

-   Node.js (v18+ recommended)
-   Python (3.10+ recommended)
-   pip
-   Git
-   FFmpeg (recommended for audio processing)

## 1. Clone Repository

``` bash
git clone <your-repository-url>
cd Disaster_Management_Agent
```

## 2. Backend Setup

``` bash
cd backend
pip install -r requirements.txt
python -m spacy download en_core_web_sm
npm run dev:api
```

Backend: - http://127.0.0.1:8000 - http://127.0.0.1:8000/docs

## 3. Frontend Setup

``` bash
cd frontend
npm install
npm run dev
```

Frontend: - http://localhost:5173

## 4. Full Project Run

### Terminal 1

``` bash
cd backend
npm run dev:api
```

### Terminal 2

``` bash
cd frontend
npm install
npm run dev
```

## 5. Running Tests

``` bash
cd backend
npm run test:all
```

## 6. Audio Support

-   macOS: `brew install ffmpeg`
-   Ubuntu: `sudo apt install ffmpeg`
-   Windows: Install FFmpeg and add to PATH

## 7. Troubleshooting

-   Allow browser microphone permissions.
-   Install FFmpeg if audio upload fails.
-   Download spaCy model if missing.
-   Free occupied ports if servers fail to start.

## 8. Quick Start

``` bash
git clone <repo-url>
cd Disaster_Management_Agent

cd backend
pip install -r requirements.txt
python -m spacy download en_core_web_sm
npm run dev:api

# new terminal
cd frontend
npm install
npm run dev
```
