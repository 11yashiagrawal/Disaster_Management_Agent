from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.presentation.api.routers.emergency_call_router import router as emergency_call_router


app = FastAPI(
    title="Disaster Management Agent API",
    description="Emergency transcript extraction, triage, and dispatch API.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


app.include_router(emergency_call_router)