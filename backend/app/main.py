from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.schemas import AnalysisResponse, ChatRequest, ChatResponse, VehicleInput
from app.services.chat_service import answer_question
from app.services.model_service import ModelNotReadyError, analyze_vehicle, load_artifacts


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        load_artifacts()
    except ModelNotReadyError:
        # Let the API start so the user can train the model next.
        pass
    yield


app = FastAPI(
    title="CarSight AI API",
    version="1.0.0",
    description="Decision-support API for used car pricing with text-based damage interpretation.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    try:
        load_artifacts()
        return {"status": "ok", "model": "ready"}
    except ModelNotReadyError:
        return {"status": "ok", "model": "missing"}


@app.post("/api/analyze", response_model=AnalysisResponse)
def analyze(payload: VehicleInput) -> AnalysisResponse:
    return analyze_vehicle(payload)


@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    return answer_question(payload.question, payload.analysis)


@app.exception_handler(ModelNotReadyError)
def model_missing_handler(_, exc: ModelNotReadyError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})
