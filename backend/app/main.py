"""
FastAPI Application Entrypoint for Stage 1 Baseline Multi-Agent System.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import settings

app = FastAPI(
    title="Adaptive Multi-Agent LLM System - Stage 1 Baseline",
    description=(
        "Fixed multi-agent pipeline baseline for complex task decomposition, "
        "parametric research, analysis, quality evaluation, and synthesis."
    ),
    version="1.0.0",
)

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(router)


@app.get("/")
async def root():
    return {
        "project": "Adaptive Multi-Agent LLM System",
        "stage": "Stage 1: Fixed Multi-Agent Baseline Prototype",
        "status": "online",
        "docs_url": "/docs",
    }
