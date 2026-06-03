"""
FastAPI application — Discharge Summary Agent API.

Endpoints
---------
POST /generate-summary   Run the agent on a patient folder
GET  /health             API health check
GET  /docs               Swagger UI (auto-generated)
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from rich.console import Console
from rich.panel import Panel

from agent.graph import build_graph
from agent.state import default_state
from models.schemas import AgentRequest, AgentResponse, StepTrace

logger = logging.getLogger(__name__)
console = Console()

# ─────────────────────────────────────────────
# Application startup
# ─────────────────────────────────────────────

_compiled_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _compiled_graph
    console.print(
        Panel.fit(
            "[bold green]Discharge Summary Agent API[/bold green]\n"
            "Starting up — initialising LangGraph …",
            border_style="green",
        )
    )
    output_dir = os.getenv("OUTPUT_DIR", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "traces"), exist_ok=True)

    try:
        _compiled_graph = build_graph(output_dir=output_dir)
        console.print("[green]✓ LangGraph compiled successfully[/green]")
    except Exception as exc:
        console.print(f"[red]✗ LangGraph init failed: {exc}[/red]")
        logger.error("Graph init failed: %s", exc)

    yield

    console.print("[bold yellow]Shutting down …[/bold yellow]")


app = FastAPI(
    title="Discharge Summary Agent",
    description=(
        "Agentic AI system that reads hospital patient PDFs and generates a "
        "structured discharge summary draft for clinician review.\n\n"
        "⚠️ **All output is a DRAFT for clinician review only — never auto-finalized.**"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────

def _run_agent(patient_folder: str, patient_id: str, max_steps: int) -> Dict[str, Any]:
    """Execute the LangGraph agent synchronously."""
    if _compiled_graph is None:
        raise RuntimeError("LangGraph is not initialised — check server startup logs.")

    init_state = default_state(
        patient_folder=patient_folder,
        patient_id=patient_id,
        max_steps=max_steps,
    )

    final_state = _compiled_graph.invoke(init_state)
    return final_state


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/health", tags=["system"])
async def health_check():
    """API and LangGraph health check."""
    return {
        "status": "ok",
        "graph_ready": _compiled_graph is not None,
    }


@app.post(
    "/generate-summary",
    response_model=AgentResponse,
    tags=["agent"],
    summary="Generate discharge summary",
    description=(
        "Run the Agentic Discharge Summary Generator on a patient's folder of PDFs.\n\n"
        "Returns a structured draft discharge summary, step traces, and review flags.\n\n"
        "**All output is a DRAFT — must be reviewed by a clinician before clinical use.**"
    ),
)
async def generate_summary(request: AgentRequest):
    """Generate a discharge summary from a patient folder."""
    console.print(
        Panel.fit(
            f"[bold cyan]New request[/bold cyan]\n"
            f"Folder : {request.patient_folder_path}\n"
            f"Patient: {request.patient_id or 'auto-detect'}\n"
            f"Steps  : {request.max_steps}",
            border_style="cyan",
        )
    )

    # Validate folder exists
    if not os.path.exists(request.patient_folder_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient folder not found: {request.patient_folder_path}",
        )

    patient_id = request.patient_id or os.path.basename(
        request.patient_folder_path.rstrip("/\\")
    )

    start_time = time.time()

    try:
        final_state = _run_agent(
            patient_folder=request.patient_folder_path,
            patient_id=patient_id,
            max_steps=request.max_steps,
        )

        elapsed = round(time.time() - start_time, 2)

        # Build response
        traces = [
            StepTrace(
                step_number=t.get("step_number", 0),
                timestamp=t.get("timestamp", ""),
                reasoning=t.get("reasoning", ""),
                tool=t.get("tool", ""),
                input_summary=t.get("input_summary", ""),
                output_summary=t.get("output_summary", ""),
                decision=t.get("decision", ""),
                success=t.get("success", True),
                error=t.get("error"),
            )
            for t in final_state.get("step_traces", [])
        ]

        return AgentResponse(
            success=True,
            patient_id=patient_id,
            discharge_summary=final_state.get("final_summary"),
            step_traces=traces,
            total_steps=final_state.get("current_step", 0),
            errors=final_state.get("errors", []),
            output_files=final_state.get("output_files", {}),
            processing_time_seconds=elapsed,
        )

    except Exception as exc:
        elapsed = round(time.time() - start_time, 2)
        logger.error("Agent failed: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "patient_id": patient_id,
                "errors": [str(exc)],
                "processing_time_seconds": elapsed,
            },
        )
