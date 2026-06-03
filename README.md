# Agentic AI Discharge Summary Generator

A production-quality agentic system that reads raw hospital patient PDFs and generates a structured, clinically safe discharge summary draft for clinician review.

Built with **LangGraph**, **Gemini API (free tier)**, **FastAPI**, **PaddleOCR**, and **Python 3.11**.

---

## Table of Contents

1. [Architecture](#architecture)  
2. [Project Structure](#project-structure)  
3. [Setup & Installation](#setup--installation)  
4. [Running the Project](#running-the-project)  
5. [Agent Workflow](#agent-workflow)  
6. [OCR Workflow](#ocr-workflow)  
7. [No-Fabrication Guardrail](#no-fabrication-guardrail)  
8. [Medication Reconciliation](#medication-reconciliation)  
9. [Conflict Handling](#conflict-handling)  
10. [Failure Handling](#failure-handling)  
11. [Trace Generation](#trace-generation)  
12. [FastAPI Endpoint](#fastapi-endpoint)  
13. [Output Format](#output-format)  
14. [Limitations](#limitations)  
15. [What I Would Do With More Time](#what-i-would-do-with-more-time)

---

## Architecture

```
Patient Folder (PDFs)
        │
        ▼
┌──────────────────┐
│  PDFLoaderTool   │  PyMuPDF text extraction
│  + OCRTool       │  PaddleOCR for scanned pages
└────────┬─────────┘
         │ extracted_documents, all_text
         ▼
┌──────────────────┐
│    PLANNER       │  Gemini decides next tool
│  (LangGraph)     │  based on current state
└────────┬─────────┘
         │
    ┌────┴────────────────────────────────────┐
    │  Tool Nodes (each loops back to Planner) │
    │                                          │
    │  DiagnosisExtractorTool                  │
    │  MedicationExtractorTool                 │
    │  AllergyExtractorTool                    │
    │  ProcedureExtractorTool                  │
    │  FollowupExtractorTool                   │
    │  PendingResultDetectorTool               │
    │  ConflictDetectorTool                    │
    │  MedicationReconciliationTool            │
    │  ReviewFlagTool                          │
    └────┬────────────────────────────────────┘
         │ when all tools complete
         ▼
┌──────────────────┐
│ SummaryGenerator │  JSON + Markdown output
└────────┬─────────┘
         │
         ▼
   outputs/{patient_id}/
     discharge_summary.json
     discharge_summary.md
     traces/trace.txt
```

The system uses a **real agent loop** — the LangGraph planner uses Gemini to decide which tool to run next based on the current state of extracted information. It is not a fixed pipeline.

---

## Project Structure

```
discharge_summary_agent/
├── agent/
│   ├── __init__.py
│   ├── state.py          # AgentState TypedDict — shared state across all nodes
│   ├── planner.py        # Gemini-driven planner with rule-based fallback
│   └── graph.py          # LangGraph StateGraph — all 12 nodes + routing
├── app/
│   ├── __init__.py
│   └── api.py            # FastAPI application
├── data/
│   ├── create_sample_data.py   # Generates synthetic patient PDFs
│   ├── patient_001/            # (created after: python main.py sample-data)
│   └── patient_002/
├── models/
│   ├── __init__.py
│   └── schemas.py        # Pydantic models
├── outputs/
│   └── traces/           # Step traces written here
├── services/
│   ├── __init__.py
│   └── gemini_service.py # Gemini API wrapper with Tenacity retries
├── tests/
│   ├── __init__.py
│   └── test_tools.py     # Unit tests (no real API calls required)
├── tools/
│   ├── __init__.py
│   ├── allergy_tool.py
│   ├── conflict_tool.py
│   ├── diagnosis_tool.py
│   ├── followup_tool.py
│   ├── medication_tool.py
│   ├── ocr_tool.py
│   ├── pdf_loader.py
│   ├── pending_tool.py
│   ├── procedure_tool.py
│   ├── reconciliation_tool.py
│   ├── review_flag_tool.py
│   └── summary_tool.py
├── .env.example
├── main.py               # CLI entry point
├── README.md
└── requirements.txt
```

---

## Setup & Installation

### Prerequisites

- Python 3.11 (Windows, macOS, or Linux)
- A free Gemini API key: https://aistudio.google.com/app/apikey

### Install

```bash
# 1. Clone / extract the project
cd discharge_summary_agent

# 2. Create a virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
copy .env.example .env        # Windows
cp .env.example .env          # macOS/Linux

# Edit .env and set your Gemini API key:
#   GEMINI_API_KEY=your_key_here
```

### Windows note — pdf2image / Poppler

`pdf2image` requires Poppler on Windows.

1. Download from: https://github.com/oschwartz10612/poppler-windows/releases  
2. Extract and note the path (e.g., `C:\poppler\Library\bin`)  
3. Add to `.env`:  
   ```
   POPPLER_PATH=C:\poppler\Library\bin
   ```

On macOS: `brew install poppler`  
On Linux: `apt-get install poppler-utils`

---

## Running the Project

### 1. Generate sample patient data

```bash
python main.py sample-data
```

Creates two synthetic patient folders in `data/`:
- `patient_001` — clean T2DM case
- `patient_002` — complex cardiac case with conflicts, missing data, and pending results

### 2. Process a patient folder

```bash
python main.py run --patient-folder data/patient_001 --patient-id P001
```

Options:
```
--patient-folder   Path to the folder with patient PDFs (required)
--patient-id       Identifier for outputs (default: folder name)
--output-dir       Output directory (default: outputs/)
--max-steps        Hard agent step cap (default: 25)
```

### 3. Run the full demo (both patients)

```bash
python main.py demo
```

### 4. Start the API server

```bash
python main.py serve
# or with hot-reload for development:
python main.py serve --reload
```

### 5. Run tests

```bash
python main.py test
# or directly:
pytest tests/ -v
```

---

## Agent Workflow

The agent is a **LangGraph StateGraph** with a real re-planning loop:

```
load_documents
      │
      ▼
   planner ◄────────────────────────┐
      │                             │
      │ (conditional routing)       │
      ▼                             │
  [tool node]  ───────────────────► ┘
  (any of 9 tools)

  When all required tools complete:
      │
      ▼
 generate_summary
      │
      ▼
     END
```

**Planning logic:**

At each step the planner:

1. Builds a state summary (which tools have run, what data is missing)
2. Sends it to Gemini with a structured prompt asking for the next best tool
3. Falls back to a deterministic rule-based sequence if Gemini fails or returns an unexpected response
4. Enforces a hard step cap (`MAX_AGENT_STEPS`, default 25) — if reached, immediately routes to summary generation

This means the agent can **re-plan** — if a tool fails or returns partial data, the planner can choose to re-run it or route differently.

**Agent state** (`agent/state.py`) is a single `TypedDict` threaded through all nodes. Fields accumulate safely using `Annotated[List, operator.add]` for `step_traces` and `errors`.

---

## OCR Workflow

```
PDF file
   │
   ▼
PyMuPDF text extraction
   │
   ├─ Text found (≥80 chars) → use directly
   │
   └─ Text sparse / none (scanned PDF)
         │
         ▼
      pdf2image → page images
         │
         ▼
      PaddleOCR → extracted text
         │
         ▼
      Update document raw_text
```

- PaddleOCR is **lazy-initialised** on first use to avoid startup cost
- If PaddleOCR is unavailable (import error), the system continues with whatever text PyMuPDF could extract — it never crashes
- Each document carries an `is_scanned` flag used by subsequent tools

---

## No-Fabrication Guardrail

**This is the most critical safety requirement.**

Every tool prompt sent to Gemini includes explicit instructions:

```
CRITICAL RULE: If a piece of information is not present in the documents,
you MUST return the string "NOT DOCUMENTED" for that field.
NEVER guess, infer, or fabricate clinical information.
```

Every tool normalises its output: any field that comes back as `None`, empty string, or missing from the Gemini response is **replaced with `"NOT DOCUMENTED"`** before being stored in state.

When a required field is `"NOT DOCUMENTED"`, the `ReviewFlagTool` automatically generates a review flag with the appropriate severity:

| Field | Severity |
|-------|----------|
| Allergies | CRITICAL |
| Principal Diagnosis | WARNING |
| Discharge Medications | WARNING |
| Admission/Discharge Date | WARNING |
| Any other undocumented field | INFO |

Example output for a missing field:

```json
{
  "allergies": "NOT DOCUMENTED",
  "review_required": true,
  "review_flags": [
    {
      "severity": "CRITICAL",
      "field": "allergies",
      "message": "Allergy status not documented — cannot safely reconcile medications",
      "action": "Clinician must document allergy status before discharge"
    }
  ]
}
```

---

## Medication Reconciliation

The `MedicationReconciliationTool` compares the admission and discharge medication lists and produces:

```json
{
  "added": [
    {
      "name": "Sitagliptin",
      "dose": "100mg",
      "frequency": "OD",
      "route": "Oral",
      "reason": "Added for additional HbA1c reduction"
    }
  ],
  "removed": [],
  "modified": [
    {
      "name": "Metformin",
      "admission_dose": "500mg BD",
      "discharge_dose": "1000mg BD",
      "reason": "Increased for improved glycaemic control"
    }
  ],
  "unchanged": ["Atorvastatin", "Aspirin"],
  "reason_missing": [
    {
      "name": "Ramipril",
      "issue": "Added per cardiology but discharge status unclear",
      "action": "Clinician review required"
    }
  ]
}
```

Any medication change **without a documented reason** is placed in `reason_missing` and generates a `WARNING` review flag. The system never silently resolves undocumented medication changes.

---

## Conflict Handling

When two or more documents contain contradictory clinical information, the `ConflictDetectorTool`:

1. **Never picks one value automatically**
2. **Always flags the conflict** with both source documents and values
3. **Requires clinician review** — `clinician_review_required: true` on every conflict

Example:

```json
{
  "conflicts": [
    {
      "field": "discharge_diagnosis",
      "document_a": "admission_note.pdf",
      "value_a": "NSTEMI",
      "document_b": "cardiology_consultation.pdf",
      "value_b": "Acute decompensated heart failure secondary to AF",
      "severity": "HIGH",
      "clinician_review_required": true,
      "note": "Conflicting primary diagnoses across documents. Clinician must confirm."
    }
  ]
}
```

Both values are preserved in the summary. The conflict appears in the `conflicts_detected` section and generates a `WARNING` or `CRITICAL` review flag.

---

## Failure Handling

| Failure type | Behaviour |
|---|---|
| Missing PDF folder | Returns empty document list; agent continues; flags as WARNING |
| Corrupted / unreadable PDF | Document marked `success: false`; error logged; agent continues |
| Empty PDF (no text, no images) | Proceeds; fields default to `NOT DOCUMENTED` |
| OCR failure | Error caught; raw_text remains empty; NOT DOCUMENTED used |
| Gemini API timeout / rate limit | Tenacity retries 3 times with exponential backoff (2s, 4s, 8s) |
| Gemini returns invalid JSON | JSON fence stripping + fallback `{}` returned; tool defaults used |
| Tool exception | Caught; error logged to `state.errors`; agent continues to next tool |
| Agent exceeds MAX_AGENT_STEPS | Planner routes immediately to summary generation |

The agent **never crashes**. A partial summary with `NOT DOCUMENTED` fields and review flags is always preferable to an exception.

---

## Trace Generation

Every node writes a structured trace entry to `state.step_traces`. After run completion, traces are written to `outputs/{patient_id}/traces/trace.txt`.

Format:

```
════════════════════════════════════════
STEP 3
════════════════════════════════════════
Timestamp : 2024-01-10T14:23:05
Tool      : DiagnosisExtractorTool
Success   : True

REASONING
---------
Extract diagnoses, demographics, hospital course and dates
from all source documents.

INPUT
-----
Combined text (8,432 chars across 4 documents)

OUTPUT
------
Principal dx: Type 2 Diabetes Mellitus — poorly controlled (HbA1c 9.4%)
Admission: 10/01/2024
Demographics: present

DECISION
--------
Return to planner for next tool selection.
```

---

## FastAPI Endpoint

### Start the server

```bash
python main.py serve
```

### POST /generate-summary

```bash
curl -X POST http://localhost:8000/generate-summary \
  -H "Content-Type: application/json" \
  -d '{"patient_folder_path": "data/patient_001", "patient_id": "P001"}'
```

Request body:

```json
{
  "patient_folder_path": "data/patient_001",
  "patient_id": "P001",
  "max_steps": 25
}
```

Response:

```json
{
  "patient_id": "P001",
  "status": "success",
  "summary": { ... },
  "output_files": {
    "json":     "outputs/P001/discharge_summary.json",
    "markdown": "outputs/P001/discharge_summary.md",
    "trace":    "outputs/P001/traces/trace.txt"
  },
  "review_flags": [ ... ],
  "conflicts_detected": [ ... ],
  "pending_results": [ ... ],
  "errors": [],
  "steps_taken": 12
}
```

### GET /health

```bash
curl http://localhost:8000/health
```

---

## Output Format

### discharge_summary.json

Contains all 15 required sections:

```json
{
  "generated_at": "2024-01-15T16:30:00",
  "patient_id": "P001",
  "is_draft": true,
  "clinician_review_required": true,
  "patient_demographics": { ... },
  "admission_date": "10/01/2024",
  "discharge_date": "15/01/2024",
  "length_of_stay": "5 days",
  "principal_diagnosis": "Type 2 DM — poorly controlled",
  "secondary_diagnoses": ["Hypertension", "Hyperlipidaemia"],
  "hospital_course": "...",
  "procedures": { ... },
  "discharge_medications": [ ... ],
  "medication_reconciliation": { ... },
  "allergies": { ... },
  "followup_instructions": { ... },
  "pending_results": { ... },
  "discharge_condition": "Stable",
  "conflicts_detected": [ ... ],
  "review_flags": [ ... ]
}
```

### discharge_summary.md

Human-readable Markdown with all sections formatted for clinical review.

### traces/trace.txt

Step-by-step agent trace for observability and audit.

---

## Limitations

1. **Gemini context window** — Documents are truncated at ~28,000 characters before being sent to Gemini. Very long admission records may lose tail content.

2. **PaddleOCR accuracy** — OCR quality degrades on low-resolution scans, handwritten notes, or non-English text. All OCR output is explicitly marked as extracted text; errors are not correctable by the agent.

3. **Unverified patient-reported medications** — If a patient's medication list comes only from their own report (as in patient_002), the agent marks it as unverified and flags it for reconciliation rather than treating it as ground truth.

4. **No drug interaction checking** — The current implementation does not call an external drug interaction API. The `ReviewFlagTool` flags undocumented medication combinations but does not assess pharmacological risk.

5. **No real-time lab reference ranges** — Lab result interpretation is limited to text extracted from documents. The agent does not call external pathology reference databases.

6. **Single-user API** — The FastAPI server is single-threaded in the default configuration. Production deployment would require a task queue (e.g., Celery + Redis) for concurrent requests.

7. **Gemini free tier rate limits** — The free Gemini tier allows ~60 requests/minute. Processing documents that require many tool calls may hit rate limits; the Tenacity retry logic handles this gracefully with backoff.

---

## What I Would Do With More Time

1. **Structured extraction with schema enforcement** — Pass Pydantic schemas directly to Gemini's function-calling API to get typed, validated responses instead of parsing free-form JSON.

2. **Drug interaction mock tool** — Add a `DrugInteractionTool` that checks discharge medications against a local database (e.g., DrugBank open data) and flags combinations requiring clinician review.

3. **Document chunking** — Replace simple truncation with semantic chunking (LangChain `RecursiveCharacterTextSplitter`) to ensure no clinically relevant content is lost.

4. **Vector retrieval** — Index extracted documents in a local vector store (ChromaDB) so each tool can retrieve only the most relevant sections rather than sending all text to Gemini.

5. **Part 2 — Clinician feedback loop** — Implement a simulated reviewer that applies a consistent editing policy, enabling preference learning (DPO) over (draft, edited) pairs to reduce edit burden over time.

6. **Async API with task queue** — Use FastAPI background tasks + Celery for long-running agent jobs, with a `/status/{job_id}` polling endpoint.

7. **Structured logging** — Replace Rich console output with structured JSON logs (structlog) for production observability.

---

## Disclaimer

This system generates **draft summaries only**. All output must be reviewed and verified by a qualified clinician before any clinical use. The system is designed to assist, not to replace, clinical judgment.

All sample data is entirely synthetic. No real patient data was used.
