"""
AgentPlanner — uses Gemini to decide which tool to execute next.

The planner analyses the current state, identifies what information is still
missing, and selects the most useful tool to call.  It also enforces the hard
iteration cap and decides when the extraction is complete.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from rich.console import Console
from services.gemini_service import GeminiService

logger = logging.getLogger(__name__)
console = Console()

# Ordered list of tools and their one-line description for the planner prompt
_TOOL_DESCRIPTIONS = {
    "DiagnosisExtractorTool":       "Extracts principal/secondary diagnoses, hospital course, admission/discharge dates, demographics, and discharge condition.",
    "MedicationExtractorTool":      "Extracts admission and discharge medication lists.",
    "AllergyExtractorTool":         "Extracts allergy and adverse reaction information.",
    "ProcedureExtractorTool":       "Extracts surgical and clinical procedures performed.",
    "FollowupExtractorTool":        "Extracts follow-up instructions and scheduled appointments.",
    "PendingResultDetectorTool":    "Detects pending lab, imaging, or culture results.",
    "ConflictDetectorTool":         "Detects factual conflicts across documents.",
    "MedicationReconciliationTool": "Compares admission vs discharge medications and flags unexplained changes.",
    "ReviewFlagTool":               "Generates final clinician review flags for missing/uncertain fields.",
    "SummaryGeneratorTool":         "Assembles the final discharge summary. Use ONLY when all other tools are done.",
}

_PLANNER_PROMPT = """\
You are an AI medical discharge summary agent planner.

Your job: decide which tool to call next to build a complete, safe discharge summary.

CURRENT STATE
=============
Step        : {step} / {max_steps}
Patient ID  : {patient_id}

Completed tools so far:
{completed_tools}

Information collected so far:
  - Demographics         : {demographics_status}
  - Principal Diagnosis  : {principal_diagnosis}
  - Secondary Diagnoses  : {sec_diag_count} found
  - Admission Date       : {admission_date}
  - Discharge Date       : {discharge_date}
  - Hospital Course      : {hospital_course_status}
  - Discharge Condition  : {discharge_condition}
  - Admission Meds       : {adm_med_count} found
  - Discharge Meds       : {dis_med_count} found
  - Allergies            : {allergies_status}
  - Procedures           : {procedures_status}
  - Follow-up            : {followup_status}
  - Pending Results      : {pending_status}
  - Conflicts Detected   : {conflict_status}
  - Medication Reconciliation: {recon_status}

AVAILABLE TOOLS
===============
{tool_descriptions}

RULES
=====
1. Run DiagnosisExtractorTool first if not yet done.
2. Run MedicationExtractorTool before MedicationReconciliationTool.
3. Run ConflictDetectorTool after at least DiagnosisExtractorTool and MedicationExtractorTool.
4. Run ReviewFlagTool only after all extraction tools are done.
5. Run SummaryGeneratorTool last — only when all others are complete.
6. Do NOT re-run a tool that is already in the completed list.
7. If step >= max_steps, always return SummaryGeneratorTool.

Respond ONLY with a JSON object:
{{
    "tool": "<exact tool name from available tools>",
    "reasoning": "<1-2 sentences explaining why this tool is needed now>",
    "is_complete": false
}}

Set "is_complete": true only when returning SummaryGeneratorTool.
"""


def _status(value: Any) -> str:
    """Return a short status string for a state value."""
    if value is None:
        return "NOT COLLECTED"
    if isinstance(value, str):
        return "✓ collected" if value not in ("NOT DOCUMENTED", "", "NOT COLLECTED") else "MISSING"
    if isinstance(value, dict):
        return "✓ collected" if value else "MISSING"
    if isinstance(value, list):
        return f"✓ {len(value)} items" if value else "MISSING (empty)"
    return str(value)


class AgentPlanner:
    """
    Gemini-driven planner.

    Given the current AgentState, returns:
        {"tool": "...", "reasoning": "...", "is_complete": bool}
    """

    def __init__(self, gemini: GeminiService) -> None:
        self.gemini = gemini

    def plan(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyse state and select the next tool.

        Falls back to a deterministic rule-based decision if Gemini is unavailable.
        """
        step = state.get("current_step", 0)
        max_steps = state.get("max_steps", 25)
        completed = state.get("completed_tools", [])

        # ── Hard cap ─────────────────────────────────────────────────────
        if step >= max_steps:
            logger.warning("Hard step cap reached (%d). Forcing summary generation.", max_steps)
            return {
                "tool": "SummaryGeneratorTool",
                "reasoning": f"Hard iteration cap ({max_steps} steps) reached. Finalising with available data.",
                "is_complete": True,
            }

        # ── All tools done ────────────────────────────────────────────────
        extraction_tools = [t for t in _TOOL_DESCRIPTIONS if t != "SummaryGeneratorTool"]
        if set(extraction_tools).issubset(set(completed)):
            if "SummaryGeneratorTool" not in completed:
                return {
                    "tool": "SummaryGeneratorTool",
                    "reasoning": "All extraction and analysis tools have completed. Building final summary.",
                    "is_complete": True,
                }

        # ── Build state summary for prompt ────────────────────────────────
        demo = state.get("demographics", {})
        tool_list = "\n".join(
            f"  - {name}: {desc}" for name, desc in _TOOL_DESCRIPTIONS.items()
            if name not in completed
        )
        completed_list = "\n".join(f"  - {t}" for t in completed) if completed else "  (none yet)"

        prompt = _PLANNER_PROMPT.format(
            step=step,
            max_steps=max_steps,
            patient_id=state.get("patient_id", "unknown"),
            completed_tools=completed_list,
            demographics_status=_status(demo),
            principal_diagnosis=state.get("principal_diagnosis", "NOT COLLECTED"),
            sec_diag_count=len(state.get("secondary_diagnoses", [])),
            admission_date=state.get("admission_date", "NOT COLLECTED"),
            discharge_date=state.get("discharge_date", "NOT COLLECTED"),
            hospital_course_status=_status(state.get("hospital_course")),
            discharge_condition=state.get("discharge_condition", "NOT COLLECTED"),
            adm_med_count=len(state.get("admission_medications", [])),
            dis_med_count=len(state.get("discharge_medications", [])),
            allergies_status=_status(state.get("allergy_data")),
            procedures_status=_status(state.get("procedure_data")),
            followup_status=_status(state.get("followup_data")),
            pending_status=_status(state.get("pending_data")),
            conflict_status=_status(state.get("conflict_data")),
            recon_status=_status(state.get("reconciliation")),
            tool_descriptions=tool_list if tool_list else "  All tools complete.",
        )

        # ── Ask Gemini ────────────────────────────────────────────────────
        try:
            decision = self.gemini.generate_json(
                prompt,
                fallback=self._rule_based_fallback(completed),
            )

            # Validate the chosen tool exists
            chosen = decision.get("tool", "")
            if chosen not in _TOOL_DESCRIPTIONS and chosen != "SummaryGeneratorTool":
                logger.warning("Planner returned unknown tool '%s' — falling back.", chosen)
                decision = self._rule_based_fallback(completed)

            return decision

        except Exception as exc:
            logger.error("Planner Gemini call failed: %s", exc)
            return self._rule_based_fallback(completed)

    def _rule_based_fallback(self, completed: List[str]) -> Dict[str, Any]:
        """Deterministic fallback when Gemini is unavailable."""
        ordered = [
            "DiagnosisExtractorTool",
            "MedicationExtractorTool",
            "AllergyExtractorTool",
            "ProcedureExtractorTool",
            "FollowupExtractorTool",
            "PendingResultDetectorTool",
            "ConflictDetectorTool",
            "MedicationReconciliationTool",
            "ReviewFlagTool",
            "SummaryGeneratorTool",
        ]
        for tool in ordered:
            if tool not in completed:
                return {
                    "tool": tool,
                    "reasoning": f"[fallback] {tool} not yet run.",
                    "is_complete": tool == "SummaryGeneratorTool",
                }
        return {
            "tool": "SummaryGeneratorTool",
            "reasoning": "[fallback] All tools done.",
            "is_complete": True,
        }


def build_trace(
    step: int,
    tool: str,
    reasoning: str,
    input_summary: str,
    output_summary: str,
    decision: str,
    success: bool = True,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """Helper to build a StepTrace dict."""
    return {
        "step_number": step,
        "timestamp": datetime.utcnow().isoformat(),
        "tool": tool,
        "reasoning": reasoning,
        "input_summary": input_summary,
        "output_summary": output_summary,
        "decision": decision,
        "success": success,
        "error": error,
    }
