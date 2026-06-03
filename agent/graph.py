"""
agent/graph.py — LangGraph workflow for the Discharge Summary Agent.

Node layout:
  load_documents → planner → [tool nodes] → planner (loop) → generate_summary → save_outputs → END

The planner drives the loop.  Each tool node returns state updates and routes
back to the planner.  A conditional edge from the planner either selects the
next tool or terminates to summary generation.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

from langgraph.graph import END, StateGraph
from rich.console import Console

from agent.planner import AgentPlanner, build_trace
from agent.state import AgentState
from services.gemini_service import GeminiService
from tools.allergy_tool import AllergyExtractorTool
from tools.conflict_tool import ConflictDetectorTool
from tools.diagnosis_tool import DiagnosisExtractorTool
from tools.followup_tool import FollowupExtractorTool
from tools.medication_tool import MedicationExtractorTool
from tools.ocr_tool import OCRTool
from tools.pdf_loader import PDFLoaderTool
from tools.pending_tool import PendingResultDetectorTool
from tools.procedure_tool import ProcedureExtractorTool
from tools.reconciliation_tool import MedicationReconciliationTool
from tools.review_flag_tool import ReviewFlagTool
from tools.summary_tool import SummaryGeneratorTool

logger = logging.getLogger(__name__)
console = Console()


# ─────────────────────────────────────────────
# Build the graph
# ─────────────────────────────────────────────

def build_graph(output_dir: str = "outputs") -> Any:
    """
    Construct and compile the LangGraph StateGraph.

    Parameters
    ----------
    output_dir : where to save output files

    Returns
    -------
    Compiled LangGraph app
    """
    # ── Shared service instances ─────────────────────────────────────────
    gemini = GeminiService()
    planner = AgentPlanner(gemini)

    pdf_loader   = PDFLoaderTool()
    ocr_tool     = OCRTool()
    diag_tool    = DiagnosisExtractorTool(gemini)
    med_tool     = MedicationExtractorTool(gemini)
    allergy_tool = AllergyExtractorTool(gemini)
    proc_tool    = ProcedureExtractorTool(gemini)
    followup_tool = FollowupExtractorTool(gemini)
    pending_tool = PendingResultDetectorTool(gemini)
    conflict_tool = ConflictDetectorTool(gemini)
    recon_tool   = MedicationReconciliationTool(gemini)
    flag_tool    = ReviewFlagTool()
    summary_tool = SummaryGeneratorTool()

    # ─────────────────────────────────────────────
    # Node implementations
    # ─────────────────────────────────────────────

    def load_documents_node(state: AgentState) -> Dict[str, Any]:
        """Load all PDFs and run OCR on scanned pages."""
        console.rule("[bold blue]STEP: load_documents[/bold blue]")
        step = state.get("current_step", 0) + 1

        documents = pdf_loader.load_patient_folder(state["patient_folder"])
        documents = ocr_tool.process_scanned_documents(documents)

        all_text = "\n\n" + ("=" * 60) + "\n\n".join(
            f"[SOURCE: {d.get('filename', 'unknown')}]\n{d.get('raw_text', '')}"
            for d in documents
            if d.get("raw_text", "").strip()
        )

        errors: list[str] = [
            f"Failed to load {d['filename']}: {d['error']}"
            for d in documents
            if not d.get("success") and d.get("error")
        ]

        trace = build_trace(
            step=step,
            tool="PDFLoaderTool + OCRTool",
            reasoning="Load all patient documents before any extraction can begin.",
            input_summary=f"Folder: {state['patient_folder']}",
            output_summary=(
                f"{len(documents)} document(s) loaded; "
                f"{sum(1 for d in documents if d.get('is_scanned'))} scanned; "
                f"{len(all_text)} total chars extracted"
            ),
            decision="Proceed to planner to begin information extraction.",
            success=bool(documents),
            error="; ".join(errors) if errors else None,
        )

        return {
            "extracted_documents": documents,
            "all_text": all_text,
            "current_step": step,
            "step_traces": [trace],
            "errors": errors,
        }

    # ── Planner node ─────────────────────────────────────────────────────

    def planner_node(state: AgentState) -> Dict[str, Any]:
        """Ask Gemini what to do next."""
        step = state.get("current_step", 0) + 1
        console.rule(f"[bold blue]STEP {step}: planner[/bold blue]")

        decision = planner.plan(state)
        next_tool = decision.get("tool", "SummaryGeneratorTool")
        reasoning = decision.get("reasoning", "")
        is_complete = decision.get("is_complete", False)

        console.print(f"  [cyan]Planner → [bold]{next_tool}[/bold][/cyan]")
        console.print(f"  [dim]{reasoning}[/dim]")

        trace = build_trace(
            step=step,
            tool="PLANNER",
            reasoning=reasoning,
            input_summary=f"Completed: {state.get('completed_tools', [])}",
            output_summary=f"Selected tool: {next_tool}",
            decision=f"Execute {next_tool}" + (" — COMPLETE" if is_complete else ""),
        )

        return {
            "next_tool": next_tool,
            "is_complete": is_complete,
            "current_step": step,
            "step_traces": [trace],
        }

    # ── Tool nodes ────────────────────────────────────────────────────────

    def diagnosis_node(state: AgentState) -> Dict[str, Any]:
        console.rule("[bold green]TOOL: DiagnosisExtractorTool[/bold green]")
        step = state.get("current_step", 0) + 1
        text = state.get("all_text", "")
        result = diag_tool.extract(text)
        errors = [result["error"]] if result.get("error") else []

        demo = result.get("demographics", {})
        trace = build_trace(
            step=step,
            tool="DiagnosisExtractorTool",
            reasoning="Extract diagnoses, demographics, hospital course, and dates from all documents.",
            input_summary=f"Combined text ({len(text)} chars)",
            output_summary=(
                f"Principal dx: {result.get('principal_diagnosis', 'NOT DOCUMENTED')}; "
                f"Admission: {result.get('admission_date', 'NOT DOCUMENTED')}; "
                f"Demographics: {bool(demo)}"
            ),
            decision="Return to planner for next tool selection.",
            success=not bool(errors),
            error=errors[0] if errors else None,
        )

        completed = list(state.get("completed_tools", []))
        completed.append("DiagnosisExtractorTool")

        return {
            "demographics":         demo,
            "principal_diagnosis":  result.get("principal_diagnosis", "NOT DOCUMENTED"),
            "secondary_diagnoses":  result.get("secondary_diagnoses", []),
            "admission_date":       result.get("admission_date", "NOT DOCUMENTED"),
            "discharge_date":       result.get("discharge_date", "NOT DOCUMENTED"),
            "length_of_stay":       result.get("length_of_stay", "NOT DOCUMENTED"),
            "hospital_course":      result.get("hospital_course", "NOT DOCUMENTED"),
            "discharge_condition":  result.get("discharge_condition", "NOT DOCUMENTED"),
            "conflicts":            result.get("conflicts", []),
            "completed_tools":      completed,
            "current_step":         step,
            "step_traces":          [trace],
            "errors":               errors,
        }

    def medication_node(state: AgentState) -> Dict[str, Any]:
        console.rule("[bold green]TOOL: MedicationExtractorTool[/bold green]")
        step = state.get("current_step", 0) + 1
        text = state.get("all_text", "")
        result = med_tool.extract(text)
        errors = [result["error"]] if result.get("error") else []

        completed = list(state.get("completed_tools", []))
        completed.append("MedicationExtractorTool")

        trace = build_trace(
            step=step,
            tool="MedicationExtractorTool",
            reasoning="Extract admission and discharge medication lists.",
            input_summary=f"Combined text ({len(text)} chars)",
            output_summary=(
                f"Admission meds: {len(result.get('admission_medications', []))}; "
                f"Discharge meds: {len(result.get('discharge_medications', []))}"
            ),
            decision="Return to planner.",
            success=not bool(errors),
            error=errors[0] if errors else None,
        )

        return {
            "admission_medications": result.get("admission_medications", []),
            "discharge_medications": result.get("discharge_medications", []),
            "completed_tools":       completed,
            "current_step":          step,
            "step_traces":           [trace],
            "errors":                errors,
        }

    def allergy_node(state: AgentState) -> Dict[str, Any]:
        console.rule("[bold green]TOOL: AllergyExtractorTool[/bold green]")
        step = state.get("current_step", 0) + 1
        text = state.get("all_text", "")
        result = allergy_tool.extract(text)
        errors = [result["error"]] if result.get("error") else []

        completed = list(state.get("completed_tools", []))
        completed.append("AllergyExtractorTool")

        trace = build_trace(
            step=step,
            tool="AllergyExtractorTool",
            reasoning="Allergy info is a critical patient-safety requirement.",
            input_summary=f"Combined text ({len(text)} chars)",
            output_summary=f"Allergies: {result.get('allergies_raw', 'NOT DOCUMENTED')}",
            decision="Return to planner.",
            success=not bool(errors),
            error=errors[0] if errors else None,
        )

        return {
            "allergy_data":    result,
            "completed_tools": completed,
            "current_step":    step,
            "step_traces":     [trace],
            "errors":          errors,
        }

    def procedure_node(state: AgentState) -> Dict[str, Any]:
        console.rule("[bold green]TOOL: ProcedureExtractorTool[/bold green]")
        step = state.get("current_step", 0) + 1
        text = state.get("all_text", "")
        result = proc_tool.extract(text)
        errors = [result["error"]] if result.get("error") else []

        completed = list(state.get("completed_tools", []))
        completed.append("ProcedureExtractorTool")

        trace = build_trace(
            step=step,
            tool="ProcedureExtractorTool",
            reasoning="Extract any surgical or clinical procedures performed during admission.",
            input_summary=f"Combined text ({len(text)} chars)",
            output_summary=f"Procedures: {len(result.get('procedures', []))}",
            decision="Return to planner.",
            success=not bool(errors),
            error=errors[0] if errors else None,
        )

        return {
            "procedure_data":  result,
            "completed_tools": completed,
            "current_step":    step,
            "step_traces":     [trace],
            "errors":          errors,
        }

    def followup_node(state: AgentState) -> Dict[str, Any]:
        console.rule("[bold green]TOOL: FollowupExtractorTool[/bold green]")
        step = state.get("current_step", 0) + 1
        text = state.get("all_text", "")
        result = followup_tool.extract(text)
        errors = [result["error"]] if result.get("error") else []

        completed = list(state.get("completed_tools", []))
        completed.append("FollowupExtractorTool")

        trace = build_trace(
            step=step,
            tool="FollowupExtractorTool",
            reasoning="Follow-up instructions are required for a complete discharge summary.",
            input_summary=f"Combined text ({len(text)} chars)",
            output_summary=f"Follow-up: {result.get('followup_instructions', 'NOT DOCUMENTED')[:80]}",
            decision="Return to planner.",
            success=not bool(errors),
            error=errors[0] if errors else None,
        )

        return {
            "followup_data":   result,
            "completed_tools": completed,
            "current_step":    step,
            "step_traces":     [trace],
            "errors":          errors,
        }

    def pending_node(state: AgentState) -> Dict[str, Any]:
        console.rule("[bold green]TOOL: PendingResultDetectorTool[/bold green]")
        step = state.get("current_step", 0) + 1
        text = state.get("all_text", "")
        result = pending_tool.detect(text)
        errors = [result["error"]] if result.get("error") else []

        completed = list(state.get("completed_tools", []))
        completed.append("PendingResultDetectorTool")

        trace = build_trace(
            step=step,
            tool="PendingResultDetectorTool",
            reasoning="Identify any outstanding investigations before discharging patient.",
            input_summary=f"Combined text ({len(text)} chars)",
            output_summary=f"Pending: {len(result.get('pending_results', []))} result(s)",
            decision="Return to planner.",
            success=not bool(errors),
            error=errors[0] if errors else None,
        )

        return {
            "pending_data":    result,
            "completed_tools": completed,
            "current_step":    step,
            "step_traces":     [trace],
            "errors":          errors,
        }

    def conflict_node(state: AgentState) -> Dict[str, Any]:
        console.rule("[bold green]TOOL: ConflictDetectorTool[/bold green]")
        step = state.get("current_step", 0) + 1
        docs = state.get("extracted_documents", [])
        existing = state.get("conflicts", [])
        result = conflict_tool.detect(docs, existing_conflicts=existing)
        errors = [result["error"]] if result.get("error") else []

        completed = list(state.get("completed_tools", []))
        completed.append("ConflictDetectorTool")

        trace = build_trace(
            step=step,
            tool="ConflictDetectorTool",
            reasoning="Cross-reference documents to surface any factual contradictions.",
            input_summary=f"{len(docs)} documents",
            output_summary=f"Conflicts: {len(result.get('conflicts', []))}",
            decision="Return to planner.",
            success=not bool(errors),
            error=errors[0] if errors else None,
        )

        return {
            "conflict_data":   result,
            "conflicts":       result.get("conflicts", []),
            "completed_tools": completed,
            "current_step":    step,
            "step_traces":     [trace],
            "errors":          errors,
        }

    def reconciliation_node(state: AgentState) -> Dict[str, Any]:
        console.rule("[bold green]TOOL: MedicationReconciliationTool[/bold green]")
        step = state.get("current_step", 0) + 1
        adm = state.get("admission_medications", [])
        dis = state.get("discharge_medications", [])
        result = recon_tool.reconcile(adm, dis)
        errors = [result["error"]] if result.get("error") else []

        completed = list(state.get("completed_tools", []))
        completed.append("MedicationReconciliationTool")

        trace = build_trace(
            step=step,
            tool="MedicationReconciliationTool",
            reasoning="Compare admission vs discharge meds and flag unexplained changes.",
            input_summary=f"Admission: {len(adm)} meds, Discharge: {len(dis)} meds",
            output_summary=(
                f"Added: {len(result.get('added', []))}  "
                f"Removed: {len(result.get('removed', []))}  "
                f"Modified: {len(result.get('modified', []))}  "
                f"Flagged: {len(result.get('reason_missing', []))}"
            ),
            decision="Return to planner.",
            success=not bool(errors),
            error=errors[0] if errors else None,
        )

        return {
            "reconciliation":  result,
            "completed_tools": completed,
            "current_step":    step,
            "step_traces":     [trace],
            "errors":          errors,
        }

    def review_flag_node(state: AgentState) -> Dict[str, Any]:
        console.rule("[bold green]TOOL: ReviewFlagTool[/bold green]")
        step = state.get("current_step", 0) + 1
        flags = flag_tool.generate_flags(state)

        # Build missing fields list
        missing = [
            f["field"]
            for f in flags
            if not f["field"].startswith(("conflict_", "pending_", "medication_change_"))
        ]

        completed = list(state.get("completed_tools", []))
        completed.append("ReviewFlagTool")

        trace = build_trace(
            step=step,
            tool="ReviewFlagTool",
            reasoning="Generate all clinician review flags for missing and uncertain information.",
            input_summary="Current agent state",
            output_summary=f"{len(flags)} flag(s); {len(missing)} missing field(s)",
            decision="Return to planner — ready for summary generation.",
        )

        return {
            "review_flags":    flags,
            "missing_fields":  missing,
            "completed_tools": completed,
            "current_step":    step,
            "step_traces":     [trace],
        }

    def generate_summary_node(state: AgentState) -> Dict[str, Any]:
        console.rule("[bold blue]FINAL: SummaryGeneratorTool[/bold blue]")
        step = state.get("current_step", 0) + 1

        # Ensure review flags are generated even if ReviewFlagTool was skipped
        flags = state.get("review_flags", [])
        if not flags:
            flags = flag_tool.generate_flags(state)

        final_state = dict(state)
        final_state["review_flags"] = flags

        json_summary = summary_tool.build_json_summary(final_state)
        markdown = summary_tool.build_markdown(json_summary)

        pid = state.get("patient_id", "unknown")
        output_files = summary_tool.save_outputs(
            summary=json_summary,
            markdown=markdown,
            traces=state.get("step_traces", []),
            output_dir=output_dir,
            patient_id=pid,
        )

        completed = list(state.get("completed_tools", []))
        if "SummaryGeneratorTool" not in completed:
            completed.append("SummaryGeneratorTool")

        trace = build_trace(
            step=step,
            tool="SummaryGeneratorTool",
            reasoning="All extraction and analysis complete. Compile final discharge summary.",
            input_summary="Full agent state",
            output_summary=f"Generated JSON + Markdown; {len(flags)} flags; saved to {list(output_files.values())}",
            decision="END — summary complete.",
        )

        return {
            "final_summary":   json_summary,
            "output_files":    output_files,
            "is_complete":     True,
            "completed_tools": completed,
            "current_step":    step,
            "step_traces":     [trace],
            "review_flags":    flags,
        }

    # ─────────────────────────────────────────────
    # Routing logic
    # ─────────────────────────────────────────────

    _TOOL_NODE_MAP = {
        "DiagnosisExtractorTool":       "diagnosis",
        "MedicationExtractorTool":      "medication",
        "AllergyExtractorTool":         "allergy",
        "ProcedureExtractorTool":       "procedure",
        "FollowupExtractorTool":        "followup",
        "PendingResultDetectorTool":    "pending",
        "ConflictDetectorTool":         "conflict",
        "MedicationReconciliationTool": "reconciliation",
        "ReviewFlagTool":               "review_flags",
        "SummaryGeneratorTool":         "generate_summary",
    }

    def route_from_planner(state: AgentState) -> str:
        next_tool = state.get("next_tool", "SummaryGeneratorTool")
        node = _TOOL_NODE_MAP.get(next_tool, "generate_summary")
        return node

    # ─────────────────────────────────────────────
    # Assemble the graph
    # ─────────────────────────────────────────────

    graph = StateGraph(AgentState)

    graph.add_node("load_documents",   load_documents_node)
    graph.add_node("planner",          planner_node)
    graph.add_node("diagnosis",        diagnosis_node)
    graph.add_node("medication",       medication_node)
    graph.add_node("allergy",          allergy_node)
    graph.add_node("procedure",        procedure_node)
    graph.add_node("followup",         followup_node)
    graph.add_node("pending",          pending_node)
    graph.add_node("conflict",         conflict_node)
    graph.add_node("reconciliation",   reconciliation_node)
    graph.add_node("review_flags",     review_flag_node)
    graph.add_node("generate_summary", generate_summary_node)

    # Entry point
    graph.set_entry_point("load_documents")

    # load_documents → planner
    graph.add_edge("load_documents", "planner")

    # planner conditionally routes to tools
    graph.add_conditional_edges(
        "planner",
        route_from_planner,
        {
            "diagnosis":        "diagnosis",
            "medication":       "medication",
            "allergy":          "allergy",
            "procedure":        "procedure",
            "followup":         "followup",
            "pending":          "pending",
            "conflict":         "conflict",
            "reconciliation":   "reconciliation",
            "review_flags":     "review_flags",
            "generate_summary": "generate_summary",
        },
    )

    # All tool nodes loop back to planner
    for tool_node in [
        "diagnosis", "medication", "allergy", "procedure",
        "followup", "pending", "conflict", "reconciliation", "review_flags",
    ]:
        graph.add_edge(tool_node, "planner")

    # generate_summary → END
    graph.add_edge("generate_summary", END)

    return graph.compile()
