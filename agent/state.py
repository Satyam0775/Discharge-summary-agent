"""
AgentState — the single shared state dictionary threaded through all LangGraph nodes.

All fields have safe defaults so any node can read them without a KeyError.
List fields that accumulate across steps use Annotated[List, operator.add].
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Optional

from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    """
    Complete agent state.

    Fields
    ------
    patient_folder  : path to the patient's document folder
    patient_id      : identifier for this patient run
    max_steps       : hard iteration cap

    pdf_files           : list of PDF file paths found
    extracted_documents : list of document dicts from PDFLoaderTool

    all_text            : concatenated text from all successfully extracted docs

    demographics        : dict of patient demographic fields
    admission_date      : string
    discharge_date      : string
    length_of_stay      : string
    principal_diagnosis : string
    secondary_diagnoses : list of strings
    hospital_course     : string
    discharge_condition : string

    admission_medications  : list of medication dicts
    discharge_medications  : list of medication dicts

    allergy_data    : dict from AllergyExtractorTool
    procedure_data  : dict from ProcedureExtractorTool
    followup_data   : dict from FollowupExtractorTool
    pending_data    : dict from PendingResultDetectorTool
    conflict_data   : dict from ConflictDetectorTool
    reconciliation  : dict from MedicationReconciliationTool

    current_step    : iteration counter
    completed_tools : tools already executed this run
    next_tool       : tool the planner chose for the next step
    is_complete     : whether the agent has finished

    review_flags    : list of flag dicts (accumulated)
    missing_fields  : list of field names that are NOT DOCUMENTED
    conflicts       : list of conflict dicts

    step_traces     : accumulated trace entries (append-only)
    errors          : accumulated error messages (append-only)

    final_summary   : assembled JSON summary dict
    output_files    : saved output file paths
    """

    # ── Input ─────────────────────────────────────────────────────────────
    patient_folder: str
    patient_id: str
    max_steps: int

    # ── Document layer ────────────────────────────────────────────────────
    pdf_files: List[str]
    extracted_documents: List[Dict[str, Any]]
    all_text: str

    # ── Extracted clinical data ───────────────────────────────────────────
    demographics: Dict[str, Any]
    admission_date: str
    discharge_date: str
    length_of_stay: str
    principal_diagnosis: str
    secondary_diagnoses: List[str]
    hospital_course: str
    discharge_condition: str

    admission_medications: List[Dict[str, Any]]
    discharge_medications: List[Dict[str, Any]]

    allergy_data: Dict[str, Any]
    procedure_data: Dict[str, Any]
    followup_data: Dict[str, Any]
    pending_data: Dict[str, Any]
    conflict_data: Dict[str, Any]
    reconciliation: Dict[str, Any]

    # ── Agent control ─────────────────────────────────────────────────────
    current_step: int
    completed_tools: List[str]
    next_tool: Optional[str]
    is_complete: bool

    # ── Quality ───────────────────────────────────────────────────────────
    review_flags: List[Dict[str, Any]]
    missing_fields: List[str]
    conflicts: List[Dict[str, Any]]

    # ── Observability (append-only lists) ─────────────────────────────────
    step_traces: Annotated[List[Dict[str, Any]], operator.add]
    errors: Annotated[List[str], operator.add]

    # ── Output ────────────────────────────────────────────────────────────
    final_summary: Optional[Dict[str, Any]]
    output_files: Dict[str, str]


def default_state(patient_folder: str, patient_id: str, max_steps: int) -> AgentState:
    """Return an initial AgentState with all fields set to safe defaults."""
    return AgentState(
        patient_folder=patient_folder,
        patient_id=patient_id,
        max_steps=max_steps,
        pdf_files=[],
        extracted_documents=[],
        all_text="",
        demographics={},
        admission_date="NOT DOCUMENTED",
        discharge_date="NOT DOCUMENTED",
        length_of_stay="NOT DOCUMENTED",
        principal_diagnosis="NOT DOCUMENTED",
        secondary_diagnoses=[],
        hospital_course="NOT DOCUMENTED",
        discharge_condition="NOT DOCUMENTED",
        admission_medications=[],
        discharge_medications=[],
        allergy_data={},
        procedure_data={},
        followup_data={},
        pending_data={},
        conflict_data={},
        reconciliation={},
        current_step=0,
        completed_tools=[],
        next_tool=None,
        is_complete=False,
        review_flags=[],
        missing_fields=[],
        conflicts=[],
        step_traces=[],
        errors=[],
        final_summary=None,
        output_files={},
    )
