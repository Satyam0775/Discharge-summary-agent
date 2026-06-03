"""
Pydantic schemas for the Discharge Summary Agent.
All clinical data models used throughout the system.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────

class DocumentType(str, Enum):
    ADMISSION_NOTE = "admission_note"
    PROGRESS_NOTE = "progress_note"
    LAB_REPORT = "lab_report"
    MEDICATION_CHART = "medication_chart"
    NURSING_NOTE = "nursing_note"
    CONSULTATION_NOTE = "consultation_note"
    CT_REPORT = "ct_report"
    ECHO_REPORT = "echo_report"
    DISCHARGE_NOTE = "discharge_note"
    DRUG_CHART = "drug_chart"
    UNKNOWN = "unknown"


class ExtractionMethod(str, Enum):
    DIRECT = "direct"
    OCR = "ocr"
    NEEDS_OCR = "needs_ocr"
    FAILED = "failed"


class FlagSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


# ─────────────────────────────────────────────
# Document Models
# ─────────────────────────────────────────────

class ExtractedDocument(BaseModel):
    """A PDF document after text extraction."""

    filename: str
    filepath: str
    document_type: DocumentType = DocumentType.UNKNOWN
    raw_text: str = ""
    extraction_method: ExtractionMethod = ExtractionMethod.DIRECT
    extraction_success: bool = False
    page_count: int = 0
    is_scanned: bool = False
    error_message: Optional[str] = None


# ─────────────────────────────────────────────
# Clinical Data Models
# ─────────────────────────────────────────────

class PatientDemographics(BaseModel):
    """Patient demographic information."""

    patient_name: str = "NOT DOCUMENTED"
    patient_id: str = "NOT DOCUMENTED"
    date_of_birth: str = "NOT DOCUMENTED"
    age: str = "NOT DOCUMENTED"
    gender: str = "NOT DOCUMENTED"
    address: str = "NOT DOCUMENTED"
    contact_number: str = "NOT DOCUMENTED"
    next_of_kin: str = "NOT DOCUMENTED"
    review_required: bool = False


class Medication(BaseModel):
    """A single medication entry."""

    name: str
    dose: str = "NOT DOCUMENTED"
    frequency: str = "NOT DOCUMENTED"
    route: str = "NOT DOCUMENTED"
    indication: Optional[str] = None
    prescriber: Optional[str] = None
    start_date: Optional[str] = None
    stop_date: Optional[str] = None


class MedicationChange(BaseModel):
    """Records a change between admission and discharge medication."""

    medication_name: str
    change_type: str  # added | removed | modified | unchanged
    admission_details: Optional[Dict[str, Any]] = None
    discharge_details: Optional[Dict[str, Any]] = None
    documented_reason: Optional[str] = None
    flag_for_review: bool = False
    flag_reason: Optional[str] = None


class MedicationReconciliation(BaseModel):
    """Full medication reconciliation between admission and discharge."""

    added: List[MedicationChange] = Field(default_factory=list)
    removed: List[MedicationChange] = Field(default_factory=list)
    modified: List[MedicationChange] = Field(default_factory=list)
    unchanged: List[MedicationChange] = Field(default_factory=list)
    reason_missing: List[MedicationChange] = Field(default_factory=list)
    review_required: bool = False
    notes: str = ""


class Conflict(BaseModel):
    """A detected conflict between two documents."""

    field: str
    source_a: str
    value_a: str
    source_b: str
    value_b: str
    description: str
    clinician_review_required: bool = True
    conflict_id: str = ""


class ReviewFlag(BaseModel):
    """A flag indicating a field that requires clinician review."""

    field: str
    reason: str
    severity: FlagSeverity = FlagSeverity.WARNING
    current_value: Optional[str] = None
    flag_id: str = ""


class PendingResult(BaseModel):
    """A pending lab or imaging result."""

    result_name: str
    result_type: str = "unknown"  # lab | imaging | culture | pathology | other
    ordered_date: Optional[str] = None
    ordered_by: Optional[str] = None
    notes: str = ""


class StepTrace(BaseModel):
    """A single step trace in the agent execution."""

    step_number: int
    timestamp: str = ""
    reasoning: str
    tool: str
    input_summary: str
    output_summary: str
    decision: str
    success: bool = True
    error: Optional[str] = None
    duration_ms: Optional[int] = None


# ─────────────────────────────────────────────
# Discharge Summary Model
# ─────────────────────────────────────────────

class DischargeSummary(BaseModel):
    """Complete discharge summary draft."""

    # Safety notice - always present
    DRAFT_NOTICE: str = (
        "⚠️  THIS IS AN AI-GENERATED DRAFT FOR CLINICIAN REVIEW ONLY. "
        "IT MUST NOT BE USED FOR CLINICAL PURPOSES WITHOUT VERIFICATION BY A QUALIFIED CLINICIAN. ⚠️"
    )

    # Patient identification
    patient_demographics: PatientDemographics = Field(
        default_factory=PatientDemographics
    )

    # Dates
    admission_date: str = "NOT DOCUMENTED"
    discharge_date: str = "NOT DOCUMENTED"
    length_of_stay: str = "NOT DOCUMENTED"

    # Diagnoses
    principal_diagnosis: str = "NOT DOCUMENTED"
    secondary_diagnoses: List[str] = Field(default_factory=list)

    # Clinical narrative
    hospital_course: str = "NOT DOCUMENTED"
    discharge_condition: str = "NOT DOCUMENTED"

    # Procedures
    procedures: List[str] = Field(default_factory=list)

    # Medications
    admission_medications: List[Dict[str, Any]] = Field(default_factory=list)
    discharge_medications: List[Dict[str, Any]] = Field(default_factory=list)
    medication_reconciliation: MedicationReconciliation = Field(
        default_factory=MedicationReconciliation
    )

    # Allergies
    allergies: str = "NOT DOCUMENTED"

    # Follow-up
    followup_instructions: str = "NOT DOCUMENTED"
    followup_appointments: List[str] = Field(default_factory=list)

    # Pending items
    pending_results: List[PendingResult] = Field(default_factory=list)

    # Quality indicators
    conflicts_detected: List[Conflict] = Field(default_factory=list)
    review_flags: List[ReviewFlag] = Field(default_factory=list)
    missing_fields: List[str] = Field(default_factory=list)

    # Metadata
    generated_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )
    agent_steps_taken: int = 0
    source_documents: List[str] = Field(default_factory=list)
    is_draft: bool = True


# ─────────────────────────────────────────────
# API Input / Output Models
# ─────────────────────────────────────────────

class AgentRequest(BaseModel):
    """Input to the discharge summary agent."""

    patient_folder_path: str = Field(
        ...,
        description="Absolute or relative path to the patient's folder containing PDF files"
    )
    patient_id: Optional[str] = Field(
        default=None,
        description="Optional patient ID. Auto-detected from documents if not provided."
    )
    max_steps: int = Field(
        default=25,
        ge=5,
        le=50,
        description="Maximum agent iterations (safety cap)"
    )


class AgentResponse(BaseModel):
    """Full response from the discharge summary agent."""

    success: bool
    patient_id: str
    discharge_summary: Optional[DischargeSummary] = None
    step_traces: List[StepTrace] = Field(default_factory=list)
    total_steps: int = 0
    errors: List[str] = Field(default_factory=list)
    output_files: Dict[str, str] = Field(default_factory=dict)
    processing_time_seconds: float = 0.0
