from datetime import datetime
from typing import Literal, Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator


class CaseRecord(BaseModel):
    """
    Standard SOC case-management record schema as defined in SIH26157.
    Represents already-generated historical records of human analyst actions.
    """
    alert_id: str = Field(..., description="Unique alert identifier (e.g. ALT-2026-1042)")
    timestamp: datetime = Field(..., description="Timestamp when the alert was triggered")
    entity_id: str = Field(..., description="Critical Sector Entity identifier (e.g. CSE-POWER-01)")
    analyst_id: str = Field(..., description="Human SOC Analyst identifier (e.g. ANL-R-SHARMA)")
    severity: Literal['low', 'medium', 'high', 'critical'] = Field(..., description="Alert severity level")
    alert_type: str = Field(..., description="Category of the security alert")
    action_taken: str = Field(..., description="Resolution or containment action documented by analyst")
    investigation_notes: str = Field(..., description="Investigation reasoning recorded by analyst")
    escalation_status: Literal['escalated', 'not_escalated'] = Field(..., description="Escalation decision")
    evidence_completeness_flag: bool = Field(..., description="Whether required forensic artifacts/evidence were attached")
    closure_time: datetime = Field(..., description="Timestamp when the case was closed")
    case_id: str = Field(..., description="Associated case management ticket ID")
    time_to_close_minutes: float = Field(..., description="Total minutes elapsed from alert to case closure")

    @field_validator('time_to_close_minutes', mode='before')
    @classmethod
    def validate_time_to_close(cls, v: Any) -> float:
        try:
            return float(v)
        except (ValueError, TypeError):
            return 0.0

    @field_validator('evidence_completeness_flag', mode='before')
    @classmethod
    def validate_bool_flag(cls, v: Any) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v)
        if isinstance(v, str):
            return v.strip().lower() in ('true', '1', 'yes', 'complete', 't')
        return False


class PositiveGapFinding(BaseModel):
    """
    Represents a detected visible/recorded human error (Positive Space Gap).
    """
    rule_id: str = Field(..., description="Rule code (e.g. POS_01, POS_02)")
    rule_name: str = Field(..., description="Short descriptive name of the positive rule")
    case_id: str = Field(..., description="Case identifier")
    alert_id: str = Field(..., description="Alert identifier")
    analyst_id: str = Field(..., description="Analyst identifier")
    entity_id: str = Field(..., description="Entity identifier")
    severity: str = Field(..., description="Alert severity")
    alert_type: str = Field(..., description="Alert type")
    gap_severity: Literal['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'] = Field(..., description="Severity of the supervisory finding")
    evidence_detail: str = Field(..., description="Concrete audit evidence explaining why the rule triggered")
    investigation_notes_snippet: str = Field(..., description="Excerpt of the analyst's notes")
    time_to_close_minutes: float = Field(..., description="Recorded time to close in minutes")
    escalation_status: str = Field(..., description="Recorded escalation status")


class NegativeGapFinding(BaseModel):
    """
    Represents an unseen/missing indicator or statistical anomaly (Negative Space Gap).
    """
    rule_id: str = Field(..., description="Rule code (e.g. NEG_01, NEG_02)")
    rule_name: str = Field(..., description="Name of the negative space anomaly")
    target_type: Literal['ANALYST', 'ENTITY'] = Field(..., description="Subject of the anomaly")
    target_id: str = Field(..., description="Analyst ID or Entity ID")
    gap_severity: Literal['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'] = Field(..., description="Supervisory risk tier")
    metric_name: str = Field(..., description="Name of statistical metric")
    observed_value: float = Field(..., description="Observed metric value")
    baseline_value: float = Field(..., description="Expected peer or historical baseline")
    evidence_detail: str = Field(..., description="Detailed explanation of the missing or anomalous pattern")
    affected_alert_type: Optional[str] = Field(None, description="Specific alert category affected if applicable")
    time_window: Optional[str] = Field(None, description="Time window of the anomaly if applicable")


class SupervisoryScore(BaseModel):
    """
    Supervisory risk score calculation result for an analyst or entity.
    """
    target_id: str
    target_type: Literal['ANALYST', 'ENTITY']
    entity_sector: Optional[str] = None
    composite_risk_score: float = Field(..., ge=0.0, le=100.0, description="Composite score 0-100")
    supervisory_tier: Literal[
        'CRITICAL_SUPERVISORY_ATTENTION',
        'ELEVATED_RISK_REVIEW',
        'MODERATE_MONITORING',
        'NOMINAL_COMPLIANT'
    ]
    total_cases_handled: int
    positive_gap_count: int
    negative_gap_count: int
    unescalated_critical_count: int
    missing_evidence_count: int
    avg_close_time_minutes: float
    score_breakdown: Dict[str, float]
    primary_concerns: List[str]


class OllamaRationale(BaseModel):
    """
    Plain-language supervisory rationale generated by Ollama or fallback synthesizer.
    """
    target_id: str
    target_type: Literal['ANALYST', 'ENTITY', 'CASE']
    summary: str
    evidence_points: List[str]
    supervisory_action_items: List[str]
    generated_by: str = "ollama:llama3.2"
