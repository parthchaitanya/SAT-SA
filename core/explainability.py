import os
from typing import Dict, Any, Optional, List
import yaml
import ollama

from core.schema import OllamaRationale, PositiveGapFinding, NegativeGapFinding, SupervisoryScore


def get_ollama_config() -> Dict[str, Any]:
    """Load Ollama config from rules.yaml or defaults."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, "config", "rules.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            return cfg.get("ollama_config", {
                "default_model": "llama3.2:latest",
                "temperature": 0.2,
                "timeout_seconds": 25,
            })
    return {
        "default_model": "llama3.2:latest",
        "temperature": 0.2,
        "timeout_seconds": 25,
    }


class OllamaExplainabilityEngine:
    """
    Offline Explainability Engine for NCIIPC Supervisory Review.
    Uses local Ollama LLM (`ollama.chat()`) with an air-gapped fallback synthesizer.
    """

    def __init__(self, model_name: Optional[str] = None):
        cfg = get_ollama_config()
        self.model_name = model_name or cfg.get("default_model", "llama3.2:latest")
        self.temperature = cfg.get("temperature", 0.2)

    def is_ollama_available(self) -> bool:
        """Check if local Ollama daemon is reachable and responding."""
        try:
            ollama.list()
            return True
        except Exception:
            return False

    def list_local_models(self) -> List[str]:
        """List models available in the local Ollama instance."""
        try:
            res = ollama.list()
            # res can be a list of Model objects or dict
            models = []
            if hasattr(res, 'models'):
                for m in res.models:
                    models.append(m.model if hasattr(m, 'model') else str(m))
            elif isinstance(res, dict) and 'models' in res:
                for m in res['models']:
                    name = m.get('name') or m.get('model')
                    if name:
                        models.append(name)
            return models if models else [self.model_name]
        except Exception:
            return [self.model_name]

    def generate_case_rationale(
        self,
        case_record: Dict[str, Any],
        positive_findings: List[PositiveGapFinding],
        force_fallback: bool = False
    ) -> OllamaRationale:
        """
        Generates a concise supervisory audit rationale for a specific flagged case.
        """
        case_id = str(case_record.get("case_id", "N/A"))
        alert_id = str(case_record.get("alert_id", "N/A"))
        analyst_id = str(case_record.get("analyst_id", "N/A"))
        entity_id = str(case_record.get("entity_id", "N/A"))
        severity = str(case_record.get("severity", "N/A")).upper()
        alert_type = str(case_record.get("alert_type", "N/A"))
        action_taken = str(case_record.get("action_taken", "N/A"))
        notes = str(case_record.get("investigation_notes", "N/A"))
        time_to_close = str(case_record.get("time_to_close_minutes", "N/A"))
        escalation_status = str(case_record.get("escalation_status", "N/A"))

        triggered_rules = [f"{p.rule_id} ({p.rule_name})" for p in positive_findings if p.case_id == case_id]
        evidence_snippets = [p.evidence_detail for p in positive_findings if p.case_id == case_id]

        system_prompt = (
            "You are an expert NCIIPC SOC Audit Supervisor reviewing human analyst case-management records. "
            "Your job is to provide concise, factual, executive-level audit briefings explaining why a case was flagged. "
            "Focus purely on human analyst triage quality, compliance, and procedural gaps. Keep your tone objective and rigorous."
        )

        user_prompt = f"""
AUDIT CASE DETAILS:
- Case ID: {case_id} (Alert ID: {alert_id})
- Critical Sector Entity: {entity_id}
- Analyst ID: {analyst_id}
- Severity: {severity}
- Alert Category: {alert_type}
- Escalation Status: {escalation_status}
- Time to Close: {time_to_close} minutes
- Action Taken: {action_taken}
- Analyst Investigation Notes: "{notes}"
- Triggered Supervisory Rules: {', '.join(triggered_rules) if triggered_rules else 'General Review'}
- Evidence Points: {' | '.join(evidence_snippets)}

TASK:
Provide a structured supervisory brief in 3 short bulleted sections:
1. Behavioral Anomaly Summary (What went wrong in the analyst's workflow)
2. Forensic Evidence Points (Exact metrics/quotes proving the gap)
3. Supervisor Action Items (Specific instructions for the supervisor)
"""

        if not force_fallback:
            try:
                response = ollama.chat(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    options={"temperature": self.temperature}
                )
                text = response.get("message", {}).get("content", "")
                if text:
                    return self._parse_llm_response(text, target_id=case_id, target_type="CASE", generated_by=f"ollama:{self.model_name}")
            except Exception:
                # Fallback to local deterministic generator
                pass

        # Deterministic Air-Gapped Fallback
        return self._generate_case_fallback_rationale(case_record, positive_findings)

    def generate_analyst_briefing(
        self,
        score: SupervisoryScore,
        positive_findings: List[PositiveGapFinding],
        negative_findings: List[NegativeGapFinding],
        force_fallback: bool = False
    ) -> OllamaRationale:
        """
        Generates comprehensive supervisory evaluation for an analyst based on composite findings.
        """
        analyst_id = score.target_id
        pos_list = [p for p in positive_findings if p.analyst_id == analyst_id]
        neg_list = [n for n in negative_findings if n.target_type == "ANALYST" and n.target_id == analyst_id]

        pos_summary = ", ".join(set(p.rule_name for p in pos_list)) or "None"
        neg_summary = ", ".join(set(n.rule_name for n in neg_list)) or "None"

        user_prompt = f"""
ANALYST SUPERVISORY AUDIT PROFILE:
- Analyst: {analyst_id}
- Composite Supervisory Risk Score: {score.composite_risk_score}/100 ({score.supervisory_tier})
- Total Cases Handled: {score.total_cases_handled}
- Positive Space Gaps Recorded: {score.positive_gap_count} ({pos_summary})
- Negative Space Anomalies: {score.negative_gap_count} ({neg_summary})
- Unescalated Critical/High Cases: {score.unescalated_critical_count}
- Missing Evidence Attachments: {score.missing_evidence_count}
- Average Time-to-Close: {score.avg_close_time_minutes} min
- Primary Flags: {', '.join(score.primary_concerns)}

TASK:
Generate a concise, 3-part NCIIPC supervisory appraisal:
1. Executive Risk Summary
2. Core Deficiencies & Evidence
3. Recommended Supervisory Action (e.g. retraining, 100% case peer review, SOP audit)
"""

        if not force_fallback:
            try:
                response = ollama.chat(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a Chief SOC Audit Officer for NCIIPC. Keep output professional and concise."},
                        {"role": "user", "content": user_prompt},
                    ],
                    options={"temperature": self.temperature}
                )
                text = response.get("message", {}).get("content", "")
                if text:
                    return self._parse_llm_response(text, target_id=analyst_id, target_type="ANALYST", generated_by=f"ollama:{self.model_name}")
            except Exception:
                pass

        return self._generate_analyst_fallback_rationale(score, pos_list, neg_list)

    def generate_entity_briefing(
        self,
        score: SupervisoryScore,
        negative_findings: List[NegativeGapFinding],
        force_fallback: bool = False
    ) -> OllamaRationale:
        """
        Generates supervisory assessment for a Critical Sector Entity regarding telemetry blindspots and blackout periods.
        """
        entity_id = score.target_id
        neg_list = [n for n in negative_findings if n.target_type == "ENTITY" and n.target_id == entity_id]

        blindspots = [n for n in neg_list if n.rule_id == "NEG_01"]
        silence = [n for n in neg_list if n.rule_id == "NEG_03"]

        user_prompt = f"""
CRITICAL SECTOR ENTITY AUDIT:
- Entity: {entity_id}
- Supervisory Risk Score: {score.composite_risk_score}/100 ({score.supervisory_tier})
- Telemetry Blindspots: {len(blindspots)} categories ({', '.join(n.affected_alert_type for n in blindspots if n.affected_alert_type)})
- Silence Windows: {len(silence)} detected
- Primary Concerns: {', '.join(score.primary_concerns)}

TASK:
Generate a concise, 3-part NCIIPC entity telemetry briefing:
1. Entity Threat Exposure Summary
2. Missing Telemetry Evidence
3. Sector Coordination Directive
"""

        if not force_fallback:
            try:
                response = ollama.chat(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a Critical Infrastructure Cyber Assessment Supervisor for NCIIPC."},
                        {"role": "user", "content": user_prompt},
                    ],
                    options={"temperature": self.temperature}
                )
                text = response.get("message", {}).get("content", "")
                if text:
                    return self._parse_llm_response(text, target_id=entity_id, target_type="ENTITY", generated_by=f"ollama:{self.model_name}")
            except Exception:
                pass

        return self._generate_entity_fallback_rationale(score, neg_list)

    def _parse_llm_response(self, text: str, target_id: str, target_type: str, generated_by: str) -> OllamaRationale:
        """Parses LLM markdown response into structured OllamaRationale model."""
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        summary = ""
        evidence_points = []
        action_items = []
        current_section = "summary"

        for line in lines:
            lower = line.lower()
            if any(k in lower for k in ["summary", "overview", "anomaly summary", "risk summary", "1."]):
                current_section = "summary"
            elif any(k in lower for k in ["evidence", "deficiencies", "points", "2."]):
                current_section = "evidence"
            elif any(k in lower for k in ["action", "recommendation", "directive", "supervisor", "3."]):
                current_section = "action"

            if line.startswith("-") or line.startswith("*") or line.startswith("•") or (line[0].isdigit() and line[1] in [".", ")"]):
                cleaned = line.lstrip("-*• 0123456789.)").strip()
                if current_section == "evidence":
                    evidence_points.append(cleaned)
                elif current_section == "action":
                    action_items.append(cleaned)
                else:
                    if not summary:
                        summary = cleaned
                    else:
                        evidence_points.append(cleaned)
            else:
                if current_section == "summary" and not summary:
                    summary = line

        if not summary:
            summary = lines[0] if lines else f"Supervisory analysis for {target_id}"
        if not evidence_points:
            evidence_points = ["Flagged based on mathematical threshold deviation against cohort baseline."]
        if not action_items:
            action_items = ["Initiate supervisory interview and sample forensic records."]

        return OllamaRationale(
            target_id=target_id,
            target_type=target_type,  # type: ignore
            summary=summary,
            evidence_points=evidence_points[:4],
            supervisory_action_items=action_items[:4],
            generated_by=generated_by
        )

    def _generate_case_fallback_rationale(
        self,
        case_record: Dict[str, Any],
        positive_findings: List[PositiveGapFinding]
    ) -> OllamaRationale:
        case_id = str(case_record.get("case_id"))
        analyst_id = str(case_record.get("analyst_id"))
        severity = str(case_record.get("severity", "")).upper()
        alert_type = str(case_record.get("alert_type"))
        time_to_close = float(case_record.get("time_to_close_minutes", 0))

        case_pos = [p for p in positive_findings if p.case_id == case_id]
        
        summary = (
            f"Case {case_id} handled by Analyst {analyst_id} triggered {len(case_pos)} positive space audit rules "
            f"during the triage of a {severity} severity alert ({alert_type})."
        )
        
        evidence = []
        for p in case_pos:
            evidence.append(f"[{p.rule_id}] {p.evidence_detail}")

        if not evidence:
            evidence.append(f"Case closed in {time_to_close:.1f} minutes with action: '{case_record.get('action_taken')}'.")

        action_items = [
            f"Cross-examine Analyst {analyst_id} regarding procedural compliance on case {case_id}.",
            f"Request retroactive CIRT peer review for {severity} alert {alert_type}.",
            "Verify whether host containment was verified before marking case as resolved."
        ]

        return OllamaRationale(
            target_id=case_id,
            target_type="CASE",
            summary=summary,
            evidence_points=evidence,
            supervisory_action_items=action_items,
            generated_by="deterministic_offline_synthesizer"
        )

    def _generate_analyst_fallback_rationale(
        self,
        score: SupervisoryScore,
        positive_findings: List[PositiveGapFinding],
        negative_findings: List[NegativeGapFinding]
    ) -> OllamaRationale:
        analyst_id = score.target_id
        summary = (
            f"Analyst {analyst_id} is classified under {score.supervisory_tier.replace('_', ' ')} "
            f"with a composite risk score of {score.composite_risk_score}/100 across {score.total_cases_handled} cases."
        )

        evidence = []
        if score.unescalated_critical_count > 0:
            evidence.append(f"Unescalated High/Critical Cases: {score.unescalated_critical_count} critical alerts closed locally without CIRT escalation.")
        if score.positive_gap_count > 0:
            evidence.append(f"Positive Space Violations: {score.positive_gap_count} recorded instances of minimal notes, templated text, or rushed closures.")
        if score.missing_evidence_count > 0:
            evidence.append(f"Forensic Evidence Deficit: {score.missing_evidence_count} closed cases lacked required forensic evidence artifacts.")
        if any(n.rule_id == "NEG_02" for n in negative_findings):
            evidence.append(f"Statistical Volume Outlier: Case throughput is statistically abnormal compared to cohort mean.")

        if not evidence:
            evidence.append("Analyst demonstrates compliant case triage with minimal procedural variance.")

        actions = [
            f"Schedule 1-on-1 supervisory performance review with {analyst_id}.",
            "Mandate 100% pre-closure supervisor sign-off on all high/critical severity tickets for the next 14 days.",
            "Assign refresher training module on standard SOC investigation documentation and CIRT escalation matrix."
        ]

        return OllamaRationale(
            target_id=analyst_id,
            target_type="ANALYST",
            summary=summary,
            evidence_points=evidence,
            supervisory_action_items=actions,
            generated_by="deterministic_offline_synthesizer"
        )

    def _generate_entity_fallback_rationale(
        self,
        score: SupervisoryScore,
        negative_findings: List[NegativeGapFinding]
    ) -> OllamaRationale:
        entity_id = score.target_id
        blindspots = [n for n in negative_findings if n.rule_id == "NEG_01"]
        silence = [n for n in negative_findings if n.rule_id == "NEG_03"]

        summary = (
            f"Critical Sector Entity {entity_id} carries a supervisory risk score of {score.composite_risk_score}/100 "
            f"due to significant negative-space telemetry anomalies and surveillance gaps."
        )

        evidence = []
        for b in blindspots:
            evidence.append(f"Telemetry Blindspot: Zero recorded events for '{b.affected_alert_type}' vs peer average of {b.baseline_value:.1f} alerts.")
        for s in silence:
            evidence.append(f"Silence Period: {s.evidence_detail}")

        if not evidence:
            evidence.append("Entity telemetry stream is active and closely aligned with sector peer baselines.")

        actions = [
            f"Issue an NCIIPC Telemetry Verification Notice to the CISO of {entity_id}.",
            "Verify Sysmon, EDR, and SCADA firewall log forwarding configurations on all perimeter sensors.",
            "Conduct simulated red-team telemetry ping to test alert pipeline integrity."
        ]

        return OllamaRationale(
            target_id=entity_id,
            target_type="ENTITY",
            summary=summary,
            evidence_points=evidence,
            supervisory_action_items=actions,
            generated_by="deterministic_offline_synthesizer"
        )
