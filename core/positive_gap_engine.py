import os
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from core.schema import PositiveGapFinding


def load_rules_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load threshold configuration from rules.yaml or return robust defaults."""
    if config_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, "config", "rules.yaml")

    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    
    # Default fallback config if file missing
    return {
        "positive_space_rules": {
            "min_note_words": 6,
            "min_note_chars": 25,
            "trivial_note_keywords": ["ok", "done", "fp", "false positive", "closed", "no issue", "resolved", "na", "checked"],
            "critical_severities": ["high", "critical"],
            "valid_escalation_status": "escalated",
            "note_similarity_threshold": 0.82,
            "min_cases_for_similarity": 3,
            "min_close_time_minutes": {"critical": 15, "high": 10, "medium": 5, "low": 2}
        }
    }


class PositiveGapEngine:
    """
    Detects visible, recorded human analyst errors (Positive Space Gaps)
    using deterministic, rule-based Pandas logic.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = load_rules_config()
        self.config = config.get("positive_space_rules", {})
        self.min_note_words = self.config.get("min_note_words", 6)
        self.min_note_chars = self.config.get("min_note_chars", 25)
        self.trivial_keywords = set(self.config.get("trivial_note_keywords", ["ok", "done", "fp", "closed"]))
        self.critical_severities = set(self.config.get("critical_severities", ["high", "critical"]))
        self.note_similarity_threshold = self.config.get("note_similarity_threshold", 0.82)
        self.min_close_time = self.config.get("min_close_time_minutes", {"critical": 15, "high": 10, "medium": 5, "low": 2})

    def detect_minimal_notes(self, df: pd.DataFrame) -> List[PositiveGapFinding]:
        """Rule POS_01: Alert acknowledged/closed with empty or minimal notes."""
        findings: List[PositiveGapFinding] = []
        for _, row in df.iterrows():
            notes = str(row.get("investigation_notes", "")).strip()
            words = notes.split()
            word_count = len(words)
            char_count = len(notes)
            lower_notes = notes.lower()

            is_empty = char_count == 0
            is_under_threshold = (word_count < self.min_note_words) or (char_count < self.min_note_chars)
            is_trivial_word = lower_notes in self.trivial_keywords or any(lower_notes == kw for kw in self.trivial_keywords)

            if is_empty or is_under_threshold or is_trivial_word:
                gap_sev = "HIGH" if row.get("severity") in ["high", "critical"] else "MEDIUM"
                evidence = (
                    f"Investigation notes contain only {word_count} words ({char_count} chars). "
                    f"Threshold requires at least {self.min_note_words} words ({self.min_note_chars} chars). "
                    f"Content: '{notes}'"
                )
                finding = PositiveGapFinding(
                    rule_id="POS_01",
                    rule_name="Minimal / Non-Informative Investigation Notes",
                    case_id=str(row.get("case_id")),
                    alert_id=str(row.get("alert_id")),
                    analyst_id=str(row.get("analyst_id")),
                    entity_id=str(row.get("entity_id")),
                    severity=str(row.get("severity")),
                    alert_type=str(row.get("alert_type")),
                    gap_severity=gap_sev,
                    evidence_detail=evidence,
                    investigation_notes_snippet=notes[:120],
                    time_to_close_minutes=float(row.get("time_to_close_minutes", 0)),
                    escalation_status=str(row.get("escalation_status")),
                )
                findings.append(finding)
        return findings

    def detect_unescalated_high_severity(self, df: pd.DataFrame) -> List[PositiveGapFinding]:
        """Rule POS_02: High or Critical severity alert closed without escalation."""
        findings: List[PositiveGapFinding] = []
        for _, row in df.iterrows():
            sev = str(row.get("severity", "")).lower()
            esc_status = str(row.get("escalation_status", "")).lower()

            if sev in self.critical_severities and esc_status != "escalated":
                gap_sev = "CRITICAL" if sev == "critical" else "HIGH"
                evidence = (
                    f"Alert is classified as {sev.upper()} severity ({row.get('alert_type')}), "
                    f"but escalation_status is '{row.get('escalation_status')}'. "
                    f"Action taken was: '{row.get('action_taken')}'. High/Critical alerts require mandatory escalation to L2/CIRT."
                )
                finding = PositiveGapFinding(
                    rule_id="POS_02",
                    rule_name="Unescalated High/Critical Security Case",
                    case_id=str(row.get("case_id")),
                    alert_id=str(row.get("alert_id")),
                    analyst_id=str(row.get("analyst_id")),
                    entity_id=str(row.get("entity_id")),
                    severity=sev,
                    alert_type=str(row.get("alert_type")),
                    gap_severity=gap_sev,
                    evidence_detail=evidence,
                    investigation_notes_snippet=str(row.get("investigation_notes", ""))[:120],
                    time_to_close_minutes=float(row.get("time_to_close_minutes", 0)),
                    escalation_status=str(row.get("escalation_status")),
                )
                findings.append(finding)
        return findings

    def detect_templated_notes(self, df: pd.DataFrame) -> List[PositiveGapFinding]:
        """Rule POS_03: Repetitive / templated near-duplicate notes across an analyst's cases."""
        findings: List[PositiveGapFinding] = []
        analysts = df["analyst_id"].unique()

        for analyst_id in analysts:
            anl_df = df[df["analyst_id"] == analyst_id].copy()
            if len(anl_df) < 3:
                continue

            # Check for exact duplicates or normalized near-duplicates
            note_counts = {}
            for idx, row in anl_df.iterrows():
                n_raw = str(row.get("investigation_notes", "")).strip()
                # Skip very short notes (handled by POS_01)
                if len(n_raw.split()) < self.min_note_words:
                    continue
                # Normalize whitespace and lower
                norm_note = " ".join(n_raw.lower().split())
                if norm_note not in note_counts:
                    note_counts[norm_note] = []
                note_counts[norm_note].append(row)

            # If the exact/normalized note appears 3 or more times for this analyst, flag them as templated
            for norm_note, rows in note_counts.items():
                if len(rows) >= 2:
                    for row in rows:
                        evidence = (
                            f"Analyst reused identical/templated investigation notes verbatim across {len(rows)} "
                            f"separate security cases. Boilerplate sample: \"{str(row.get('investigation_notes', ''))[:80]}...\" "
                            f"Demonstrates copy-pasting without individualized case analysis."
                        )
                        finding = PositiveGapFinding(
                            rule_id="POS_03",
                            rule_name="Templated / Boilerplate Investigation Notes",
                            case_id=str(row.get("case_id")),
                            alert_id=str(row.get("alert_id")),
                            analyst_id=str(row.get("analyst_id")),
                            entity_id=str(row.get("entity_id")),
                            severity=str(row.get("severity")),
                            alert_type=str(row.get("alert_type")),
                            gap_severity="MEDIUM",
                            evidence_detail=evidence,
                            investigation_notes_snippet=str(row.get("investigation_notes", ""))[:120],
                            time_to_close_minutes=float(row.get("time_to_close_minutes", 0)),
                            escalation_status=str(row.get("escalation_status")),
                        )
                        findings.append(finding)

        return findings

    def detect_rapid_closures(self, df: pd.DataFrame) -> List[PositiveGapFinding]:
        """Rule POS_04: Time to close below realistic minimum threshold for that severity."""
        findings: List[PositiveGapFinding] = []
        for _, row in df.iterrows():
            sev = str(row.get("severity", "")).lower()
            ttc = float(row.get("time_to_close_minutes", 0))
            min_expected = self.min_close_time.get(sev, 2)

            if ttc < min_expected:
                gap_sev = "HIGH" if sev in ["high", "critical"] else "LOW"
                evidence = (
                    f"Case was closed in {ttc:.1f} minutes, which is below the minimum realistic triage "
                    f"threshold of {min_expected} minutes for a {sev.upper()} severity alert ({row.get('alert_type')}). "
                    f"Indicates rushed closure or superficial triage."
                )
                finding = PositiveGapFinding(
                    rule_id="POS_04",
                    rule_name="Premature / Rushed Case Closure",
                    case_id=str(row.get("case_id")),
                    alert_id=str(row.get("alert_id")),
                    analyst_id=str(row.get("analyst_id")),
                    entity_id=str(row.get("entity_id")),
                    severity=sev,
                    alert_type=str(row.get("alert_type")),
                    gap_severity=gap_sev,
                    evidence_detail=evidence,
                    investigation_notes_snippet=str(row.get("investigation_notes", ""))[:120],
                    time_to_close_minutes=ttc,
                    escalation_status=str(row.get("escalation_status")),
                )
                findings.append(finding)
        return findings

    def run_all_positive_rules(self, df: pd.DataFrame) -> List[PositiveGapFinding]:
        """Execute all positive space gap rules and return consolidated findings."""
        findings: List[PositiveGapFinding] = []
        findings.extend(self.detect_minimal_notes(df))
        findings.extend(self.detect_unescalated_high_severity(df))
        findings.extend(self.detect_templated_notes(df))
        findings.extend(self.detect_rapid_closures(df))
        return findings
