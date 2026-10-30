import os
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import yaml

from core.schema import SupervisoryScore, PositiveGapFinding, NegativeGapFinding


class SupervisoryRiskScoringEngine:
    """
    Computes multi-factor composite risk scores (0-100) for analysts and CSE entities,
    prioritizing where NCIIPC supervisors should focus their manual review time.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(base_dir, "config", "rules.yaml")
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
            else:
                config = {}
        
        weights = config.get("risk_scoring_weights", {})
        self.analyst_weights = weights.get("analyst_weights", {
            "positive_gap_ratio": 0.35,
            "negative_gap_ratio": 0.25,
            "unescalated_critical_ratio": 0.20,
            "missing_evidence_ratio": 0.10,
            "volume_outlier_penalty": 0.10,
        })
        self.entity_weights = weights.get("entity_weights", {
            "telemetry_blindspot_score": 0.40,
            "silence_window_penalty": 0.30,
            "overall_case_gap_rate": 0.20,
            "evidence_deficiency_rate": 0.10,
        })

    def _determine_tier(self, score: float) -> str:
        if score >= 70.0:
            return "CRITICAL_SUPERVISORY_ATTENTION"
        elif score >= 45.0:
            return "ELEVATED_RISK_REVIEW"
        elif score >= 20.0:
            return "MODERATE_MONITORING"
        else:
            return "NOMINAL_COMPLIANT"

    def score_analysts(
        self,
        df: pd.DataFrame,
        positive_findings: List[PositiveGapFinding],
        negative_findings: List[NegativeGapFinding]
    ) -> List[SupervisoryScore]:
        """
        Calculates composite supervisory risk score for each human SOC analyst.
        """
        results: List[SupervisoryScore] = []
        analysts = df["analyst_id"].unique()

        # Map findings by analyst
        pos_by_analyst: Dict[str, List[PositiveGapFinding]] = {a: [] for a in analysts}
        for pf in positive_findings:
            if pf.analyst_id in pos_by_analyst:
                pos_by_analyst[pf.analyst_id].append(pf)

        neg_by_analyst: Dict[str, List[NegativeGapFinding]] = {a: [] for a in analysts}
        for nf in negative_findings:
            if nf.target_type == "ANALYST" and nf.target_id in neg_by_analyst:
                neg_by_analyst[nf.target_id].append(nf)

        for analyst_id in analysts:
            anl_df = df[df["analyst_id"] == analyst_id]
            total_cases = len(anl_df)
            if total_cases == 0:
                continue

            # 1. Positive Gap Ratio (0 - 100)
            pos_list = pos_by_analyst.get(analyst_id, [])
            unique_flagged_cases = set(p.case_id for p in pos_list)
            pos_gap_ratio = min(1.0, len(unique_flagged_cases) / total_cases) * 100

            # 2. Negative Gap impact (0 - 100)
            neg_list = neg_by_analyst.get(analyst_id, [])
            neg_gap_impact = min(100.0, len(neg_list) * 45.0)

            # 3. Unescalated critical/high ratio (0 - 100)
            crit_high_df = anl_df[anl_df["severity"].isin(["high", "critical"])]
            crit_count = len(crit_high_df)
            unesc_count = len(crit_high_df[crit_high_df["escalation_status"] != "escalated"])
            unescalated_ratio = (unesc_count / crit_count * 100) if crit_count > 0 else 0.0

            # 4. Missing Evidence Ratio (0 - 100)
            missing_evidence_count = len(anl_df[~anl_df["evidence_completeness_flag"]])
            missing_evidence_ratio = (missing_evidence_count / total_cases) * 100

            # 5. Volume Outlier Penalty (0 - 100)
            has_vol_outlier = any(nf.rule_id == "NEG_02" for nf in neg_list)
            vol_penalty = 80.0 if has_vol_outlier else 0.0

            # Weighted Composite
            w = self.analyst_weights
            composite = (
                (pos_gap_ratio * w.get("positive_gap_ratio", 0.35)) +
                (neg_gap_impact * w.get("negative_gap_ratio", 0.25)) +
                (unescalated_ratio * w.get("unescalated_critical_ratio", 0.20)) +
                (missing_evidence_ratio * w.get("missing_evidence_ratio", 0.10)) +
                (vol_penalty * w.get("volume_outlier_penalty", 0.10))
            )
            composite = max(0.0, min(100.0, composite))

            # Primary concerns identification
            concerns: List[str] = []
            if any(p.rule_id == "POS_01" for p in pos_list):
                concerns.append("Minimal / Empty investigation notes")
            if any(p.rule_id == "POS_02" for p in pos_list):
                concerns.append("Critical/High alerts closed without escalation")
            if any(p.rule_id == "POS_03" for p in pos_list):
                concerns.append("Templated / boilerplate notes repetition")
            if any(p.rule_id == "POS_04" for p in pos_list):
                concerns.append("Abnormally rushed case closures (< SLA)")
            if has_vol_outlier:
                concerns.append("Statistical low volume outlier (disengagement)")
            if missing_evidence_ratio >= 30.0:
                concerns.append(f"High missing forensic evidence rate ({missing_evidence_ratio:.0f}%)")

            if not concerns:
                concerns.append("Compliant standard triage behavior")

            avg_time = float(anl_df["time_to_close_minutes"].mean())

            score_obj = SupervisoryScore(
                target_id=analyst_id,
                target_type="ANALYST",
                composite_risk_score=round(composite, 1),
                supervisory_tier=self._determine_tier(composite),
                total_cases_handled=total_cases,
                positive_gap_count=len(pos_list),
                negative_gap_count=len(neg_list),
                unescalated_critical_count=unesc_count,
                missing_evidence_count=missing_evidence_count,
                avg_close_time_minutes=round(avg_time, 1),
                score_breakdown={
                    "positive_gap_ratio": round(pos_gap_ratio, 1),
                    "negative_gap_impact": round(neg_gap_impact, 1),
                    "unescalated_critical_ratio": round(unescalated_ratio, 1),
                    "missing_evidence_ratio": round(missing_evidence_ratio, 1),
                    "volume_penalty": round(vol_penalty, 1),
                },
                primary_concerns=concerns,
            )
            results.append(score_obj)

        # Sort descending by composite score
        results.sort(key=lambda x: x.composite_risk_score, reverse=True)
        return results

    def score_entities(
        self,
        df: pd.DataFrame,
        positive_findings: List[PositiveGapFinding],
        negative_findings: List[NegativeGapFinding]
    ) -> List[SupervisoryScore]:
        """
        Calculates composite supervisory risk score for each Critical Sector Entity (CSE).
        """
        results: List[SupervisoryScore] = []
        entities = df["entity_id"].unique()

        neg_by_entity: Dict[str, List[NegativeGapFinding]] = {e: [] for e in entities}
        for nf in negative_findings:
            if nf.target_type == "ENTITY" and nf.target_id in neg_by_entity:
                neg_by_entity[nf.target_id].append(nf)

        pos_by_entity: Dict[str, List[PositiveGapFinding]] = {e: [] for e in entities}
        for pf in positive_findings:
            if pf.entity_id in pos_by_entity:
                pos_by_entity[pf.entity_id].append(pf)

        for entity_id in entities:
            ent_df = df[df["entity_id"] == entity_id]
            total_cases = len(ent_df)
            if total_cases == 0:
                continue

            neg_list = neg_by_entity.get(entity_id, [])
            pos_list = pos_by_entity.get(entity_id, [])

            # 1. Telemetry Blindspot Score (0 - 100)
            blindspot_count = sum(1 for nf in neg_list if nf.rule_id == "NEG_01")
            blindspot_score = min(100.0, blindspot_count * 45.0)

            # 2. Silence Window Penalty (0 - 100)
            has_silence = any(nf.rule_id == "NEG_03" for nf in neg_list)
            silence_penalty = 90.0 if has_silence else 0.0

            # 3. Overall Case Gap Rate (0 - 100)
            unique_flagged_cases = set(p.case_id for p in pos_list)
            case_gap_rate = (len(unique_flagged_cases) / total_cases * 100) if total_cases > 0 else 0.0

            # 4. Evidence Deficiency Rate (0 - 100)
            missing_evidence_count = len(ent_df[~ent_df["evidence_completeness_flag"]])
            evidence_deficiency_rate = (missing_evidence_count / total_cases * 100) if total_cases > 0 else 0.0

            # Weighted Composite
            w = self.entity_weights
            composite = (
                (blindspot_score * w.get("telemetry_blindspot_score", 0.40)) +
                (silence_penalty * w.get("silence_window_penalty", 0.30)) +
                (case_gap_rate * w.get("overall_case_gap_rate", 0.20)) +
                (evidence_deficiency_rate * w.get("evidence_deficiency_rate", 0.10))
            )
            composite = max(0.0, min(100.0, composite))

            concerns: List[str] = []
            if blindspot_count > 0:
                blindspot_types = [nf.affected_alert_type for nf in neg_list if nf.affected_alert_type]
                concerns.append(f"Telemetry blindspots in {len(blindspot_types)} threat categories ({', '.join(blindspot_types[:2])})")
            if has_silence:
                silence_finding = next(nf for nf in neg_list if nf.rule_id == "NEG_03")
                concerns.append(f"Telemetry blackout window ({silence_finding.time_window})")
            if case_gap_rate >= 25.0:
                concerns.append(f"High analyst handling gap rate ({case_gap_rate:.0f}%)")
            if evidence_deficiency_rate >= 30.0:
                concerns.append(f"Evidence attachment omission ({evidence_deficiency_rate:.0f}%)")

            if not concerns:
                concerns.append("Normal telemetry feed and compliant case processing")

            crit_high_df = ent_df[ent_df["severity"].isin(["high", "critical"])]
            unesc_count = len(crit_high_df[crit_high_df["escalation_status"] != "escalated"])
            avg_time = float(ent_df["time_to_close_minutes"].mean())

            score_obj = SupervisoryScore(
                target_id=entity_id,
                target_type="ENTITY",
                composite_risk_score=round(composite, 1),
                supervisory_tier=self._determine_tier(composite),
                total_cases_handled=total_cases,
                positive_gap_count=len(pos_list),
                negative_gap_count=len(neg_list),
                unescalated_critical_count=unesc_count,
                missing_evidence_count=missing_evidence_count,
                avg_close_time_minutes=round(avg_time, 1),
                score_breakdown={
                    "blindspot_score": round(blindspot_score, 1),
                    "silence_penalty": round(silence_penalty, 1),
                    "case_gap_rate": round(case_gap_rate, 1),
                    "evidence_deficiency_rate": round(evidence_deficiency_rate, 1),
                },
                primary_concerns=concerns,
            )
            results.append(score_obj)

        results.sort(key=lambda x: x.composite_risk_score, reverse=True)
        return results
