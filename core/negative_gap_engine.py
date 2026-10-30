import os
from typing import List, Dict, Any, Optional
from datetime import timedelta
import pandas as pd
import numpy as np
import yaml

from core.schema import NegativeGapFinding
from core.peer_benchmarking import PeerBenchmarkingEngine


def load_rules_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load threshold configuration from rules.yaml or return robust defaults."""
    if config_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, "config", "rules.yaml")

    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    return {
        "negative_space_rules": {
            "telemetry_min_expected_ratio": 0.20,
            "analyst_volume_zscore_lower_threshold": -1.40,
            "analyst_volume_iqr_multiplier": 1.5,
            "silence_window_days": 10,
            "silence_volume_drop_ratio": 0.20,
            "analyst_missing_evidence_rate_threshold": 0.30,
        }
    }


class NegativeGapEngine:
    """
    Negative Space Gap Detection Engine: Identifies unseen, omitted,
    or statistically aberrant patterns across analysts and CSE entities.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = load_rules_config()
        self.config = config.get("negative_space_rules", {})
        self.telemetry_min_ratio = self.config.get("telemetry_min_expected_ratio", 0.20)
        self.zscore_lower = self.config.get("analyst_volume_zscore_lower_threshold", -1.40)
        self.silence_window_days = self.config.get("silence_window_days", 10)
        self.silence_drop_ratio = self.config.get("silence_volume_drop_ratio", 0.20)
        self.missing_evidence_threshold = self.config.get("analyst_missing_evidence_rate_threshold", 0.30)
        self.benchmarking = PeerBenchmarkingEngine()

    def detect_telemetry_blindspots(self, df: pd.DataFrame) -> List[NegativeGapFinding]:
        """
        Rule NEG_01: Entities with zero or near-zero alerts of a category
        that peer entities consistently generate (missing expected telemetry).
        """
        findings: List[NegativeGapFinding] = []
        benchmarks = self.benchmarking.compute_entity_telemetry_distribution(df)
        ct = benchmarks["crosstab"]
        peer_averages = benchmarks["peer_type_averages"]

        for entity_id in ct.index:
            for alert_type, peer_avg in peer_averages.items():
                if peer_avg < 1.0:  # Ignore rare alert types
                    continue

                observed_count = float(ct.loc[entity_id, alert_type]) if alert_type in ct.columns else 0.0
                ratio = observed_count / peer_avg if peer_avg > 0 else 1.0

                # If peer average is significant (>= 1.5 alerts) and entity has 0 or < 20% of peer average
                if observed_count == 0 and peer_avg >= 1.2:
                    evidence = (
                        f"Critical Entity {entity_id} has ZERO recorded alerts for '{alert_type}', "
                        f"whereas peer critical entities average {peer_avg:.1f} alerts in the same period. "
                        f"This indicates a severe telemetry blindspot, misconfigured SIEM connector, or unmonitored attack vector."
                    )
                    findings.append(
                        NegativeGapFinding(
                            rule_id="NEG_01",
                            rule_name="Telemetry Blindspot / Missing Expected Threat Category",
                            target_type="ENTITY",
                            target_id=str(entity_id),
                            gap_severity="HIGH",
                            metric_name="Alert Category Volume vs Peer Average",
                            observed_value=observed_count,
                            baseline_value=peer_avg,
                            evidence_detail=evidence,
                            affected_alert_type=alert_type,
                        )
                    )
                elif ratio < self.telemetry_min_ratio and peer_avg >= 2.5:
                    evidence = (
                        f"Entity {entity_id} recorded only {int(observed_count)} alerts for '{alert_type}' "
                        f"(ratio {ratio:.1%}), significantly below peer baseline of {peer_avg:.1f} alerts. "
                        f"Indicates probable log filter suppression or partial sensor degradation."
                    )
                    findings.append(
                        NegativeGapFinding(
                            rule_id="NEG_01",
                            rule_name="Anomalously Low Category Telemetry Volume",
                            target_type="ENTITY",
                            target_id=str(entity_id),
                            gap_severity="MEDIUM",
                            metric_name="Alert Category Volume vs Peer Average",
                            observed_value=observed_count,
                            baseline_value=peer_avg,
                            evidence_detail=evidence,
                            affected_alert_type=alert_type,
                        )
                    )
        return findings

    def detect_analyst_volume_outliers(self, df: pd.DataFrame) -> List[NegativeGapFinding]:
        """
        Rule NEG_02: Analysts whose case volume is a statistical outlier (low) vs peer group (z-score / IQR).
        """
        findings: List[NegativeGapFinding] = []
        benchmarks = self.benchmarking.compute_analyst_volume_benchmarks(df)
        mean_vol = benchmarks["mean"]
        analyst_stats = benchmarks["analyst_stats"]

        for analyst_id, stats in analyst_stats.items():
            z = stats["z_score"]
            count = stats["count"]
            is_outlier = stats["is_iqr_outlier"]

            if is_outlier or z <= self.zscore_lower:
                gap_sev = "HIGH" if z <= -1.8 else "MEDIUM"
                evidence = (
                    f"Analyst {analyst_id} closed only {count} cases over the 90-day assessment period "
                    f"(Z-score: {z:.2f}, Cohort Mean: {mean_vol:.1f} cases). "
                    f"This extreme statistical under-volume indicates potential workload disengagement, "
                    f"cherry-picking, or severe under-utilization."
                )
                findings.append(
                    NegativeGapFinding(
                        rule_id="NEG_02",
                        rule_name="Analyst Case Volume Statistical Low Outlier",
                        target_type="ANALYST",
                        target_id=str(analyst_id),
                        gap_severity=gap_sev,
                        metric_name="Case Volume Z-Score",
                        observed_value=float(count),
                        baseline_value=mean_vol,
                        evidence_detail=evidence,
                    )
                )
        return findings

    def detect_entity_silence_windows(self, df: pd.DataFrame) -> List[NegativeGapFinding]:
        """
        Rule NEG_03: Time windows with anomalously low overall activity for an entity
        compared to its own historical baseline (silence / telemetry failure periods).
        """
        findings: List[NegativeGapFinding] = []
        if "timestamp" not in df.columns or len(df) == 0:
            return findings

        df_copy = df.copy()
        df_copy["timestamp"] = pd.to_datetime(df_copy["timestamp"])
        min_date = df_copy["timestamp"].min()
        max_date = df_copy["timestamp"].max()
        total_span_days = (max_date - min_date).days

        if total_span_days < (self.silence_window_days * 2):
            return findings

        # Analyze each entity's daily rolling counts
        for entity_id in df_copy["entity_id"].unique():
            ent_df = df_copy[df_copy["entity_id"] == entity_id]
            daily_counts = ent_df.set_index("timestamp").resample("D").size()
            
            # Reindex across full date range to capture 0 days
            full_date_idx = pd.date_range(start=min_date.floor("D"), end=max_date.ceil("D"), freq="D")
            daily_series = daily_counts.reindex(full_date_idx, fill_value=0)

            overall_daily_mean = daily_series.mean()
            if overall_daily_mean == 0:
                continue

            # Rolling window of N days
            rolling_sum = daily_series.rolling(window=self.silence_window_days).sum()
            expected_window_volume = overall_daily_mean * self.silence_window_days

            # Find windows where volume is 0 or drops below threshold ratio
            low_windows = rolling_sum[rolling_sum <= (expected_window_volume * self.silence_drop_ratio)]
            
            if not low_windows.empty:
                # Find worst window
                worst_end_date = low_windows.idxmin()
                worst_vol = float(low_windows.min())
                worst_start_date = worst_end_date - timedelta(days=self.silence_window_days - 1)
                
                window_str = f"{worst_start_date.strftime('%Y-%m-%d')} to {worst_end_date.strftime('%Y-%m-%d')}"
                evidence = (
                    f"Entity {entity_id} experienced a critical telemetry silence window between {window_str}. "
                    f"Observed alert count in this {self.silence_window_days}-day period was {int(worst_vol)}, "
                    f"compared to expected baseline of {expected_window_volume:.1f} alerts. "
                    f"Indicates telemetry blackout, forwarder network failure, or agent heartbeat stoppage."
                )
                findings.append(
                    NegativeGapFinding(
                        rule_id="NEG_03",
                        rule_name="Entity Telemetry Blackout / Silence Window",
                        target_type="ENTITY",
                        target_id=str(entity_id),
                        gap_severity="CRITICAL" if worst_vol == 0 else "HIGH",
                        metric_name=f"{self.silence_window_days}-Day Window Alert Volume",
                        observed_value=worst_vol,
                        baseline_value=round(expected_window_volume, 1),
                        evidence_detail=evidence,
                        time_window=window_str,
                    )
                )
        return findings

    def detect_evidence_omission_patterns(self, df: pd.DataFrame) -> List[NegativeGapFinding]:
        """
        Rule NEG_04: Cases with evidence_completeness_flag missing across an unusually
        high share of an analyst's closed cases.
        """
        findings: List[NegativeGapFinding] = []
        benchmarks = self.benchmarking.compute_analyst_evidence_rates(df)
        cohort_rate = benchmarks["cohort_missing_evidence_rate"]
        analyst_rates = benchmarks["analyst_rates"]

        for analyst_id, data in analyst_rates.items():
            total = data["total_cases"]
            missing_count = data["missing_evidence_cases"]
            missing_rate = data["missing_evidence_rate"]

            if total >= 5 and (missing_rate >= self.missing_evidence_threshold or missing_rate >= (cohort_rate * 2.0)):
                gap_sev = "HIGH" if missing_rate >= 0.50 else "MEDIUM"
                evidence = (
                    f"Analyst {analyst_id} failed to attach complete forensic evidence on {missing_count} out of "
                    f"{total} closed cases (Missing Evidence Rate: {missing_rate:.1%}, vs Cohort Baseline: {cohort_rate:.1%}). "
                    f"Systematic omission of triage artifacts, PCAPs, or sandbox reports violates NCIIPC audit guidelines."
                )
                findings.append(
                    NegativeGapFinding(
                        rule_id="NEG_04",
                        rule_name="Systematic Forensic Evidence Omission Pattern",
                        target_type="ANALYST",
                        target_id=str(analyst_id),
                        gap_severity=gap_sev,
                        metric_name="Missing Evidence Flag Rate",
                        observed_value=round(missing_rate * 100, 1),
                        baseline_value=round(cohort_rate * 100, 1),
                        evidence_detail=evidence,
                    )
                )
        return findings

    def run_all_negative_rules(self, df: pd.DataFrame) -> List[NegativeGapFinding]:
        """Execute all negative space gap rules and return consolidated findings."""
        findings: List[NegativeGapFinding] = []
        findings.extend(self.detect_telemetry_blindspots(df))
        findings.extend(self.detect_analyst_volume_outliers(df))
        findings.extend(self.detect_entity_silence_windows(df))
        findings.extend(self.detect_evidence_omission_patterns(df))
        return findings
