from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


class PeerBenchmarkingEngine:
    """
    Computes cohort baselines, peer distributions, z-scores, and IQR fences
    across analysts and critical sector entities (CSEs).
    """

    def __init__(self):
        pass

    def compute_analyst_volume_benchmarks(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculates case volume distribution across all analysts,
        computing mean, standard deviation, z-scores, and IQR fences.
        """
        counts = df["analyst_id"].value_counts()
        analysts = counts.index.tolist()
        volumes = counts.values.astype(float)

        if len(volumes) < 3:
            # Not enough data for robust stats
            return {
                "mean": float(np.mean(volumes)) if len(volumes) > 0 else 0.0,
                "std": float(np.std(volumes)) if len(volumes) > 0 else 1.0,
                "median": float(np.median(volumes)) if len(volumes) > 0 else 0.0,
                "q1": 0.0,
                "q3": 0.0,
                "iqr_lower_fence": 0.0,
                "analyst_stats": {a: {"count": int(c), "z_score": 0.0, "is_iqr_outlier": False} for a, c in zip(analysts, volumes)},
            }

        mean_val = float(np.mean(volumes))
        std_val = float(np.std(volumes, ddof=1)) if len(volumes) > 1 else 1.0
        if std_val == 0:
            std_val = 1.0

        q1 = float(np.percentile(volumes, 25))
        q3 = float(np.percentile(volumes, 75))
        iqr = q3 - q1
        iqr_lower_fence = float(q1 - 1.5 * iqr)

        analyst_stats: Dict[str, Dict[str, Any]] = {}
        for analyst_id, count in zip(analysts, volumes):
            z = float((count - mean_val) / std_val)
            is_outlier = count < iqr_lower_fence or z < -1.40
            analyst_stats[analyst_id] = {
                "count": int(count),
                "z_score": round(z, 2),
                "is_iqr_outlier": is_outlier,
            }

        return {
            "mean": round(mean_val, 2),
            "std": round(std_val, 2),
            "median": round(float(np.median(volumes)), 2),
            "q1": round(q1, 2),
            "q3": round(q3, 2),
            "iqr_lower_fence": round(iqr_lower_fence, 2),
            "analyst_stats": analyst_stats,
        }

    def compute_entity_telemetry_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Builds cross-tabulated entity x alert_type matrix and computes
        cohort-level alert frequencies and sector averages.
        """
        ct = pd.crosstab(df["entity_id"], df["alert_type"])
        alert_types = ct.columns.tolist()
        entities = ct.index.tolist()

        # Sector/Cohort averages
        peer_type_averages = ct.mean(axis=0).to_dict()

        return {
            "crosstab": ct,
            "alert_types": alert_types,
            "entities": entities,
            "peer_type_averages": {k: round(float(v), 2) for k, v in peer_type_averages.items()},
        }

    def compute_analyst_evidence_rates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Computes the evidence completeness rate per analyst compared to cohort baseline.
        """
        analyst_grp = df.groupby("analyst_id")
        total_cases = analyst_grp.size()
        
        # Missing evidence is where evidence_completeness_flag == False
        missing_cases = df[~df["evidence_completeness_flag"]].groupby("analyst_id").size()
        
        cohort_total = len(df)
        cohort_missing = len(df[~df["evidence_completeness_flag"]])
        cohort_missing_rate = (cohort_missing / cohort_total) if cohort_total > 0 else 0.0

        analyst_rates: Dict[str, Dict[str, Any]] = {}
        for analyst_id in total_cases.index:
            total = int(total_cases[analyst_id])
            missing = int(missing_cases.get(analyst_id, 0))
            missing_rate = (missing / total) if total > 0 else 0.0
            analyst_rates[analyst_id] = {
                "total_cases": total,
                "missing_evidence_cases": missing,
                "missing_evidence_rate": round(missing_rate, 3),
            }

        return {
            "cohort_missing_evidence_rate": round(cohort_missing_rate, 3),
            "analyst_rates": analyst_rates,
        }
