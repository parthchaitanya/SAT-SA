import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd
from faker import Faker

from core.schema import CaseRecord

fake = Faker()
Faker.seed(42)
random.seed(42)

# Entities across Indian Critical Sector Entities (CSEs) under NCIIPC oversight
CRITICAL_SECTOR_ENTITIES = [
    {"entity_id": "CSE-POWER-01", "name": "National Power Grid Corp", "sector": "Energy & Power"},
    {"entity_id": "CSE-POWER-02", "name": "Regional Thermal Dispatch", "sector": "Energy & Power"},
    {"entity_id": "CSE-FIN-01", "name": "Apex Central Clearing Bank", "sector": "Banking & Finance"},
    {"entity_id": "CSE-FIN-02", "name": "National Stock & Securities", "sector": "Banking & Finance"},
    {"entity_id": "CSE-TEL-01", "name": "Bharat Telecom Backbone", "sector": "Telecom"},
    {"entity_id": "CSE-TEL-02", "name": "Strategic SatCom Gateway", "sector": "Telecom"},
    {"entity_id": "CSE-TRANS-01", "name": "Civil Aviation Air Traffic Control", "sector": "Transportation"},
    {"entity_id": "CSE-TRANS-02", "name": "National Freight Rail System", "sector": "Transportation"},
    {"entity_id": "CSE-HEALTH-01", "name": "National Health Record Exchange", "sector": "Healthcare"},
    {"entity_id": "CSE-DEF-01", "name": "Strategic Defense Tech Grid", "sector": "Defense & Strategic"},
]

# Human SOC Analysts with specific behavioral personas for audit demonstration
ANALYST_PERSONAS = [
    {
        "analyst_id": "ANL-R-SHARMA",
        "name": "Rohit Sharma",
        "role": "L1 Triage Analyst",
        "experience_years": 1.5,
        "persona_tendency": "copy_paster_minimalist",  # Injects POS_01 and POS_03
    },
    {
        "analyst_id": "ANL-P-VERMA",
        "name": "Pooja Verma",
        "role": "L1 Triage Analyst",
        "experience_years": 2.0,
        "persona_tendency": "speed_rusher_missing_evidence",  # Injects POS_04 and NEG_04
    },
    {
        "analyst_id": "ANL-S-PATEL",
        "name": "Siddharth Patel",
        "role": "L2 Security Analyst",
        "experience_years": 3.5,
        "persona_tendency": "escalation_hesitant",  # Injects POS_02 (closes high/critical unescalated)
    },
    {
        "analyst_id": "ANL-N-RAO",
        "name": "Naveen Rao",
        "role": "L1 Triage Analyst",
        "experience_years": 1.0,
        "persona_tendency": "low_volume_slacker",  # Injects NEG_02 (outlier low case count)
    },
    {
        "analyst_id": "ANL-A-GUPTA",
        "name": "Ananya Gupta",
        "role": "Senior L2 Analyst",
        "experience_years": 5.0,
        "persona_tendency": "diligent_benchmark",  # High quality, reference benchmark
    },
    {
        "analyst_id": "ANL-M-SINGH",
        "name": "Manish Singh",
        "role": "L1 Triage Analyst",
        "experience_years": 2.5,
        "persona_tendency": "mostly_compliant",  # Typical good analyst with rare minor gaps
    },
    {
        "analyst_id": "ANL-K-DESHMUKH",
        "name": "Kavita Deshmukh",
        "role": "L2 Security Analyst",
        "experience_years": 4.0,
        "persona_tendency": "diligent_benchmark",  # High quality, reference benchmark
    },
]

ALERT_TYPES = [
    "Ransomware Activity Detected",
    "Unauthorized Privilege Escalation",
    "DDoS Volume Spike",
    "Lateral Movement SMB/WinRM",
    "Data Exfiltration Over DNS/HTTPS",
    "Phishing Credential Harvest",
    "Brute Force Authentication Storm",
    "Suspicious PowerShell Script Execution",
    "Anomalous C2 Outbound Beaconing",
    "SCADA Protocol Anomaly / Modbus Tampering",
]

NORMAL_ACTIONS = {
    "low": [
        "Closed - False Positive after IP reputation check",
        "Closed - Expected user password reset activity",
        "Whitelisted legitimate scheduled cron task",
        "User contacted and confirmed MFA push approval",
    ],
    "medium": [
        "Endpoint isolated temporarily, cached credentials cleared, restored",
        "Investigated anomalous script execution; confirmed authorized IT patch tool",
        "Blocked external malicious IP on perimeter firewall and closed case",
        "Reset compromised service account credential and confirmed session termination",
    ],
    "high": [
        "Host quarantined, malware artifact submitted to sandbox, escalated to L2 Incident Response",
        "Revoked compromised domain admin session, initiated credential rotation, escalated to CIRT",
        "Terminated suspicious outbound tunnel, packet capture attached, escalated to Tier-2 SOC",
        "Corroborated lateral movement telemetry with EDR logs, host containment verified, escalated",
    ],
    "critical": [
        "CRITICAL: Emergency host isolation engaged, domain controller lockdown advised, escalated to NCIIPC CIRT",
        "CRITICAL: Active ransomware payload blocked on file server, volume snapshots preserved, escalated to Tier-3 IR",
        "CRITICAL: Multi-node C2 beaconing confirmed across SCADA subnet; physical airgap enforced and escalated to NCIIPC",
    ],
}

TEMPLATED_NOTES_POOL = [
    "Alert reviewed and found to be benign. Standard operating procedure followed. No indicators of compromise discovered upon inspection of logs. Case marked as closed.",
    "Alert reviewed and found to be benign. Standard operating procedure followed. No indicators of compromise discovered upon inspection of logs. Case marked as closed.",
    "Standard triage conducted. Event was verified against internal known whitelist. No further action needed.",
    "Standard triage conducted. Event was verified against internal known whitelist. No further action needed.",
    "Standard triage conducted. Event was verified against internal known whitelist. No further action needed.",
]

MINIMAL_NOTES_POOL = [
    "ok",
    "done",
    "fp",
    "closed",
    "no issue",
    "checked",
    "n/a",
    "cleared",
    "false positive",
    "resolved",
]


def generate_synthetic_records(
    num_records: int = 140,
    start_date: Optional[datetime] = None,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Generate realistic SOC case-management records spanning 90 days.
    Deliberately injects positive and negative space gaps into ~20-25% of records.
    """
    random.seed(seed)
    Faker.seed(seed)

    if start_date is None:
        start_date = datetime.now() - timedelta(days=90)

    records: List[Dict[str, Any]] = []

    # Map entities and analysts
    entity_ids = [e["entity_id"] for e in CRITICAL_SECTOR_ENTITIES]
    
    # We will generate case records across 90 days
    # To demonstrate NEG_03 (Entity Silence Window): CSE-FIN-02 has 0 alerts during days 35 to 50
    # To demonstrate NEG_01 (Missing Expected Telemetry): CSE-POWER-01 never gets 'Ransomware Activity Detected' or 'SCADA Protocol Anomaly'
    
    analyst_weights = {
        "ANL-R-SHARMA": 28,      # Standard volume
        "ANL-P-VERMA": 28,       # Standard volume
        "ANL-S-PATEL": 25,       # Standard volume
        "ANL-A-GUPTA": 28,       # Standard volume (Diligent benchmark)
        "ANL-M-SINGH": 25,       # Standard volume
        "ANL-K-DESHMUKH": 25,    # Standard volume (Diligent benchmark)
        "ANL-N-RAO": 5,          # DELIBERATE NEG_02: Extreme low volume outlier (only 4-6 cases)
    }
    
    # Normalize distribution
    analysts_pool = []
    for aid, count in analyst_weights.items():
        analysts_pool.extend([aid] * count)

    for i in range(1, num_records + 1):
        alert_id = f"ALT-2026-{1000 + i}"
        case_id = f"CASE-2026-{4000 + i}"
        
        # Pick random day offset between 0 and 89
        day_offset = random.randint(0, 89)
        alert_timestamp = start_date + timedelta(
            days=day_offset,
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59)
        )

        # Pick entity
        entity_obj = random.choice(CRITICAL_SECTOR_ENTITIES)
        entity_id = entity_obj["entity_id"]

        # Enforce NEG_03 (Silence window for CSE-FIN-02 during days 35-50)
        if entity_id == "CSE-FIN-02" and 35 <= day_offset <= 50:
            # Shift to another entity to simulate complete silence for CSE-FIN-02
            entity_id = "CSE-FIN-01"

        # Pick alert type
        alert_type = random.choice(ALERT_TYPES)

        # Enforce NEG_01 (Missing Telemetry for CSE-POWER-01: zero Ransomware or SCADA alerts)
        if entity_id == "CSE-POWER-01" and alert_type in ["Ransomware Activity Detected", "SCADA Protocol Anomaly / Modbus Tampering"]:
            # Shift to common low-level alert so POWER-01 has an analytical blindspot
            alert_type = "Brute Force Authentication Storm"

        # Pick analyst
        analyst_id = random.choice(analysts_pool)
        persona = next(a["persona_tendency"] for a in ANALYST_PERSONAS if a["analyst_id"] == analyst_id)

        # Severity distribution
        sev_roll = random.random()
        if sev_roll < 0.30:
            severity = "low"
        elif sev_roll < 0.65:
            severity = "medium"
        elif sev_roll < 0.88:
            severity = "high"
        else:
            severity = "critical"

        # Default normal parameters
        evidence_complete = True
        escalation_status = "not_escalated"
        if severity in ["high", "critical"]:
            escalation_status = "escalated" if random.random() < 0.90 else "not_escalated"

        # Realistic time to close by severity
        if severity == "low":
            time_to_close = random.uniform(5, 30)
        elif severity == "medium":
            time_to_close = random.uniform(15, 60)
        elif severity == "high":
            time_to_close = random.uniform(35, 180)
        else:
            time_to_close = random.uniform(60, 360)

        # Realistic notes generator
        action_taken = random.choice(NORMAL_ACTIONS[severity])
        investigation_notes = (
            f"Analyst verified alert telemetry for {entity_id}. "
            f"Correlated host event logs and network flow data. "
            f"Evaluated threat signatures against MITRE ATT&CK matrix. "
            f"Action performed: {action_taken}. Verification completed successfully."
        )

        # --- DELIBERATE GAP INJECTIONS BASED ON PERSONAS & GAP TARGETS ---

        # 1. ANL-R-SHARMA: Copy-Paster & Minimalist (POS_01 & POS_03)
        if persona == "copy_paster_minimalist":
            gap_type_roll = random.random()
            if gap_type_roll < 0.40:
                # POS_01: Minimal / Empty notes
                investigation_notes = random.choice(MINIMAL_NOTES_POOL)
                action_taken = "Closed without detailed triage"
            elif gap_type_roll < 0.85:
                # POS_03: Templated Near-Duplicate Notes
                investigation_notes = random.choice(TEMPLATED_NOTES_POOL)
                action_taken = "Closed - Benign verified"

        # 2. ANL-P-VERMA: Speed Rusher & Missing Evidence (POS_04 & NEG_04)
        elif persona == "speed_rusher_missing_evidence":
            # POS_04: Abnormally fast closure
            if severity in ["high", "critical"] and random.random() < 0.65:
                time_to_close = random.uniform(1.2, 4.5)  # Under 5 mins for high/critical!
                investigation_notes = f"Quick inspection of {alert_type}. Looks non-critical. Closed promptly."
            # NEG_04: High rate of missing evidence flag
            if random.random() < 0.60:
                evidence_complete = False

        # 3. ANL-S-PATEL: Escalation Hesitant (POS_02)
        elif persona == "escalation_hesitant":
            if severity in ["high", "critical"] and random.random() < 0.70:
                # POS_02: Critical/High alert closed as not_escalated
                escalation_status = "not_escalated"
                action_taken = "Closed - Assumed False Positive without CIRT notification"
                investigation_notes = (
                    f"Observed {severity} alert {alert_type}. "
                    "Decided not to escalate to Tier-2 / CIRT to avoid alert fatigue. "
                    "Marked resolved locally."
                )

        # Minor sporadic gaps in general pool to reflect ~20-25% total gap rate
        elif persona == "mostly_compliant" and random.random() < 0.12:
            if random.random() < 0.5:
                evidence_complete = False
            else:
                time_to_close = max(2.0, time_to_close * 0.25)

        closure_time = alert_timestamp + timedelta(minutes=time_to_close)

        record = {
            "alert_id": alert_id,
            "timestamp": alert_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "entity_id": entity_id,
            "analyst_id": analyst_id,
            "severity": severity,
            "alert_type": alert_type,
            "action_taken": action_taken,
            "investigation_notes": investigation_notes,
            "escalation_status": escalation_status,
            "evidence_completeness_flag": evidence_complete,
            "closure_time": closure_time.strftime("%Y-%m-%d %H:%M:%S"),
            "case_id": case_id,
            "time_to_close_minutes": round(time_to_close, 2),
        }
        records.append(record)

    # Sort chronologically
    records.sort(key=lambda x: x["timestamp"])
    return records


def get_synthetic_dataframe(num_records: int = 140, seed: int = 42) -> pd.DataFrame:
    """Helper to get synthetic dataset as a clean typed pandas DataFrame."""
    records = generate_synthetic_records(num_records=num_records, seed=seed)
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["closure_time"] = pd.to_datetime(df["closure_time"])
    df["evidence_completeness_flag"] = df["evidence_completeness_flag"].astype(bool)
    df["time_to_close_minutes"] = df["time_to_close_minutes"].astype(float)
    return df
