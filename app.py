import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime
import os
import yaml

# Core in-process analytical modules
from core.data_generator import get_synthetic_dataframe, CRITICAL_SECTOR_ENTITIES, ANALYST_PERSONAS
from core.positive_gap_engine import PositiveGapEngine, load_rules_config
from core.negative_gap_engine import NegativeGapEngine
from core.peer_benchmarking import PeerBenchmarkingEngine
from core.risk_scoring import SupervisoryRiskScoringEngine
from core.explainability import OllamaExplainabilityEngine

# Configure Streamlit page (Engineering / Utilitarian Terminal)
st.set_page_config(
    page_title="SAT-SA // NCIIPC Supervisory Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Engineering Cyber-Command Theme (High-contrast, utilitarian, dense, monospace-accented)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    code, pre, .mono-font, .metric-value, .status-tag, [data-baseweb="tab"] {
        font-family: 'JetBrains Mono', 'Consolas', 'Courier New', monospace !important;
    }

    /* Main background */
    .stApp {
        background-color: #0b0f17;
        color: #d1d7e0;
    }

    /* Technical Command Header */
    .command-header {
        background: #111827;
        border: 1px solid #1f2937;
        border-left: 4px solid #3b82f6;
        border-radius: 4px;
        padding: 16px 22px;
        margin-bottom: 20px;
    }

    .system-status-strip {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: #60a5fa;
        margin-bottom: 6px;
    }
    
    .command-title {
        font-size: 20px;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: -0.2px;
        color: #f3f4f6;
        margin: 0 0 6px 0;
    }
    
    .command-subtitle {
        font-size: 13px;
        color: #9ca3af;
        margin: 0;
        line-height: 1.4;
    }

    /* Dense Metric Card */
    .metric-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-top: 2px solid #374151;
        border-radius: 4px;
        padding: 12px 16px;
    }
    
    .metric-card.accent-blue { border-top-color: #3b82f6; }
    .metric-card.accent-red { border-top-color: #ef4444; }
    .metric-card.accent-amber { border-top-color: #f59e0b; }
    .metric-card.accent-orange { border-top-color: #f97316; }
    .metric-card.accent-cyan { border-top-color: #06b6d4; }
    
    .metric-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #9ca3af;
        margin-bottom: 4px;
    }
    
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #f9fafb;
        line-height: 1.1;
    }
    
    .metric-sub {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        color: #6b7280;
        margin-top: 4px;
    }

    /* Utilitarian Status Tags */
    .status-tag {
        display: inline-block;
        font-size: 11px;
        font-weight: 600;
        padding: 2px 6px;
        border-radius: 2px;
        border: 1px solid transparent;
        text-transform: uppercase;
    }

    .tag-critical {
        background: #2b1216;
        color: #f87171;
        border-color: #7f1d1d;
    }
    
    .tag-elevated {
        background: #2c1a0e;
        color: #fb923c;
        border-color: #7c2d12;
    }
    
    .tag-moderate {
        background: #1e1b12;
        color: #fbbf24;
        border-color: #78350f;
    }
    
    .tag-compliant {
        background: #0d2217;
        color: #4ade80;
        border-color: #14532d;
    }

    .tag-info {
        background: #0f1e36;
        color: #60a5fa;
        border-color: #1e3a8a;
    }

    /* Explanation / Technical Dossier Box */
    .ai-brief-box {
        background: #0e1626;
        border: 1px solid #1e3a8a;
        border-left: 3px solid #3b82f6;
        border-radius: 4px;
        padding: 16px;
        margin-top: 12px;
    }

    .brief-header {
        font-family: 'JetBrains Mono', monospace;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        color: #60a5fa;
        margin-bottom: 8px;
    }

    /* Section Subheaders */
    .section-header {
        font-family: 'JetBrains Mono', monospace;
        font-size: 14px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #e5e7eb;
        border-bottom: 1px solid #1f2937;
        padding-bottom: 4px;
        margin-top: 12px;
        margin-bottom: 8px;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: transparent;
        border-bottom: 1px solid #1f2937;
        padding-bottom: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #111827;
        border: 1px solid #1f2937;
        border-bottom: none;
        border-radius: 3px 3px 0 0;
        color: #9ca3af;
        padding: 6px 14px;
        font-size: 12px;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: #1f2937 !important;
        border-color: #3b82f6 !important;
        color: #60a5fa !important;
        font-weight: 600 !important;
    }

    /* Streamlit overrides for high density */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# STATE MANAGEMENT & DATA CACHING
# ==========================================
if "data_seed" not in st.session_state:
    st.session_state.data_seed = 42

if "num_records" not in st.session_state:
    st.session_state.num_records = 140

@st.cache_data
def load_and_process_dataset(num_records: int, seed: int):
    """Generate or retrieve synthetic records and execute analytical pipeline."""
    df = get_synthetic_dataframe(num_records=num_records, seed=seed)
    
    # Run Positive Space Engine
    pos_engine = PositiveGapEngine()
    pos_findings = pos_engine.run_all_positive_rules(df)
    
    # Run Negative Space Engine
    neg_engine = NegativeGapEngine()
    neg_findings = neg_engine.run_all_negative_rules(df)
    
    # Run Risk Scoring Engine
    risk_engine = SupervisoryRiskScoringEngine()
    analyst_scores = risk_engine.score_analysts(df, pos_findings, neg_findings)
    entity_scores = risk_engine.score_entities(df, pos_findings, neg_findings)
    
    return df, pos_findings, neg_findings, analyst_scores, entity_scores


# ==========================================
# SIDEBAR CONTROLS & SYSTEM DIAGNOSTICS
# ==========================================
with st.sidebar:
    st.markdown('<div class="section-header">// SYSTEM CONTROLS</div>', unsafe_allow_html=True)
    
    st.code("MODE: AIR-GAPPED\nEXEC: IN-PROCESS\nNET: 0 EXTERNAL CALLS", language="yaml")
    
    # Ollama Model Selection
    explainer = OllamaExplainabilityEngine()
    is_ollama_online = explainer.is_ollama_available()
    local_models = explainer.list_local_models() if is_ollama_online else ["offline-fallback"]
    
    if is_ollama_online:
        st.markdown('<span class="status-tag tag-compliant">[DAEMON: ONLINE] OLLAMA CONNECTED</span>', unsafe_allow_html=True)
        selected_model = st.selectbox("LLM Runtime Model", options=local_models, index=0)
    else:
        st.markdown('<span class="status-tag tag-elevated">[DAEMON: OFFLINE] DETERMINISTIC FALLBACK</span>', unsafe_allow_html=True)
        selected_model = "deterministic-fallback"

    st.markdown('<div class="section-header">// SIMULATION PARAMETERS</div>', unsafe_allow_html=True)
    num_rec_slider = st.slider("Case Record Count", min_value=100, max_value=250, value=st.session_state.num_records, step=10)
    seed_input = st.number_input("RNG Seed", min_value=1, max_value=9999, value=st.session_state.data_seed)
    
    if st.button("RECALCULATE PIPELINE", use_container_width=True):
        st.session_state.num_records = num_rec_slider
        st.session_state.data_seed = seed_input
        st.cache_data.clear()
        st.rerun()

    st.markdown('<div class="section-header">// TARGET AUTHORITY</div>', unsafe_allow_html=True)
    st.markdown("**NCIIPC / NTRO**")
    st.caption("National Critical Information Infrastructure Protection Centre")
    st.caption("SIH26157 Supervisory Analytics Architecture")


# Load Data
df, pos_findings, neg_findings, analyst_scores, entity_scores = load_and_process_dataset(
    st.session_state.num_records, st.session_state.data_seed
)

# Header Section
st.markdown("""
<div class="command-header">
    <div class="system-status-strip">SYSTEM: SAT-SA-V1.0 // NCIIPC-NTRO SUPERVISORY INTELLIGENCE // PROBLEM: SIH26157</div>
    <div class="command-title">SUPERVISORY ANALYTICS TOOL FOR SOC ASSESSMENT</div>
    <div class="command-subtitle">
        Automated Human Analyst Quality Audit & Negative-Space Telemetry Blindspot Detection across Critical Sector Entities (CSEs)
    </div>
</div>
""", unsafe_allow_html=True)


# Global KPI Cards
col1, col2, col3, col4, col5 = st.columns(5)

total_cases = len(df)
total_pos = len(pos_findings)
total_neg = len(neg_findings)
critical_analysts = sum(1 for a in analyst_scores if a.supervisory_tier in ["CRITICAL_SUPERVISORY_ATTENTION", "ELEVATED_RISK_REVIEW"])
blindspot_entities = sum(1 for e in entity_scores if e.score_breakdown.get("blindspot_score", 0) > 0)

with col1:
    st.markdown(f"""
    <div class="metric-card accent-blue">
        <div class="metric-title">Cases Evaluated</div>
        <div class="metric-value">{total_cases}</div>
        <div class="metric-sub">[WINDOW: 90 DAYS]</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card accent-red">
        <div class="metric-title">Positive Space Gaps</div>
        <div class="metric-value" style="color: #f87171;">{total_pos}</div>
        <div class="metric-sub">[DIRECT HUMAN ERRORS]</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card accent-amber">
        <div class="metric-title">Negative Space Gaps</div>
        <div class="metric-value" style="color: #fbbf24;">{total_neg}</div>
        <div class="metric-sub">[BLINDSPOTS & OMISSIONS]</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card accent-orange">
        <div class="metric-title">High-Risk Analysts</div>
        <div class="metric-value" style="color: #fb923c;">{critical_analysts}</div>
        <div class="metric-sub">[REQUIRES SUPERVISORY AUDIT]</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="metric-card accent-cyan">
        <div class="metric-title">Telemetry Blindspots</div>
        <div class="metric-value" style="color: #38bdf8;">{blindspot_entities} CSEs</div>
        <div class="metric-sub">[UNMONITORED FEEDS]</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ==========================================
# MAIN NAVIGATION TABS
# ==========================================
tab_overview, tab_positive, tab_negative, tab_drilldown, tab_config = st.tabs([
    "[01] OVERVIEW & RISK INDEX",
    "[02] POSITIVE SPACE ANALYSIS",
    "[03] NEGATIVE SPACE ANALYSIS",
    "[04] FORENSIC DRILL-DOWN & AI",
    "[05] CONFIG & AUDIT EXPORT"
])


# =========================================================
# TAB 1: EXECUTIVE OVERVIEW & RISK LEADERBOARDS
# =========================================================
with tab_overview:
    st.markdown('<div class="section-header">// SUPERVISORY PRIORITY LEADERBOARDS</div>', unsafe_allow_html=True)
    st.caption("Calculated via multi-factor weighted composite risk formula (0-100). Higher score indicates higher urgency for manual intervention.")

    c_left, c_right = st.columns([1, 1])

    with c_left:
        st.markdown("**ANALYST SUPERVISORY RISK RANKING**")
        
        analyst_rows = []
        for a in analyst_scores:
            analyst_rows.append({
                "Analyst ID": a.target_id,
                "Risk Score": a.composite_risk_score,
                "Supervisory Tier": f"[{a.supervisory_tier}]",
                "Cases": a.total_cases_handled,
                "Pos Gaps": a.positive_gap_count,
                "Neg Gaps": a.negative_gap_count,
                "Unesc. Critical": a.unescalated_critical_count,
                "Missing Evidence": a.missing_evidence_count,
                "Primary Concern": a.primary_concerns[0] if a.primary_concerns else "Compliant",
            })
        
        anl_table_df = pd.DataFrame(analyst_rows)
        st.dataframe(
            anl_table_df,
            use_container_width=True,
            column_config={
                "Risk Score": st.column_config.ProgressColumn(
                    "Risk Score (0-100)",
                    help="Composite supervisory score",
                    format="%.1f",
                    min_value=0,
                    max_value=100,
                ),
            },
            hide_index=True
        )

    with c_right:
        st.markdown("**CRITICAL SECTOR ENTITY (CSE) VULNERABILITY RANKING**")
        
        entity_rows = []
        for e in entity_scores:
            entity_rows.append({
                "Entity ID": e.target_id,
                "Risk Score": e.composite_risk_score,
                "Supervisory Tier": f"[{e.supervisory_tier}]",
                "Cases": e.total_cases_handled,
                "Blindspot Score": e.score_breakdown.get("blindspot_score", 0),
                "Silence Score": e.score_breakdown.get("silence_penalty", 0),
                "Primary Concern": e.primary_concerns[0] if e.primary_concerns else "Nominal",
            })
        
        ent_table_df = pd.DataFrame(entity_rows)
        st.dataframe(
            ent_table_df,
            use_container_width=True,
            column_config={
                "Risk Score": st.column_config.ProgressColumn(
                    "Risk Score (0-100)",
                    help="Composite vulnerability score",
                    format="%.1f",
                    min_value=0,
                    max_value=100,
                ),
            },
            hide_index=True
        )

    st.markdown('<div class="section-header">// 90-DAY TEMPORAL TELEMETRY TREND</div>', unsafe_allow_html=True)
    st.caption("Distribution of total ingested SOC cases vs detected operational gaps over the 90-day assessment window.")

    # Time series aggregation
    df_trend = df.copy()
    df_trend["date"] = df_trend["timestamp"].dt.date
    
    # Map findings by date
    pos_case_ids = set(p.case_id for p in pos_findings)
    df_trend["has_pos_gap"] = df_trend["case_id"].apply(lambda x: 1 if x in pos_case_ids else 0)
    
    # Daily aggregation
    daily_summary = df_trend.groupby("date").agg(
        Total_Cases=("case_id", "count"),
        Positive_Space_Gaps=("has_pos_gap", "sum"),
    ).reset_index()
    daily_summary.set_index("date", inplace=True)
    
    st.line_chart(daily_summary[["Total_Cases", "Positive_Space_Gaps"]], use_container_width=True)


# =========================================================
# TAB 2: POSITIVE SPACE FINDINGS
# =========================================================
with tab_positive:
    st.markdown('<div class="section-header">// POSITIVE SPACE GAP DETECTION (RECORDED HUMAN ERRORS)</div>', unsafe_allow_html=True)
    st.caption("Deterministic evaluation of recorded ticket attributes against regulatory SOP thresholds.")

    st.markdown(r"""
    - **POS_01**: Notes < 6 words, < 25 chars, or trivial match (`'ok'`, `'done'`, `'fp'`).
    - **POS_02**: High/Critical severity closed locally with `escalation_status != 'escalated'`.
    - **POS_03**: Repetitive templated notes (TF-IDF / verbatim duplication across $\ge 2$ distinct cases).
    - **POS_04**: Premature closure below minimum expected triage time for severity.
    """)

    # Filter controls
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        rule_filter = st.multiselect(
            "Filter Rules",
            options=["POS_01", "POS_02", "POS_03", "POS_04"],
            default=["POS_01", "POS_02", "POS_03", "POS_04"]
        )
    with f_col2:
        analyst_filter = st.multiselect(
            "Filter Analysts",
            options=sorted(df["analyst_id"].unique().tolist()),
            default=sorted(df["analyst_id"].unique().tolist())
        )
    with f_col3:
        sev_filter = st.multiselect(
            "Filter Alert Severity",
            options=["low", "medium", "high", "critical"],
            default=["low", "medium", "high", "critical"]
        )

    filtered_pos = [
        p for p in pos_findings
        if p.rule_id in rule_filter and p.analyst_id in analyst_filter and p.severity.lower() in sev_filter
    ]

    st.markdown(f"**MATCHED POSITIVE GAP RECORDS: {len(filtered_pos)}**")

    if filtered_pos:
        pos_table_data = []
        for p in filtered_pos:
            pos_table_data.append({
                "Rule": p.rule_id,
                "Rule Name": p.rule_name,
                "Case ID": p.case_id,
                "Alert ID": p.alert_id,
                "Analyst": p.analyst_id,
                "Entity": p.entity_id,
                "Severity": p.severity.upper(),
                "Time to Close": f"{p.time_to_close_minutes:.1f}m",
                "Audit Evidence": p.evidence_detail,
                "Notes Snippet": p.investigation_notes_snippet,
            })
        
        st.dataframe(
            pd.DataFrame(pos_table_data),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No positive space findings match the active filter criteria.")


# =========================================================
# TAB 3: NEGATIVE SPACE FINDINGS
# =========================================================
with tab_negative:
    st.markdown('<div class="section-header">// NEGATIVE SPACE GAP DETECTION (UNSEEN & STATISTICAL ANOMALIES)</div>', unsafe_allow_html=True)
    st.caption("Peer benchmarking algorithms detecting missing indicators, statistical throughput outliers, silence windows, and evidence omissions.")

    neg_tab1, neg_tab2, neg_tab3, neg_tab4 = st.tabs([
        "[NEG_01] Telemetry Blindspots",
        "[NEG_02] Volume Outliers",
        "[NEG_03] Silence Windows",
        "[NEG_04] Evidence Omission"
    ])

    with neg_tab1:
        st.markdown("**NEG_01: CRITICAL SECTOR ENTITY TELEMETRY BLINDSPOTS**")
        st.caption("Entities with zero or near-zero alerts in categories consistently observed across peer entities.")

        blindspots = [n for n in neg_findings if n.rule_id == "NEG_01"]
        if blindspots:
            b_rows = []
            for b in blindspots:
                b_rows.append({
                    "Entity ID": b.target_id,
                    "Threat Category Missing": b.affected_alert_type,
                    "Observed Count": int(b.observed_value),
                    "Peer Cohort Average": f"{b.baseline_value:.1f}",
                    "Severity": b.gap_severity,
                    "Audit Analysis": b.evidence_detail,
                })
            st.dataframe(pd.DataFrame(b_rows), use_container_width=True, hide_index=True)
            
            st.markdown("**ENTITY VS THREAT CATEGORY ALERT CROSS-TABULATION MATRIX**")
            ct_df = pd.crosstab(df["entity_id"], df["alert_type"])
            st.dataframe(ct_df, use_container_width=True)
        else:
            st.info("No telemetry blindspots detected across entities.")

    with neg_tab2:
        st.markdown("**NEG_02: ANALYST CASE THROUGHPUT STATISTICAL LOW OUTLIERS**")
        st.caption("Detects human analysts whose processed case count deviates below cohort Z-score (Z <= -1.40) or IQR lower fence.")

        benchmarking = PeerBenchmarkingEngine()
        vol_bench = benchmarking.compute_analyst_volume_benchmarks(df)
        
        v_col1, v_col2, v_col3 = st.columns(3)
        with v_col1:
            st.metric("Cohort Mean Volume", f"{vol_bench['mean']} cases")
        with v_col2:
            st.metric("Cohort Std Deviation", f"{vol_bench['std']}")
        with v_col3:
            st.metric("IQR Lower Fence", f"{vol_bench['iqr_lower_fence']} cases")

        anl_vol_data = []
        for aid, st_dict in vol_bench["analyst_stats"].items():
            anl_vol_data.append({
                "Analyst ID": aid,
                "Cases Handled": st_dict["count"],
                "Z-Score": st_dict["z_score"],
                "Statistical Status": "[FLAGGED: LOW_OUTLIER]" if st_dict["is_iqr_outlier"] else "[NORMAL]",
            })
        
        st.dataframe(pd.DataFrame(anl_vol_data), use_container_width=True, hide_index=True)

    with neg_tab3:
        st.markdown("**NEG_03: ENTITY TELEMETRY SILENCE & BLACKOUT WINDOWS**")
        st.caption("Identifies multi-day windows where an entity's alert throughput plummeted to zero or >80% below baseline.")

        silence_findings = [n for n in neg_findings if n.rule_id == "NEG_03"]
        if silence_findings:
            s_rows = []
            for s in silence_findings:
                s_rows.append({
                    "Entity ID": s.target_id,
                    "Blackout Time Window": s.time_window,
                    "Window Alert Count": int(s.observed_value),
                    "Expected Baseline": s.baseline_value,
                    "Severity": s.gap_severity,
                    "Audit Analysis": s.evidence_detail,
                })
            st.dataframe(pd.DataFrame(s_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No entity silence windows detected.")

    with neg_tab4:
        st.markdown("**NEG_04: SYSTEMATIC FORENSIC EVIDENCE OMISSION**")
        st.caption("Identifies analysts with an unusually high percentage of closed cases lacking complete forensic evidence flags.")

        ev_bench = benchmarking.compute_analyst_evidence_rates(df)
        st.metric("Cohort Average Missing Evidence Rate", f"{ev_bench['cohort_missing_evidence_rate']:.1%}")

        ev_rows = []
        for aid, edata in ev_bench["analyst_rates"].items():
            is_flagged = edata["missing_evidence_rate"] >= 0.30
            ev_rows.append({
                "Analyst ID": aid,
                "Total Cases": edata["total_cases"],
                "Missing Evidence Cases": edata["missing_evidence_cases"],
                "Missing Evidence %": f"{edata['missing_evidence_rate']:.1%}",
                "Compliance Status": "[FLAGGED: HIGH_OMISSION]" if is_flagged else "[COMPLIANT]",
            })
        st.dataframe(pd.DataFrame(ev_rows), use_container_width=True, hide_index=True)


# =========================================================
# TAB 4: SUPERVISORY CASE DRILL-DOWN & AI BRIEFS (OLLAMA)
# =========================================================
with tab_drilldown:
    st.markdown('<div class="section-header">// FORENSIC DRILL-DOWN & LOCAL AI EXPLAINABILITY</div>', unsafe_allow_html=True)
    st.caption("Inspect individual case records and generate plain-language supervisory briefings using local Ollama LLM.")

    drill_type = st.radio("Select Inspection Scope", ["Individual Security Case", "Human SOC Analyst Profile", "Critical Sector Entity Profile"], horizontal=True)

    if drill_type == "Individual Security Case":
        case_options = df["case_id"].tolist()
        default_case = pos_findings[0].case_id if pos_findings else case_options[0]
        selected_case_id = st.selectbox("Select Case ID to Inspect", options=case_options, index=case_options.index(default_case) if default_case in case_options else 0)
        
        case_row = df[df["case_id"] == selected_case_id].iloc[0].to_dict()
        case_pos = [p for p in pos_findings if p.case_id == selected_case_id]

        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("**RAW SOC CASE RECORD (INGESTED)**")
            st.json({
                "case_id": case_row["case_id"],
                "alert_id": case_row["alert_id"],
                "timestamp": str(case_row["timestamp"]),
                "entity_id": case_row["entity_id"],
                "analyst_id": case_row["analyst_id"],
                "severity": case_row["severity"],
                "alert_type": case_row["alert_type"],
                "action_taken": case_row["action_taken"],
                "investigation_notes": case_row["investigation_notes"],
                "escalation_status": case_row["escalation_status"],
                "evidence_completeness_flag": bool(case_row["evidence_completeness_flag"]),
                "time_to_close_minutes": case_row["time_to_close_minutes"],
                "closure_time": str(case_row["closure_time"]),
            })

        with c2:
            st.markdown("**TRIGGERED SUPERVISORY AUDIT RULES**")
            if case_pos:
                for p in case_pos:
                    st.error(f"[{p.rule_id}] {p.rule_name} (SEVERITY: {p.gap_severity})\n\n{p.evidence_detail}")
            else:
                st.info("No positive space rule violations triggered for this specific case.")

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("EXECUTE CASE EXPLAINABILITY SYNTHESIS (OLLAMA)", use_container_width=True, type="primary"):
                with st.spinner("Executing local LLM explainability in-process..."):
                    explainer_engine = OllamaExplainabilityEngine(model_name=selected_model if selected_model != "offline-fallback" else None)
                    force_fb = (selected_model == "offline-fallback" or selected_model == "deterministic-fallback")
                    rationale = explainer_engine.generate_case_rationale(case_row, pos_findings, force_fallback=force_fb)
                    
                    st.markdown(f"""
                    <div class="ai-brief-box">
                        <div class="brief-header">// NCIIPC SUPERVISORY AUDIT DOSSIER [ENGINE: {rationale.generated_by}]</div>
                        <p><strong>Behavioral Summary:</strong> {rationale.summary}</p>
                        <strong>Forensic Evidence Points:</strong>
                        <ul>
                            {''.join(f'<li>{pt}</li>' for pt in rationale.evidence_points)}
                        </ul>
                        <strong>Recommended Supervisory Actions:</strong>
                        <ul>
                            {''.join(f'<li>{act}</li>' for act in rationale.supervisory_action_items)}
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

    elif drill_type == "Human SOC Analyst Profile":
        selected_aid = st.selectbox("Select Analyst ID", options=[a.target_id for a in analyst_scores])
        score_obj = next(a for a in analyst_scores if a.target_id == selected_aid)
        
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown(f"**SUPERVISORY AUDIT PROFILE: {selected_aid}**")
            st.metric("Composite Risk Index", f"{score_obj.composite_risk_score}/100", score_obj.supervisory_tier)
            st.write(f"**Total Cases Handled:** {score_obj.total_cases_handled} | **Positive Gaps:** {score_obj.positive_gap_count} | **Negative Anomalies:** {score_obj.negative_gap_count}")
            st.write(f"**Unescalated High/Critical Cases:** {score_obj.unescalated_critical_count}")
            st.write(f"**Missing Evidence Attachments:** {score_obj.missing_evidence_count}")
            st.write(f"**Average Time to Close:** {score_obj.avg_close_time_minutes} minutes")
            st.write("**Primary Identified Concerns:**")
            for c in score_obj.primary_concerns:
                st.markdown(f"- `[FLAG]` {c}")

        with c2:
            st.markdown("**SUPERVISORY APPRAISAL DOSSIER**")
            if st.button("GENERATE ANALYST PERFORMANCE DOSSIER (OLLAMA)", use_container_width=True, type="primary"):
                with st.spinner("Synthesizing analyst supervisory appraisal..."):
                    explainer_engine = OllamaExplainabilityEngine(model_name=selected_model if selected_model != "offline-fallback" else None)
                    force_fb = (selected_model == "offline-fallback" or selected_model == "deterministic-fallback")
                    rationale = explainer_engine.generate_analyst_briefing(score_obj, pos_findings, neg_findings, force_fallback=force_fb)
                    
                    st.markdown(f"""
                    <div class="ai-brief-box">
                        <div class="brief-header">// ANALYST APPRAISAL DOSSIER [ENGINE: {rationale.generated_by}]</div>
                        <p><strong>Executive Summary:</strong> {rationale.summary}</p>
                        <strong>Core Deficiencies & Evidence:</strong>
                        <ul>
                            {''.join(f'<li>{pt}</li>' for pt in rationale.evidence_points)}
                        </ul>
                        <strong>Supervisory Directives:</strong>
                        <ul>
                            {''.join(f'<li>{act}</li>' for act in rationale.supervisory_action_items)}
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

    else:
        selected_eid = st.selectbox("Select Critical Sector Entity ID", options=[e.target_id for e in entity_scores])
        score_obj = next(e for e in entity_scores if e.target_id == selected_eid)
        
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown(f"**ENTITY TELEMETRY ASSESSMENT: {selected_eid}**")
            st.metric("Vulnerability Risk Score", f"{score_obj.composite_risk_score}/100", score_obj.supervisory_tier)
            st.write(f"**Total Ingested Alerts:** {score_obj.total_cases_handled}")
            st.write(f"**Telemetry Blindspots Score:** {score_obj.score_breakdown.get('blindspot_score', 0):.0f}/100")
            st.write(f"**Silence Blackout Penalty:** {score_obj.score_breakdown.get('silence_penalty', 0):.0f}/100")
            st.write("**Primary Telemetry Flags:**")
            for c in score_obj.primary_concerns:
                st.markdown(f"- `[TELEMETRY_GAP]` {c}")

        with c2:
            st.markdown("**NCIIPC SECTOR COORDINATION DIRECTIVE**")
            if st.button("GENERATE ENTITY TELEMETRY DIRECTIVE (OLLAMA)", use_container_width=True, type="primary"):
                with st.spinner("Synthesizing entity telemetry directive..."):
                    explainer_engine = OllamaExplainabilityEngine(model_name=selected_model if selected_model != "offline-fallback" else None)
                    force_fb = (selected_model == "offline-fallback" or selected_model == "deterministic-fallback")
                    rationale = explainer_engine.generate_entity_briefing(score_obj, neg_findings, force_fallback=force_fb)
                    
                    st.markdown(f"""
                    <div class="ai-brief-box">
                        <div class="brief-header">// NCIIPC ENTITY TELEMETRY DIRECTIVE [ENGINE: {rationale.generated_by}]</div>
                        <p><strong>Surveillance Exposure:</strong> {rationale.summary}</p>
                        <strong>Missing Telemetry Evidence:</strong>
                        <ul>
                            {''.join(f'<li>{pt}</li>' for pt in rationale.evidence_points)}
                        </ul>
                        <strong>Supervisory Directives:</strong>
                        <ul>
                            {''.join(f'<li>{act}</li>' for act in rationale.supervisory_action_items)}
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)


# =========================================================
# TAB 5: RULES, THRESHOLDS & AUDIT EXPORT
# =========================================================
with tab_config:
    st.markdown('<div class="section-header">// SYSTEM CONFIGURATION & AUDIT EXPORT</div>', unsafe_allow_html=True)
    
    c_cfg1, c_cfg2 = st.columns([1, 1])
    
    with c_cfg1:
        st.markdown("**ACTIVE DETECTION THRESHOLDS (`rules.yaml`)**")
        cfg = load_rules_config()
        st.code(yaml.dump(cfg, default_flow_style=False), language="yaml")

    with c_cfg2:
        st.markdown("**OFFICIAL SUPERVISORY AUDIT REPORT EXPORT**")
        st.markdown("""
        Exports an official NCIIPC Supervisory Audit Dossier containing complete positive and negative space findings,
        analyst performance leaderboards, and telemetry coverage matrices.
        """)
        
        # Build Markdown Audit Report
        report_md = f"""# NCIIPC / NTRO SUPERVISORY ASSESSMENT REPORT (SAT-SA)
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Scope: {total_cases} Case-Management Records across {len(entity_scores)} Critical Sector Entities (CSEs) and {len(analyst_scores)} Human Analysts.
Assessment Window: 90 Days

---

## 1. Executive Summary
- Total Positive Space Violations (Direct Human Errors): {total_pos}
- Total Negative Space Anomalies (Missing Telemetry & Outliers): {total_neg}
- Analysts Requiring Supervisory Review: {critical_analysts}
- Entities with Telemetry Blindspots: {blindspot_entities}

---

## 2. Human Analyst Supervisory Priority Ranking
| Analyst ID | Composite Risk | Tier | Total Cases | Positive Gaps | Negative Anomalies | Unesc. Critical |
|---|---|---|---|---|---|---|
"""
        for a in analyst_scores:
            report_md += f"| {a.target_id} | {a.composite_risk_score:.1f} | {a.supervisory_tier} | {a.total_cases_handled} | {a.positive_gap_count} | {a.negative_gap_count} | {a.unescalated_critical_count} |\n"

        report_md += f"""
---

## 3. Critical Sector Entity Vulnerability Ranking
| Entity ID | Composite Risk | Tier | Cases | Primary Concern |
|---|---|---|---|---|
"""
        for e in entity_scores:
            concern = e.primary_concerns[0] if e.primary_concerns else "Compliant"
            report_md += f"| {e.target_id} | {e.composite_risk_score:.1f} | {e.supervisory_tier} | {e.total_cases_handled} | {concern} |\n"

        st.download_button(
            label="DOWNLOAD AUDIT REPORT (.MD)",
            data=report_md,
            file_name=f"SAT_SA_Audit_Report_{datetime.now().strftime('%Y%m%d')}.md",
            mime="text/markdown",
            use_container_width=True
        )

        st.download_button(
            label="EXPORT CASE DATASET (.CSV)",
            data=df.to_csv(index=False),
            file_name=f"SAT_SA_Case_Records_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )

        # JSON Export
        export_payload = {
            "metadata": {
                "system": "SAT-SA",
                "authority": "NCIIPC / NTRO",
                "problem_statement": "SIH26157",
                "generated_at": datetime.now().isoformat(),
                "total_records": total_cases,
            },
            "analyst_scores": [a.model_dump() for a in analyst_scores],
            "entity_scores": [e.model_dump() for e in entity_scores],
            "positive_findings_count": total_pos,
            "negative_findings_count": total_neg,
        }
        
        st.download_button(
            label="EXPORT METRICS & TELEMETRY (.JSON)",
            data=json.dumps(export_payload, indent=2),
            file_name=f"SAT_SA_Metrics_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json",
            use_container_width=True
        )
