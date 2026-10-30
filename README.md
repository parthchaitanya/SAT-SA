# Supervisory Analytics Tool for SOC Assessment (SAT-SA)

**SAT-SA** is an air-gapped, supervisory intelligence proof-of-concept designed for **NCIIPC supervisors** to evaluate already-generated SOC case-management records across **Critical Sector Entities (CSEs)**.

Instead of analyzing AI agents or doing real-time threat detection, SAT-SA focuses on **HUMAN ANALYST BEHAVIOR**:
1. **Positive Space Gaps (Visible Human Errors)**: Unescalated critical alerts, blank/minimal investigation notes, copy-pasted boilerplate templates, and impossibly fast closures.
2. **Negative Space Gaps (Missing Indicators & Telemetry Blindspots)**: CSEs missing expected alert categories, statistical low-throughput analysts (Z-score/IQR outliers), telemetry silence/blackout windows, and systematic omission of forensic evidence flags.
3. **Supervisory Risk Scoring**: Composite multi-factor risk index (0–100) prioritizing analysts and entities needing manual supervisory intervention.
4. **Local Ollama Explainability**: Plain-language supervisory briefings generated locally via Ollama (`llama3.2:latest` / `llama3.2:3b`), with an air-gapped deterministic fallback.

---

## 🏗️ Architecture & Philosophy

```
c:\Users\HP\OneDrive\Desktop\SIH\
├── config\
│   └── rules.yaml                 # Configurable thresholds, z-score limits, and scoring weights
├── core\
│   ├── schema.py                  # Pydantic data models for records, findings, and scores
│   ├── data_generator.py          # 100-150 synthetic records across 10 CSEs & 7 analysts (~25% injected gaps)
│   ├── positive_gap_engine.py     # Rule-based positive gap detection (POS_01 to POS_04)
│   ├── negative_gap_engine.py     # Negative space gap detection (NEG_01 to NEG_04)
│   ├── peer_benchmarking.py       # Peer baseline statistical computations (z-score, IQR, sector-level distributions)
│   ├── risk_scoring.py            # Composite multi-factor supervisory risk scoring (0-100) & ranking
│   └── explainability.py          # Local Ollama LLM rationale generation + structured fallback synthesizer
├── app.py                         # Single Streamlit Cyber-Command Dashboard (In-process execution)
├── test_pipeline.py               # Standalone end-to-end pipeline test script
├── requirements.txt               # Dependencies
└── README.md                      # Documentation & instructions
```

- **In-Process Python**: All analytical engines run directly inside the Streamlit process. No FastAPI, no separate backend server, no REST overhead.
- **Air-Gapped Guarantee**: 100% offline. Zero external network calls.

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.10+
- (Optional but recommended) [Ollama](https://ollama.com/) installed locally for LLM explainability.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Setting Up Ollama (Local LLM)
If you wish to use the local LLM explainability feature:
```bash
# Pull the recommended model
ollama pull llama3.2:latest
# or
ollama pull llama3.2:3b
```
> **Note**: If Ollama is not installed or running, SAT-SA automatically falls back to an offline deterministic synthesizer without interruption or errors.

### 4. Run the Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📊 Core Analytical Modules & Rules

### 1. Positive Space Gap Engine (Direct Mistakes)
| Rule ID | Name | Description |
|---|---|---|
| **POS_01** | Minimal Investigation Notes | Notes < 6 words or < 25 chars or trivial non-informative keywords ("ok", "done", "fp"). |
| **POS_02** | Unescalated High/Critical Severity | High or Critical severity alerts closed locally without CIRT / L2 escalation. |
| **POS_03** | Templated / Boilerplate Notes | Reusing identical/near-identical template text verbatim across multiple distinct cases. |
| **POS_04** | Premature / Rushed Closure | `time_to_close_minutes` below minimum realistic triage duration for that severity. |

### 2. Negative Space Gap Engine (Missing Indicators & Anomalies)
| Rule ID | Name | Description |
|---|---|---|
| **NEG_01** | Telemetry Blindspots | Entities with zero/near-zero alerts in categories consistently observed across peer entities. |
| **NEG_02** | Analyst Volume Low Outliers | Analysts whose case volume is a statistical outlier vs peer group ($Z < -1.4$ or below IQR fence). |
| **NEG_03** | Entity Silence Windows | Multi-day periods where entity alert volume drops to zero or >80% below baseline. |
| **NEG_04** | Evidence Omission Patterns | Analysts with unusually high percentage ($\ge 30\%$) of closed cases lacking forensic evidence. |

### 3. Supervisory Risk Scoring
Composite risk index ($0 - 100$) dynamically weights:
- Positive Gap Ratio (35%)
- Negative Space Impact (25%)
- Unescalated Critical Alert Ratio (20%)
- Missing Evidence Ratio (10%)
- Volume / Workload Penalty (10%)

---

## 🧪 Verification & Standalone Test

To verify the analytical engines and Ollama connection standalone:
```bash
python test_pipeline.py
```

---