import sys
from core.data_generator import get_synthetic_dataframe
from core.positive_gap_engine import PositiveGapEngine
from core.negative_gap_engine import NegativeGapEngine
from core.risk_scoring import SupervisoryRiskScoringEngine
from core.explainability import OllamaExplainabilityEngine

def main():
    print("=" * 70)
    print("1. GENERATING SYNTHETIC DATASET...")
    df = get_synthetic_dataframe(num_records=140, seed=42)
    print(f"   Generated {len(df)} records across {df['entity_id'].nunique()} entities and {df['analyst_id'].nunique()} analysts.")

    print("\n2. RUNNING POSITIVE SPACE GAP ENGINE...")
    pos_engine = PositiveGapEngine()
    pos_findings = pos_engine.run_all_positive_rules(df)
    print(f"   Found {len(pos_findings)} Positive Space Findings.")
    pos_rule_counts = {}
    for p in pos_findings:
        pos_rule_counts[p.rule_id] = pos_rule_counts.get(p.rule_id, 0) + 1
    for r_id, cnt in sorted(pos_rule_counts.items()):
        print(f"   - {r_id}: {cnt} instances")

    print("\n3. RUNNING NEGATIVE SPACE GAP ENGINE...")
    neg_engine = NegativeGapEngine()
    neg_findings = neg_engine.run_all_negative_rules(df)
    print(f"   Found {len(neg_findings)} Negative Space Findings.")
    neg_rule_counts = {}
    for n in neg_findings:
        neg_rule_counts[n.rule_id] = neg_rule_counts.get(n.rule_id, 0) + 1
    for r_id, cnt in sorted(neg_rule_counts.items()):
        print(f"   - {r_id}: {cnt} instances")

    print("\n4. COMPUTING SUPERVISORY RISK SCORES...")
    risk_engine = SupervisoryRiskScoringEngine()
    analyst_scores = risk_engine.score_analysts(df, pos_findings, neg_findings)
    entity_scores = risk_engine.score_entities(df, pos_findings, neg_findings)

    print("\n   TOP FLAGGED ANALYSTS:")
    for a in analyst_scores:
        print(f"   - {a.target_id:<16} | Risk: {a.composite_risk_score:5.1f} | Tier: {a.supervisory_tier:<30} | Pos: {a.positive_gap_count} | Neg: {a.negative_gap_count}")

    print("\n   TOP FLAGGED ENTITIES:")
    for e in entity_scores:
        print(f"   - {e.target_id:<16} | Risk: {e.composite_risk_score:5.1f} | Tier: {e.supervisory_tier:<30} | Blindspots: {e.score_breakdown.get('blindspot_score', 0)}")

    print("\n5. TESTING OLLAMA EXPLAINABILITY LAYER...")
    explainer = OllamaExplainabilityEngine()
    is_ollama_up = explainer.is_ollama_available()
    print(f"   Ollama available: {is_ollama_up} (Models: {explainer.list_local_models()})")

    # Test case briefing
    sample_case = df[df["case_id"] == pos_findings[0].case_id].iloc[0].to_dict() if pos_findings else df.iloc[0].to_dict()
    case_brief = explainer.generate_case_rationale(sample_case, pos_findings)
    print(f"\n   [Sample Case Rationale for {case_brief.target_id} ({case_brief.generated_by})]")
    print(f"   Summary: {case_brief.summary}")
    print(f"   Evidence: {case_brief.evidence_points}")
    print(f"   Actions: {case_brief.supervisory_action_items}")

    # Test analyst briefing
    top_analyst = analyst_scores[0]
    anl_brief = explainer.generate_analyst_briefing(top_analyst, pos_findings, neg_findings)
    print(f"\n   [Sample Analyst Brief for {anl_brief.target_id} ({anl_brief.generated_by})]")
    print(f"   Summary: {anl_brief.summary}")
    print(f"   Evidence: {anl_brief.evidence_points}")
    print(f"   Actions: {anl_brief.supervisory_action_items}")

    print("\n" + "=" * 70)
    print("ALL CORE MODULE TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
