import unittest

from screening_logic import evaluate_genai_analysis, parse_genai_analysis


def valid_payload():
    return """{
      "experience": {
        "total_professional_years": 7,
        "evidence_status": "demonstrated",
        "evidence": "Seven years across dated full-time roles."
      },
      "criteria": {
        "hands_on_ai": {"score": 2, "evidence_status": "demonstrated", "evidence": "Shipped two RAG workflows and owned prompts and evaluation."},
        "technical_depth": {"score": 2, "evidence_status": "demonstrated", "evidence": "Designed retrieval, APIs, SQL analysis, and evaluation datasets."},
        "product_program_delivery": {"score": 1, "evidence_status": "demonstrated", "evidence": "Owned roadmap and delivery."},
        "stakeholder_leadership": {"score": 1, "evidence_status": "demonstrated", "evidence": "Led client and engineering workshops."},
        "regulated_life_sciences": {"score": 1, "evidence_status": "demonstrated", "evidence": "Delivered validated pharma workflows."},
        "pedigree_or_complexity": {"score": 1, "evidence_status": "demonstrated", "evidence": "Led a complex global enterprise rollout."}
      },
      "key_strengths": ["Hands-on RAG delivery"],
      "concerns_gaps": [],
      "summary": "Strong evidence-backed fit."
    }"""


class ParseGenAIAnalysisTests(unittest.TestCase):
    def test_parses_fenced_json(self):
        analysis = parse_genai_analysis(f"```json\n{valid_payload()}\n```")
        self.assertEqual(analysis["experience"]["total_professional_years"], 7.0)
        self.assertEqual(analysis["criteria"]["hands_on_ai"]["score"], 2)

    def test_rejects_missing_criterion(self):
        response = valid_payload().replace(
            '"technical_depth": {"score": 2, "evidence_status": "demonstrated", "evidence": "Designed retrieval, APIs, SQL analysis, and evaluation datasets."},',
            "",
        )
        with self.assertRaisesRegex(ValueError, "technical_depth"):
            parse_genai_analysis(response)

    def test_rejects_full_score_without_demonstrated_evidence(self):
        response = valid_payload().replace(
            '"hands_on_ai": {"score": 2, "evidence_status": "demonstrated"',
            '"hands_on_ai": {"score": 2, "evidence_status": "partial"',
        )
        with self.assertRaisesRegex(ValueError, "full score for hands_on_ai"):
            parse_genai_analysis(response)


class EvaluateGenAIAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.analysis = parse_genai_analysis(valid_payload())

    def test_strong_candidate_proceeds(self):
        decision = evaluate_genai_analysis(self.analysis, "PASS")
        self.assertTrue(decision["proceed"])
        self.assertEqual(decision["final_score"], 8)

    def test_quality_failure_applies_penalty(self):
        decision = evaluate_genai_analysis(self.analysis, "FAIL")
        self.assertTrue(decision["proceed"])
        self.assertEqual(decision["final_score"], 7)
        self.assertEqual(decision["quality_penalty"], 1)

    def test_hands_on_ai_is_a_hard_gate(self):
        self.analysis["criteria"]["hands_on_ai"]["score"] = 1
        self.analysis["criteria"]["hands_on_ai"]["evidence_status"] = "partial"
        decision = evaluate_genai_analysis(self.analysis, "PASS")
        self.assertFalse(decision["proceed"])
        self.assertFalse(decision["gates"]["hands_on_ai"])

    def test_six_years_is_a_hard_gate(self):
        self.analysis["experience"]["total_professional_years"] = 5.9
        decision = evaluate_genai_analysis(self.analysis, "PASS")
        self.assertFalse(decision["proceed"])
        self.assertFalse(decision["gates"]["minimum_experience"])

    def test_technical_depth_is_a_hard_gate(self):
        self.analysis["criteria"]["technical_depth"]["score"] = 0
        self.analysis["criteria"]["technical_depth"]["evidence_status"] = "not_demonstrated"
        decision = evaluate_genai_analysis(self.analysis, "PASS")
        self.assertFalse(decision["proceed"])
        self.assertFalse(decision["gates"]["technical_depth"])

    def test_quality_penalty_can_drop_candidate_below_threshold(self):
        self.analysis["criteria"]["technical_depth"]["score"] = 1
        self.analysis["criteria"]["pedigree_or_complexity"]["score"] = 0
        self.analysis["criteria"]["pedigree_or_complexity"]["evidence_status"] = "not_demonstrated"
        decision = evaluate_genai_analysis(self.analysis, "FAIL")
        self.assertFalse(decision["proceed"])
        self.assertEqual(decision["final_score"], 5)
        self.assertFalse(decision["gates"]["minimum_score"])


if __name__ == "__main__":
    unittest.main()
