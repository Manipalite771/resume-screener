"""Pure validation and decision logic for structured resume screening."""

from __future__ import annotations

import json
import re
from typing import Any


GENAI_CRITERIA = {
    "hands_on_ai": ("Hands-on AI implementation", 2),
    "technical_depth": ("Technical solution depth", 2),
    "product_program_delivery": ("Product/program delivery", 1),
    "stakeholder_leadership": ("Stakeholder leadership", 1),
    "regulated_life_sciences": ("Regulated/life-sciences experience", 1),
    "pedigree_or_complexity": ("Pedigree or equivalent complex-delivery signal", 1),
}

EVIDENCE_STATUSES = {"demonstrated", "partial", "not_demonstrated"}
GENAI_MINIMUM_YEARS = 6
GENAI_MINIMUM_SCORE = 6


def _json_payload(response: str) -> dict[str, Any]:
    if not response or not response.strip():
        raise ValueError("The model returned an empty assessment.")

    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", response, re.DOTALL | re.IGNORECASE)
    raw = fenced.group(1) if fenced else response.strip()

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("The model assessment did not contain valid JSON.") from None
        try:
            payload = json.loads(raw[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"The model assessment contained invalid JSON: {exc.msg}.") from None

    if not isinstance(payload, dict):
        raise ValueError("The model assessment must be a JSON object.")
    return payload


def _clean_string_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field_name} must be a list of strings.")
    return [item.strip() for item in value if item.strip()]


def parse_genai_analysis(response: str) -> dict[str, Any]:
    """Parse and validate the model's structured GenAI assessment.

    Invalid or incomplete model output raises instead of silently producing a
    favorable hiring decision.
    """
    payload = _json_payload(response)

    experience = payload.get("experience")
    if not isinstance(experience, dict):
        raise ValueError("The assessment is missing the experience section.")

    years = experience.get("total_professional_years")
    if isinstance(years, bool) or not isinstance(years, (int, float)) or years < 0:
        raise ValueError("total_professional_years must be a non-negative number.")

    experience_status = experience.get("evidence_status")
    if experience_status not in EVIDENCE_STATUSES:
        raise ValueError("The experience evidence_status is invalid.")

    experience_evidence = experience.get("evidence")
    if not isinstance(experience_evidence, str) or not experience_evidence.strip():
        raise ValueError("The assessment is missing experience evidence.")

    raw_criteria = payload.get("criteria")
    if not isinstance(raw_criteria, dict):
        raise ValueError("The assessment is missing the criteria section.")

    criteria: dict[str, dict[str, Any]] = {}
    for key, (_, maximum) in GENAI_CRITERIA.items():
        item = raw_criteria.get(key)
        if not isinstance(item, dict):
            raise ValueError(f"The assessment is missing criterion: {key}.")

        score = item.get("score")
        if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= maximum:
            raise ValueError(f"The score for {key} must be an integer from 0 to {maximum}.")

        status = item.get("evidence_status")
        if status not in EVIDENCE_STATUSES:
            raise ValueError(f"The evidence_status for {key} is invalid.")
        if score == maximum and status != "demonstrated":
            raise ValueError(f"A full score for {key} requires demonstrated evidence.")
        if score > 0 and status == "not_demonstrated":
            raise ValueError(f"A positive score for {key} requires supporting evidence.")

        evidence = item.get("evidence")
        if not isinstance(evidence, str) or not evidence.strip():
            raise ValueError(f"The assessment is missing evidence for {key}.")

        criteria[key] = {
            "score": score,
            "evidence_status": status,
            "evidence": evidence.strip(),
        }

    summary = payload.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("The assessment is missing its summary.")

    return {
        "experience": {
            "total_professional_years": float(years),
            "evidence_status": experience_status,
            "evidence": experience_evidence.strip(),
        },
        "criteria": criteria,
        "key_strengths": _clean_string_list(payload.get("key_strengths"), "key_strengths"),
        "concerns_gaps": _clean_string_list(payload.get("concerns_gaps"), "concerns_gaps"),
        "summary": summary.strip(),
    }


def evaluate_genai_analysis(analysis: dict[str, Any], quality_verdict: str) -> dict[str, Any]:
    """Calculate the score and enforce non-negotiable GenAI hiring gates."""
    criteria = analysis["criteria"]
    years = analysis["experience"]["total_professional_years"]
    experience_status = analysis["experience"]["evidence_status"]

    role_fit_score = sum(criteria[key]["score"] for key in GENAI_CRITERIA)
    quality_penalty = 1 if str(quality_verdict).upper() == "FAIL" else 0
    final_score = max(0, role_fit_score - quality_penalty)

    gates = {
        "minimum_experience": years >= GENAI_MINIMUM_YEARS and experience_status == "demonstrated",
        "hands_on_ai": (
            criteria["hands_on_ai"]["score"] == 2
            and criteria["hands_on_ai"]["evidence_status"] == "demonstrated"
        ),
        "technical_depth": (
            criteria["technical_depth"]["score"] >= 1
            and criteria["technical_depth"]["evidence_status"] != "not_demonstrated"
        ),
        "minimum_score": final_score >= GENAI_MINIMUM_SCORE,
    }

    gate_labels = {
        "minimum_experience": "At least 6 years of clearly demonstrated professional experience",
        "hands_on_ai": "Clearly demonstrated hands-on GenAI implementation experience",
        "technical_depth": "At least partial technical solution depth",
        "minimum_score": "Final score of at least 6/8",
    }

    return {
        "role_fit_score": role_fit_score,
        "quality_penalty": quality_penalty,
        "final_score": final_score,
        "gates": gates,
        "gate_failures": [gate_labels[key] for key, passed in gates.items() if not passed],
        "proceed": all(gates.values()),
    }
