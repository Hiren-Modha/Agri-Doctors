"""
Guardrail Category 7 — Human-in-the-Loop (HITL) Guardrails.

Ensures human review for critical decisions:
  - Escalation triggers when AI confidence is low or disease is unidentifiable
  - Chemical treatment recommendation review flag
  - Critical output review flag before download
  - Escalation to expert recommendation injection
"""

import re

# ── Escalation trigger conditions ─────────────────────────────────────────────
_LOW_CONFIDENCE_TERMS = re.compile(r"(?i)\b(low|uncertain|unclear|unknown|unable to determine)\b")
_CHEMICAL_TREATMENT_TERMS = re.compile(
    r"(?i)\b(fungicide|pesticide|herbicide|insecticide|bactericide|"
    r"chemical\s+treatment|spray|apply\s+\w+\s+(product|chemical))\b"
)
_HIGH_SEVERITY_TERMS = re.compile(
    r"(?i)\b(severe|critical|widespread|entire\s+(crop|field)|"
    r"complete\s+loss|total\s+loss|epidemic|outbreak)\b"
)


def hitl_escalation_check(diagnosis: dict, advisory: dict) -> dict:
    """
    Evaluate the combined agent outputs and return a HITL assessment dict.

    Returns a dict with:
        requires_human_review: bool
        escalation_reasons: list[str]
        escalation_level: "none" | "advisory" | "mandatory"
        expert_message: str    — message to display to the farmer
    """
    reasons = []
    escalation_level = "none"

    disease = diagnosis.get("disease", "")
    confidence = diagnosis.get("confidence", "")
    raw_advisory = advisory.get("raw_advisory", "")

    # 1. Low confidence or unidentifiable disease
    if _LOW_CONFIDENCE_TERMS.search(confidence) or "unable to determine" in disease.lower():
        reasons.append(
            "Disease could not be reliably identified (low confidence or 'Unable to determine'). "
            "A local expert must verify before any treatment is applied."
        )
        escalation_level = "mandatory"

    # 2. Chemical treatment recommended
    if _CHEMICAL_TREATMENT_TERMS.search(raw_advisory):
        reasons.append(
            "Chemical treatment options were mentioned in the advisory. "
            "Human review is recommended before purchasing or applying any chemicals. "
            "Always consult your local agricultural extension officer for approved products."
        )
        escalation_level = max(escalation_level, "advisory",
                               key=lambda v: ["none", "advisory", "mandatory"].index(v))

    # 3. High-severity language in diagnosis or advisory
    combined_text = (
        diagnosis.get("raw_report", "") + " " + raw_advisory
    )
    if _HIGH_SEVERITY_TERMS.search(combined_text):
        reasons.append(
            "Severe or widespread crop damage was indicated. "
            "Immediate consultation with an agricultural extension officer is strongly advised."
        )
        escalation_level = "mandatory"

    requires_review = escalation_level != "none"

    expert_message = ""
    if escalation_level == "mandatory":
        expert_message = (
            "HUMAN REVIEW REQUIRED: Based on this AI assessment, you should contact "
            "your local agricultural extension officer or a certified plant pathologist "
            "BEFORE taking any treatment action. Do not apply chemicals based solely on "
            "this AI-generated report."
        )
    elif escalation_level == "advisory":
        expert_message = (
            "EXPERT CONSULTATION RECOMMENDED: Some aspects of this advisory involve "
            "chemical treatments or uncertain findings. We recommend confirming with "
            "your local agricultural extension officer before proceeding."
        )

    return {
        "requires_human_review": requires_review,
        "escalation_level": escalation_level,
        "escalation_reasons": reasons,
        "expert_message": expert_message,
    }
