"""
Guardrail Category 2 — Output Guardrails.

Validates AI-generated responses BEFORE displaying them to users:
  - Empty / too-short response detection
  - Hallucination marker detection (confident claims with no evidence)
  - Fabricated dosage detection (output safety)
  - Toxicity / harmful content keyword filtering
  - Sensitive information filtering (credentials, PII leakage)
  - Response format validation (expected sections present)
"""

import re

# ── Toxicity / harmful content ────────────────────────────────────────────────
_TOXIC_PATTERNS = re.compile(
    r"(?i)\b(kill\s+yourself|hate\s+you|die\s+slowly|"
    r"you\s+are\s+worthless|go\s+to\s+hell|stupid\s+(farmer|idiot)|"
    r"bomb|terrorist|weapon|murder|suicide)\b"
)

# ── Hallucination markers ─────────────────────────────────────────────────────
# Signs the model fabricated disease names or made confident claims without evidence
_HALLUCINATION_MARKERS = re.compile(
    r"(?i)(definitely\s+caused\s+by|100\s*%\s+certain|"
    r"I\s+guarantee\s+this\s+is|this\s+is\s+absolutely|"
    r"no\s+doubt\s+it\s+is|confirmed\s+diagnosis)"
)

# ── Fabricated dosage patterns ────────────────────────────────────────────────
# Catches specific numeric dosage claims the model must never invent
_DOSAGE_PATTERN = re.compile(
    r"(?i)\b(\d+(\.\d+)?\s*(ml|g|kg|litre|liter|oz|tbsp|tsp|cc|mg)\s*"
    r"(per|/)\s*(litre|liter|l|gallon|acre|hectare|plant|kg|tank))\b"
)

# ── Sensitive information leakage ─────────────────────────────────────────────
_SENSITIVE_PATTERNS = re.compile(
    r"(?i)(api[_\s]?key|password|secret|token|bearer\s+[a-z0-9\-_]{10,}|"
    r"sk-[a-z0-9]{20,}|access[_\s]?key|private[_\s]?key)"
)

# ── Required sections for each agent ─────────────────────────────────────────
_DIAGNOSIS_REQUIRED_SECTIONS = ["SUSPECTED DISEASE", "OBSERVED SYMPTOMS", "CONFIDENCE LEVEL"]
_ADVISORY_REQUIRED_SECTIONS  = ["IMMEDIATE ACTIONS", "SAFETY WARNINGS"]

_MIN_RESPONSE_CHARS = 80


def validate_diagnosis_output(text: str) -> tuple[bool, str, list]:
    """
    Validate the Pathologist Agent's raw LLM output.

    Returns:
        (passed: bool, reason: str, warnings: list[str])
        - passed=True means display is allowed; warnings are shown as soft alerts.
        - passed=False blocks display entirely.
    """
    warnings = []
    text = text.strip()

    # 1. Empty / trivially short response — hard block
    if len(text) < _MIN_RESPONSE_CHARS:
        return False, (
            "The Pathologist Agent returned an unusually short response. "
            "This may indicate a model error. Please try again."
        ), []

    # 2. Sensitive information leakage — hard block
    if _SENSITIVE_PATTERNS.search(text):
        return False, (
            "The agent response was blocked because it may contain sensitive information. "
            "Please try again."
        ), []

    # 3. Toxicity — hard block
    if _TOXIC_PATTERNS.search(text):
        return False, (
            "The agent response contained inappropriate content and was blocked."
        ), []

    # 4. Hallucination markers — soft warning (do not block)
    if _HALLUCINATION_MARKERS.search(text):
        warnings.append(
            "The diagnosis contains highly confident language. "
            "AI diagnoses are preliminary — always verify with a qualified plant pathologist."
        )

    # 5. Format validation — expected sections present (soft warning only for diagnosis
    #    because the LLM may also legitimately return UNCLEAR IMAGE)
    is_unclear = re.search(r"(?i)UNCLEAR IMAGE\s*:", text)
    if not is_unclear:
        missing = [s for s in _DIAGNOSIS_REQUIRED_SECTIONS
                   if not re.search(rf"(?i){re.escape(s)}", text)]
        if missing:
            warnings.append(
                f"The diagnosis response is missing expected sections: "
                f"{', '.join(missing)}. Some information may not display correctly."
            )

    return True, "", warnings


def validate_advisory_output(text: str) -> tuple[bool, str, list]:
    """
    Validate the Extension Officer Agent's raw LLM output.

    Returns:
        (passed: bool, reason: str, warnings: list[str])
    """
    warnings = []
    text = text.strip()

    # 1. Empty / trivially short response — hard block
    if len(text) < _MIN_RESPONSE_CHARS:
        return False, (
            "The Extension Officer Agent returned an unusually short response. "
            "Please try again."
        ), []

    # 2. Sensitive information leakage — hard block
    if _SENSITIVE_PATTERNS.search(text):
        return False, (
            "The advisory response was blocked because it may contain sensitive information."
        ), []

    # 3. Toxicity — hard block
    if _TOXIC_PATTERNS.search(text):
        return False, (
            "The advisory response contained inappropriate content and was blocked."
        ), []

    # 4. Fabricated dosage — HARD BLOCK (safety-critical rule)
    dosage_match = _DOSAGE_PATTERN.search(text)
    if dosage_match:
        return False, (
            f"The advisory was blocked because it contained a specific chemical dosage "
            f"('{dosage_match.group()}') that the AI must not invent. "
            "Please consult your local extension officer or product label for safe dosage guidance."
        ), []

    # 5. Hallucination markers — soft warning
    if _HALLUCINATION_MARKERS.search(text):
        warnings.append(
            "The advisory contains very confident language. "
            "Always verify recommendations with a qualified agricultural extension officer."
        )

    # 6. Required sections present — soft warning
    missing = [s for s in _ADVISORY_REQUIRED_SECTIONS
               if not re.search(rf"(?i){re.escape(s)}", text)]
    if missing:
        warnings.append(
            f"The advisory is missing sections: {', '.join(missing)}. "
            "Some guidance may not appear below."
        )

    return True, "", warnings
