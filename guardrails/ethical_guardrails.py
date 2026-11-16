"""
Guardrail Category 9 — Ethical and Compliance Guardrails.

Ensures responsible and compliant AI behaviour:
  - Bias detection (over-confident claims, demographic assumptions)
  - Fairness validation (advice should apply regardless of farm size or location)
  - Explainability — tags AI-generated content clearly
  - Regulatory compliance markers
  - Responsible AI principles check
  - Audit trail (immutable log of all guardrail decisions per session)
"""

import re
import hashlib
from datetime import datetime
from dataclasses import dataclass, field
from typing import List

# ── Bias / fairness patterns ──────────────────────────────────────────────────
_BIAS_PATTERNS = re.compile(
    r"(?i)("
    r"only\s+(small|large|rich|poor|developed|developing)\s+(farm|country|region)|"
    r"farmers\s+in\s+\w+\s+should|"
    r"this\s+(only|never)\s+happens\s+in|"
    r"(\ball\s+farmers\b|\bno\s+farmer\b)\s+(should|must|can)"
    r")"
)

# ── Responsible AI — absolute prohibition patterns ────────────────────────────
# (things the AI must never recommend under any circumstances)
_PROHIBITED_CONTENT = re.compile(
    r"(?i)("
    r"burn\s+(down|your)(\s+\w+)?\s+(crop|field|farm)|"
    r"illegal\s+(pesticide|chemical)|"
    r"banned\s+(pesticide|chemical|product)|"
    r"mix\s+with\s+alcohol|"
    r"ignore\s+(local\s+)?(regulation|law|guideline)"
    r")"
)


@dataclass
class AuditEvent:
    """A single guardrail audit event."""
    timestamp: str
    guardrail_category: str
    check_name: str
    result: str         # "PASS" | "BLOCK" | "WARN"
    detail: str
    input_digest: str   # SHA-256 of the input that was checked (privacy-safe)


@dataclass
class AuditTrail:
    """
    Immutable per-session audit trail for all guardrail decisions.

    Usage:
        trail = AuditTrail()
        trail.log("Input", "validate_image_input", "PASS", "Image OK", image_bytes)
        ...
        report = trail.to_text()
    """
    events: List[AuditEvent] = field(default_factory=list)

    def log(self, category: str, check_name: str, result: str,
            detail: str, input_data: object = None) -> None:
        """
        Append a guardrail event to the trail.

        Args:
            category:   Guardrail category name (e.g., "Input", "Output").
            check_name: Name of the specific check (e.g., "validate_image_input").
            result:     "PASS", "BLOCK", or "WARN".
            detail:     Human-readable description of the outcome.
            input_data: The data that was checked — only its digest is stored.
        """
        digest = ""
        if input_data is not None:
            raw = input_data if isinstance(input_data, bytes) else str(input_data).encode()
            digest = hashlib.sha256(raw).hexdigest()[:12]  # short digest, privacy-safe

        self.events.append(AuditEvent(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            guardrail_category=category,
            check_name=check_name,
            result=result,
            detail=detail,
            input_digest=digest,
        ))

    def get_summary(self) -> dict:
        """Return counts of PASS / WARN / BLOCK events."""
        return {
            "total": len(self.events),
            "pass": sum(1 for e in self.events if e.result == "PASS"),
            "warn": sum(1 for e in self.events if e.result == "WARN"),
            "block": sum(1 for e in self.events if e.result == "BLOCK"),
        }

    def to_text(self) -> str:
        """Render the audit trail as a plain-text string for the downloadable report."""
        if not self.events:
            return "  (No guardrail events recorded)\n"
        lines = []
        for ev in self.events:
            lines.append(
                f"  [{ev.timestamp}] [{ev.result:5s}] "
                f"[{ev.guardrail_category}] {ev.check_name}: {ev.detail} "
                f"(digest:{ev.input_digest})"
            )
        return "\n".join(lines)


def check_bias_flags(text: str) -> tuple[bool, list]:
    """
    Scan agent output for language that may indicate bias or unfair generalisation.

    Returns:
        (bias_detected: bool, warning_messages: list[str])
    """
    warnings = []
    if _BIAS_PATTERNS.search(text):
        warnings.append(
            "Fairness check: the AI response contains language that may generalise "
            "unfairly across farm types or geographies. "
            "Advice should apply to all farmers — verify with a local expert."
        )
    return len(warnings) > 0, warnings


def check_prohibited_content(text: str) -> tuple[bool, str]:
    """
    Hard-block any output that contains absolutely prohibited recommendations.

    Returns:
        (True, "")          — content is acceptable
        (False, reason_str) — prohibited content detected
    """
    match = _PROHIBITED_CONTENT.search(text)
    if match:
        return False, (
            f"Ethical guardrail: the AI output contains a prohibited recommendation "
            f"('{match.group()}') and has been blocked. "
            "This application does not support advice that may be harmful or illegal."
        )
    return True, ""


def add_explainability_note(diagnosis: dict, advisory: dict) -> str:
    """
    Generate a short explainability summary describing how the AI reached its conclusions.
    Returned string is inserted into the UI and report as a transparency note.
    """
    disease = diagnosis.get("disease", "unknown disease")
    confidence = diagnosis.get("confidence", "unspecified confidence")
    symptom_count = len(diagnosis.get("symptoms", []))
    action_count = len(advisory.get("immediate_actions", []))

    return (
        f"Explainability Note: The Pathologist Agent (vision LLM) examined the uploaded "
        f"image and identified visible symptoms ({symptom_count} symptom(s) observed). "
        f"Based on these visual patterns, it assessed the suspected condition as "
        f"'{disease}' with {confidence} confidence. "
        f"The Extension Officer Agent (text LLM) then received this diagnosis report and "
        f"generated {action_count} immediate action(s) using agricultural best-practice "
        f"knowledge. No external databases were queried. All conclusions are based solely "
        f"on the uploaded image and the AI models' training data."
    )
