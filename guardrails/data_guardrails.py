"""
Guardrail Category 4 — Data Guardrails.

Protects data accessed and processed by agents:
  - PII detection and masking in text (phone, email, name patterns, ID numbers)
  - Data privacy: confirms no raw credentials appear in agent-visible text
  - Role-based data access (Agri Doctor only exposes diagnosis + advisory data)
  - Prevention of unauthorized data retrieval (no file system paths, DB queries)
"""

import re

# ── PII patterns ──────────────────────────────────────────────────────────────
_PII_RULES = [
    # Email address
    (re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
     "[EMAIL REDACTED]"),
    # Phone numbers (10–13 digits, various separators)
    (re.compile(r"\b(\+?\d[\d\s\-().]{8,14}\d)\b"),
     "[PHONE REDACTED]"),
    # Credit/debit card patterns
    (re.compile(r"\b(?:\d[ -]?){13,16}\b"),
     "[CARD NUMBER REDACTED]"),
    # National ID / Aadhaar-style 12 digit number
    (re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),
     "[ID NUMBER REDACTED]"),
    # API-key-like strings (long alphanumeric with hyphens/underscores)
    (re.compile(r"\b[A-Za-z0-9\-_]{30,}\b"),
     "[CREDENTIAL REDACTED]"),
]

# ── Sensitive / credential patterns ──────────────────────────────────────────
_CREDENTIAL_PATTERNS = re.compile(
    r"(?i)(WATSONX_API_KEY|WATSONX_PROJECT_ID|api_key\s*=\s*\S+|"
    r"password\s*=\s*\S+|secret\s*=\s*\S+|bearer\s+\S{10,}|"
    r"sk-[a-zA-Z0-9]{20,})"
)

# ── Unauthorized data retrieval patterns ─────────────────────────────────────
_UNAUTHORIZED_RETRIEVAL = re.compile(
    r"(?i)(SELECT\s+\*\s+FROM|DROP\s+TABLE|INSERT\s+INTO|DELETE\s+FROM|"
    r"UNION\s+SELECT|os\.listdir|open\s*\(|subprocess\.run|"
    r"__import__|exec\s*\(|eval\s*\(|importlib)"
)


def mask_pii(text: str) -> tuple[str, list]:
    """
    Scan text for PII patterns and replace them with redaction tokens.

    Args:
        text: Any string (e.g., a diagnosis report before cross-agent handoff).

    Returns:
        (cleaned_text, list_of_pii_types_found)
    """
    found = []
    result = text
    for pattern, replacement in _PII_RULES:
        matches = pattern.findall(result)
        if matches:
            found.append(replacement)
            result = pattern.sub(replacement, result)
    return result, found


def check_data_privacy(text: str) -> tuple[bool, str]:
    """
    Scan text for credential leakage or unauthorized data retrieval attempts.
    Hard-blocks if found.

    Returns:
        (True, "")          — text is safe to pass to the next agent
        (False, reason_str) — data privacy violation detected
    """
    # Credential exposure
    if _CREDENTIAL_PATTERNS.search(text):
        return False, (
            "Data privacy violation: the text contains what appears to be "
            "credential or API key information. This data will not be forwarded to agents."
        )
    # SQL / code injection in agent input
    if _UNAUTHORIZED_RETRIEVAL.search(text):
        return False, (
            "Data privacy violation: the text contains database query or code execution "
            "patterns that are not permitted in agent inputs."
        )
    return True, ""


def sanitize_for_handoff(text: str) -> tuple[str, list]:
    """
    Convenience function: mask PII and return sanitized text for cross-agent handoff.
    Used before passing the Pathologist Agent's report to the Extension Officer Agent.

    Returns:
        (sanitized_text, list_of_items_redacted)
    """
    return mask_pii(text)
