"""
Guardrail Category 5 — Security Guardrails.

Protects the Agri Doctor application from security threats:
  - Prompt injection detection (text in filename, metadata, or any user-visible field)
  - Jailbreak attempt detection
  - SQL injection prevention
  - Cross-agent communication integrity check (digest comparison)
  - Credential and secret exposure prevention
"""

import re
import hashlib

# ── Prompt injection patterns ─────────────────────────────────────────────────
_INJECTION_PATTERNS = re.compile(
    r"(?i)("
    r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|context)|"
    r"system\s*:\s*you\s+are|"
    r"<\s*system\s*>|"
    r"\[system\]|"
    r"act\s+as\s+(a\s+)?(different|new|evil|unrestricted)|"
    r"you\s+are\s+now\s+(dan|jailbreak|free)|"
    r"forget\s+(your|all)\s+(instructions?|training|rules?)|"
    r"override\s+(safety|filter|guardrail)|"
    r"disable\s+(content\s+filter|safety)|"
    r"bypass\s+(moderation|filter|safety)|"
    r"pretend\s+(you\s+are|to\s+be)\s+(un|not)"
    r")"
)

# ── Jailbreak patterns ────────────────────────────────────────────────────────
_JAILBREAK_PATTERNS = re.compile(
    r"(?i)("
    r"\bdan\s+mode\b|"
    r"\bdo\s+anything\s+now\b|"
    r"\bjailbreak\b|"
    r"developer\s+mode|"
    r"unrestricted\s+mode|"
    r"no\s+restrictions?|"
    r"without\s+(any\s+)?(rules?|restrictions?|filters?)|"
    r"hypothetically\s+if\s+you\s+had\s+no\s+(rules?|filter)|"
    r"in\s+a\s+fictional\s+(world|scenario)\s+where\s+you\s+can"
    r")"
)

# ── SQL injection patterns ────────────────────────────────────────────────────
_SQL_INJECTION = re.compile(
    r"(?i)("
    r"(';|\";\s*)(\s*DROP|\s*DELETE|\s*INSERT|\s*UPDATE|\s*SELECT)|"
    r"\bOR\s+1\s*=\s*1\b|"
    r"\bAND\s+1\s*=\s*1\b|"
    r"--\s*$|"
    r"/\*.*?\*/|"
    r"UNION\s+(ALL\s+)?SELECT"
    r")"
)

# ── Credential exposure ───────────────────────────────────────────────────────
_CREDENTIAL_EXPOSURE = re.compile(
    r"(?i)(WATSONX_API_KEY\s*=\s*\S+|WATSONX_PROJECT_ID\s*=\s*\S+|"
    r"api[_\-\s]?key\s*[:=]\s*['\"]?\S{10,}|"
    r"password\s*[:=]\s*['\"]?\S{6,}|"
    r"bearer\s+[A-Za-z0-9\-_.~+/]{20,})"
)


def detect_prompt_injection(text: str) -> tuple[bool, str]:
    """
    Scan any text (filename, metadata, indirect input) for prompt injection attempts.

    Returns:
        (True, "")          — no injection detected
        (False, reason_str) — injection attempt detected, block processing
    """
    if _INJECTION_PATTERNS.search(text):
        return False, (
            "Security alert: the input contains a prompt injection pattern. "
            "This request has been blocked to protect the AI agents."
        )
    return True, ""


def detect_jailbreak(text: str) -> tuple[bool, str]:
    """
    Detect jailbreak attempts in any user-supplied text.

    Returns:
        (True, "")          — no jailbreak detected
        (False, reason_str) — jailbreak attempt detected
    """
    if _JAILBREAK_PATTERNS.search(text):
        return False, (
            "Security alert: the input contains a jailbreak pattern. "
            "This request has been blocked."
        )
    return True, ""


def detect_sql_injection(text: str) -> tuple[bool, str]:
    """
    Detect SQL injection patterns in user-supplied or agent-generated text.

    Returns:
        (True, "")          — no SQL injection detected
        (False, reason_str) — SQL injection pattern found
    """
    if _SQL_INJECTION.search(text):
        return False, (
            "Security alert: SQL injection pattern detected in input. "
            "This request has been blocked."
        )
    return True, ""


def detect_credential_exposure(text: str) -> tuple[bool, str]:
    """
    Scan any agent output or inter-agent message for credential leakage.

    Returns:
        (True, "")          — no credentials detected
        (False, reason_str) — credentials found in text
    """
    if _CREDENTIAL_EXPOSURE.search(text):
        return False, (
            "Security alert: possible API credentials detected in agent output. "
            "This response has been blocked to protect your secrets."
        )
    return True, ""


def check_cross_agent_integrity(message: str, expected_digest: str) -> tuple[bool, str]:
    """
    Verify the integrity of a message passed between agents using an SHA-256 digest.
    Call compute_message_digest() on the original message, store the result,
    then call this function before the second agent processes it.

    Returns:
        (True, "")          — message has not been tampered with
        (False, reason_str) — digest mismatch, message may have been altered
    """
    actual_digest = compute_message_digest(message)
    if actual_digest != expected_digest:
        return False, (
            "Cross-agent security check failed: the diagnostic report passed between "
            "agents has been modified unexpectedly. "
            "Please run the diagnosis again."
        )
    return True, ""


def compute_message_digest(message: str) -> str:
    """Return the SHA-256 hex digest of a message string (UTF-8 encoded)."""
    return hashlib.sha256(message.encode("utf-8")).hexdigest()
