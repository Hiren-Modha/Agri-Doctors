"""
Guardrail Category 8 — Multi-Agent Guardrails.

Ensures safe coordination between the Pathologist Agent and Extension Officer Agent:
  - Handoff validation (Agent 1 output is structurally valid before Agent 2 receives it)
  - Agent loop detection (prevents infinite retry cycles)
  - Cross-agent output consistency check
  - Agent communication monitoring
  - Conflict detection between agent outputs
"""

import re

# ── Required fields for a valid handoff from Pathologist → Extension Officer ──
_REQUIRED_HANDOFF_FIELDS = ["raw_report", "disease", "symptoms"]
_MIN_REPORT_CHARS = 50

# ── Loop detection state ───────────────────────────────────────────────────────
_MAX_WORKFLOW_RUNS = 5    # per Streamlit session


def validate_agent_handoff(diagnosis: dict) -> tuple[bool, str]:
    """
    Validate that the Pathologist Agent's output is suitable for handoff to
    the Extension Officer Agent.

    Checks:
      - Required keys are present
      - raw_report has sufficient content
      - No error key is set
      - unclear_image is False (agent 1 must have succeeded)

    Returns:
        (True, "")          — handoff is valid
        (False, reason_str) — handoff blocked
    """
    # Must not have an error
    if diagnosis.get("error"):
        return False, (
            f"Multi-agent handoff blocked: Pathologist Agent reported an error — "
            f"{diagnosis['error']}"
        )

    # Must not be an unclear image result
    if diagnosis.get("unclear_image"):
        return False, (
            "Multi-agent handoff blocked: Pathologist Agent could not complete diagnosis "
            "(unclear image). Extension Officer Agent will not receive the report."
        )

    # Required fields
    missing = [f for f in _REQUIRED_HANDOFF_FIELDS if not diagnosis.get(f)]
    if missing:
        return False, (
            f"Multi-agent handoff blocked: Pathologist Agent output is missing required "
            f"fields: {', '.join(missing)}. The Extension Officer Agent cannot proceed."
        )

    # Report content length
    if len(diagnosis.get("raw_report", "")) < _MIN_REPORT_CHARS:
        return False, (
            "Multi-agent handoff blocked: Pathologist Agent report is too short to "
            "be meaningful. Please try the diagnosis again."
        )

    return True, ""


def detect_agent_loop(run_count: int) -> tuple[bool, str]:
    """
    Prevent infinite agent execution loops.

    Args:
        run_count: Number of times the full workflow has been triggered this session.

    Returns:
        (True, "")          — within safe limit
        (False, reason_str) — loop limit exceeded
    """
    if run_count > _MAX_WORKFLOW_RUNS:
        return False, (
            f"Agent loop guard: the full diagnosis workflow has been run "
            f"{run_count} times in this session (maximum: {_MAX_WORKFLOW_RUNS}). "
            "Please refresh the page to start a new session."
        )
    return True, ""


def check_output_consistency(diagnosis: dict, advisory: dict) -> tuple[bool, list]:
    """
    Check whether the advisory is consistent with the diagnosis.
    Flags (soft warning) if the advisory mentions diseases or conditions
    not referenced in the original diagnosis.

    Returns:
        (consistent: bool, inconsistency_warnings: list[str])
    """
    warnings = []

    disease = diagnosis.get("disease", "").lower()
    raw_advisory = advisory.get("raw_advisory", "").lower()
    raw_report = diagnosis.get("raw_report", "").lower()

    # If advisory is empty, nothing to check
    if not raw_advisory:
        return True, []

    # Check: advisory should not introduce new unrelated diseases
    # Heuristic: if advisory contains a disease name not in the diagnosis report,
    # flag it as a potential consistency issue.
    disease_mentions = re.findall(r"\b(?:blight|rust|mildew|wilt|rot|mosaic|"
                                  r"canker|scab|smut|anthracnose|leaf\s+spot)\b",
                                  raw_advisory)
    for mention in set(disease_mentions):
        if mention not in raw_report:
            warnings.append(
                f"Consistency check: the advisory mentions '{mention}' which was not "
                "in the original diagnosis report. Please review for accuracy."
            )

    consistent = len(warnings) == 0
    return consistent, warnings
