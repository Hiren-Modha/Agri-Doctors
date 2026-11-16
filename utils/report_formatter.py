"""
Plain-text advisory report formatter.

Combines the outputs of both agents into a single, clean plain-text report
suitable for download as a .txt file.

Optionally includes an audit trail from the guardrails package.
"""

from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from guardrails.ethical_guardrails import AuditTrail


_HEADER = "=" * 70
_SECTION = "-" * 70

_DISCLAIMER = (
    "IMPORTANT DISCLAIMER\n"
    + _SECTION + "\n"
    "This report is an AI-GENERATED PRELIMINARY ASSESSMENT only.\n"
    "It is NOT a confirmed laboratory diagnosis.\n"
    "Always verify findings with a qualified plant pathologist or your local\n"
    "agricultural extension officer before taking any action.\n"
    "Do NOT rely solely on this report for major treatment decisions."
)


def _bullet_list(items: list, indent: str = "  ") -> str:
    """Format a list of strings as an indented bullet list."""
    if not items:
        return f"{indent}(None provided)\n"
    return "".join(f"{indent}• {item}\n" for item in items)


def format_full_report(diagnosis: dict, advisory: dict, audit_trail=None) -> str:
    """
    Format the combined outputs of both agents into a downloadable plain-text report.

    Args:
        diagnosis:   Dict returned by run_pathologist_agent().
        advisory:    Dict returned by run_extension_officer_agent().
        audit_trail: Optional AuditTrail instance; if provided, appends a guardrails
                     audit section to the report.

    Returns:
        A multi-section plain-text string.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []

    # ── Header ────────────────────────────────────────────────────────────────
    lines.append(_HEADER)
    lines.append("         AGRI DOCTOR — Crop Disease Advisory Report")
    lines.append(f"         Generated: {timestamp}")
    lines.append(_HEADER)
    lines.append("")

    # ── Disclaimer ────────────────────────────────────────────────────────────
    lines.append(_DISCLAIMER)
    lines.append("")

    # ── Section 1: Pathologist Agent Diagnosis ────────────────────────────────
    lines.append(_HEADER)
    lines.append("SECTION 1: PATHOLOGIST AGENT — CROP DISEASE DIAGNOSIS")
    lines.append(_HEADER)
    lines.append("")

    if diagnosis.get("unclear_image"):
        lines.append("IMAGE STATUS: Unclear or not a plant image")
        lines.append(f"REASON: {diagnosis.get('unclear_reason', 'No reason provided.')}")
        lines.append("")
        lines.append(
            "Please upload a clear, well-lit photograph of the affected crop or leaf\n"
            "and run the diagnosis again."
        )
    else:
        lines.append(f"SUSPECTED DISEASE:  {diagnosis.get('disease', 'Not identified')}")
        lines.append(f"CONFIDENCE LEVEL:   {diagnosis.get('confidence', 'Not stated')}")
        lines.append("")

        lines.append("OBSERVED SYMPTOMS:")
        lines.append(_bullet_list(diagnosis.get("symptoms", [])))

        lines.append("POSSIBLE CAUSES:")
        lines.append(_bullet_list(diagnosis.get("causes", [])))

        alternatives = diagnosis.get("alternative_diagnoses", [])
        if alternatives:
            lines.append("ALTERNATIVE DIAGNOSES:")
            lines.append(_bullet_list(alternatives))

        notes = diagnosis.get("additional_notes", "").strip()
        if notes:
            lines.append("ADDITIONAL NOTES:")
            lines.append(f"  {notes}")
            lines.append("")

    lines.append("")

    # ── Section 2: Extension Officer Agent Advisory ───────────────────────────
    lines.append(_HEADER)
    lines.append("SECTION 2: EXTENSION OFFICER AGENT — FARMER ADVISORY")
    lines.append(_HEADER)
    lines.append("")

    if advisory.get("error"):
        lines.append(f"ERROR: {advisory['error']}")
    else:
        immediate = advisory.get("immediate_actions", [])
        if immediate:
            lines.append("IMMEDIATE ACTIONS:")
            lines.append(_bullet_list(immediate))

        preventive = advisory.get("preventive_measures", [])
        if preventive:
            lines.append("PREVENTIVE MEASURES:")
            lines.append(_bullet_list(preventive))

        crop_care = advisory.get("crop_care", [])
        if crop_care:
            lines.append("CROP CARE RECOMMENDATIONS:")
            lines.append(_bullet_list(crop_care))

        treatment = advisory.get("treatment_options", [])
        if treatment:
            lines.append("TREATMENT OPTIONS:")
            lines.append(_bullet_list(treatment))
            lines.append(
                "  NOTE: Always read product labels for correct dosage.\n"
                "  Follow local agricultural regulations before purchasing or applying\n"
                "  any chemical treatment."
            )
            lines.append("")

        safety = advisory.get("safety_warnings", [])
        if safety:
            lines.append("SAFETY WARNINGS:")
            lines.append(_bullet_list(safety))

        expert_rec = advisory.get("expert_recommendation", "").strip()
        if expert_rec:
            lines.append("EXPERT RECOMMENDATION:")
            lines.append(f"  {expert_rec}")
            lines.append("")

    # ── Section 3: Guardrails Audit Trail (if provided) ───────────────────────
    if audit_trail is not None:
        lines.append(_HEADER)
        lines.append("SECTION 3: GUARDRAILS AUDIT TRAIL")
        lines.append(_HEADER)
        lines.append("")
        summary = audit_trail.get_summary()
        lines.append(
            f"Total checks: {summary['total']}  |  "
            f"Passed: {summary['pass']}  |  "
            f"Warnings: {summary['warn']}  |  "
            f"Blocked: {summary['block']}"
        )
        lines.append("")
        lines.append(audit_trail.to_text())
        lines.append("")

    # ── Footer ────────────────────────────────────────────────────────────────
    lines.append(_SECTION)
    lines.append(
        "Contact your local agricultural extension officer or a certified plant\n"
        "pathologist for expert verification of this AI-generated report."
    )
    lines.append("")
    lines.append("Agri Doctor — AI Crop Disease Advisor | Powered by IBM watsonx.ai")
    lines.append(_SECTION)

    return "\n".join(lines)
