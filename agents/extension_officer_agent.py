"""
Extension Officer Agent — Remedial Action Report.

Receives the Pathologist Agent's diagnostic report as text and uses the IBM
watsonx.ai Granite text LLM to generate a farmer-friendly advisory in simple,
plain language.

Safety rules enforced via the system prompt:
- Never invent pesticide dosages.
- Always include safety/PPE warnings.
- Always recommend consulting a local expert if disease is uncertain.
"""

import re
from ibm_watsonx_ai.foundation_models import ModelInference

# ── System prompt ─────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """You are a friendly and knowledgeable agricultural extension officer AI.
Your role is to read a plant disease diagnosis report and prepare practical,
easy-to-understand advice for a farmer.

LANGUAGE RULES:
- Use simple, everyday language. Avoid technical jargon.
- Write as if speaking directly to a farmer who may have limited agricultural training.
- Be encouraging and supportive in tone.

CONTENT RULES:
1. Base your advice ONLY on the diagnosis provided. Do not add diseases not mentioned.
2. If the diagnosis says "Unable to determine" or the disease is uncertain, your FIRST
   recommendation must be: "Please contact your local agricultural extension officer or
   a certified plant pathologist for a confirmed diagnosis before taking any action."
3. You MAY recommend chemical treatments if appropriate, but:
   - NEVER invent or state specific dosage amounts (e.g., "10 ml per litre").
   - ALWAYS direct the farmer to read the product label for correct dosage.
   - ALWAYS advise following local agricultural regulations.
4. ALWAYS include a safety section reminding the farmer to:
   - Wear personal protective equipment (gloves, mask, goggles) when handling chemicals.
   - Keep chemicals away from children and food crops.
   - Consult local agricultural guidelines or an extension officer before purchase.

FORMAT — provide your advisory in EXACTLY this structure:

IMMEDIATE ACTIONS:
- <action 1>
- <action 2>
- <action 3 if needed>

PREVENTIVE MEASURES:
- <measure 1>
- <measure 2>

CROP CARE RECOMMENDATIONS:
- <recommendation 1>
- <recommendation 2>

TREATMENT OPTIONS:
- <option 1>
- <option 2>
(If chemical treatment: recommend product type/class, NOT specific dosage. Direct farmer to product label.)

SAFETY WARNINGS:
- Always wear gloves, a face mask, and goggles when handling pesticides or chemicals.
- Keep all agricultural chemicals locked away and out of reach of children.
- Read and follow the product label instructions carefully before use.
- Dispose of empty containers safely according to local regulations.
- <any additional safety note specific to this disease/treatment>

EXPERT RECOMMENDATION:
<A short paragraph advising when to contact a local extension officer or expert.>"""


def _build_prompt(diagnosis_report: str) -> str:
    """
    Format the diagnosis report into a prompt for the Extension Officer Agent.

    The diagnosis report is the raw_report text from the Pathologist Agent.
    """
    return (
        "Here is the plant disease diagnosis report from the Pathologist Agent:\n\n"
        "---\n"
        f"{diagnosis_report}\n"
        "---\n\n"
        "Based on this diagnosis, please prepare a practical, farmer-friendly advisory "
        "following the format in your instructions."
    )


def _parse_advisory(raw_text: str) -> dict:
    """
    Extract structured sections from the LLM's advisory text.
    Returns a dict with keys aligned to what app.py and report_formatter expect.
    """
    text = raw_text.strip()

    def _extract_bullets(start_label: str, end_labels: list) -> list:
        end_pattern = "|".join(re.escape(e) for e in end_labels)
        pattern = rf"(?i){re.escape(start_label)}\s*:\s*\n(.*?)(?={end_pattern}|$)"
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if not m:
            return []
        block = m.group(1)
        bullets = re.findall(r"[-•*]\s*(.+)", block)
        return [b.strip() for b in bullets if b.strip()]

    def _extract_paragraph(start_label: str, end_labels: list) -> str:
        end_pattern = "|".join(re.escape(e) for e in end_labels)
        pattern = rf"(?i){re.escape(start_label)}\s*:\s*\n(.*?)(?={end_pattern}|$)"
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else ""

    immediate_actions = _extract_bullets(
        "IMMEDIATE ACTIONS",
        ["PREVENTIVE MEASURES", "CROP CARE RECOMMENDATIONS", "TREATMENT OPTIONS",
         "SAFETY WARNINGS", "EXPERT RECOMMENDATION"],
    )
    preventive_measures = _extract_bullets(
        "PREVENTIVE MEASURES",
        ["CROP CARE RECOMMENDATIONS", "TREATMENT OPTIONS", "SAFETY WARNINGS",
         "EXPERT RECOMMENDATION"],
    )
    crop_care = _extract_bullets(
        "CROP CARE RECOMMENDATIONS",
        ["TREATMENT OPTIONS", "SAFETY WARNINGS", "EXPERT RECOMMENDATION"],
    )
    treatment_options = _extract_bullets(
        "TREATMENT OPTIONS",
        ["SAFETY WARNINGS", "EXPERT RECOMMENDATION"],
    )
    safety_warnings = _extract_bullets(
        "SAFETY WARNINGS",
        ["EXPERT RECOMMENDATION"],
    )
    expert_recommendation = _extract_paragraph(
        "EXPERT RECOMMENDATION",
        [],
    )

    return {
        "immediate_actions": immediate_actions,
        "preventive_measures": preventive_measures,
        "crop_care": crop_care,
        "treatment_options": treatment_options,
        "safety_warnings": safety_warnings,
        "expert_recommendation": expert_recommendation,
        "raw_advisory": raw_text,
    }


def run_extension_officer_agent(diagnosis_report: str, model: ModelInference) -> dict:
    """
    Run the Extension Officer Agent.

    Args:
        diagnosis_report: The raw_report text string from the Pathologist Agent's output.
        model: A configured watsonx.ai ModelInference instance (text model).

    Returns:
        A dict with advisory sections, or an 'error' key on failure.
    """
    try:
        prompt = _build_prompt(diagnosis_report)

        # Use the generate interface (text-only LLM)
        response = model.generate(prompt=prompt)

        # Extract the generated text
        raw_text = ""
        if isinstance(response, dict):
            results = response.get("results", [])
            if results:
                raw_text = results[0].get("generated_text", "")
        if not raw_text:
            raw_text = str(response)

        return _parse_advisory(raw_text)

    except ValueError as exc:
        return {"error": str(exc), "raw_advisory": ""}
    except Exception as exc:
        return {
            "error": (
                f"The Extension Officer Agent encountered an error while preparing "
                f"the advisory. Details: {exc}"
            ),
            "raw_advisory": "",
        }
