"""
Pathologist Agent — Crop Disease Diagnosis.

Accepts raw image bytes, encodes them as base64, and calls the IBM watsonx.ai
vision LLM to produce a structured crop disease diagnosis.

The agent deliberately refuses to guess when an image is unclear or not a plant.
"""

import base64
import re
from ibm_watsonx_ai.foundation_models import ModelInference

# ── System prompt ─────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """You are an expert plant pathologist AI assistant.
Your role is to examine crop and plant-leaf images to diagnose diseases.

IMPORTANT RULES:
1. Only analyse plant, crop, or leaf images. If the image does not show a plant, politely
   state that and do not attempt a diagnosis.
2. If the image is blurry, too dark, too small, or otherwise unclear, respond with:
   UNCLEAR IMAGE: <reason why>
   Do NOT guess when the image is insufficient.
3. For clear plant images, provide your analysis in EXACTLY this format:

SUSPECTED DISEASE: <disease name or "Unable to determine">
OBSERVED SYMPTOMS:
- <symptom 1>
- <symptom 2>
- <symptom 3 if applicable>
CONFIDENCE LEVEL: <High / Medium / Low>
POSSIBLE CAUSES:
- <cause 1>
- <cause 2 if applicable>
ALTERNATIVE DIAGNOSES:
- <alternative 1 or "None">
- <alternative 2 if applicable>
ADDITIONAL NOTES: <any other relevant observation>

4. Base your diagnosis only on what is visible in the image. Do not fabricate symptoms.
5. If the disease cannot be reliably identified, state "Unable to determine" and advise
   consulting a local plant pathology expert."""


def _encode_image(image_bytes: bytes) -> str:
    """Return base64-encoded string of raw image bytes."""
    return base64.b64encode(image_bytes).decode("utf-8")


def _build_messages(image_b64: str, mime_type: str) -> list:
    """
    Build the multimodal message payload for the watsonx.ai vision model.
    Uses the chat-style messages format with an inline image.
    """
    return [
        {
            "role": "system",
            "content": _SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{image_b64}",
                    },
                },
                {
                    "type": "text",
                    "text": (
                        "Please examine this crop/plant image carefully and provide "
                        "your diagnosis following the format in your instructions."
                    ),
                },
            ],
        },
    ]


def _parse_diagnosis(raw_text: str) -> dict:
    """
    Extract structured fields from the LLM's plain-text response.
    Returns a dict with keys aligned to what app.py and report_formatter expect.
    """
    text = raw_text.strip()

    # Detect unclear-image response
    if re.search(r"(?i)^UNCLEAR IMAGE\s*:", text, re.MULTILINE):
        reason_match = re.search(r"(?i)UNCLEAR IMAGE\s*:\s*(.+)", text)
        reason = reason_match.group(1).strip() if reason_match else "Image is unclear."
        return {
            "unclear_image": True,
            "unclear_reason": reason,
            "raw_report": raw_text,
        }

    # Detect non-plant image response
    if not re.search(r"(?i)SUSPECTED DISEASE", text):
        return {
            "unclear_image": True,
            "unclear_reason": (
                "The uploaded image does not appear to show a plant or crop. "
                "Please upload a clear photo of the affected crop or leaf."
            ),
            "raw_report": raw_text,
        }

    def _extract(pattern: str, default: str = "Not identified") -> str:
        m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        return m.group(1).strip() if m else default

    def _extract_bullets(start_label: str, end_labels: list) -> list:
        """Extract a bullet-point section between start_label and the next section."""
        # Build a pattern that captures everything until the next section header
        end_pattern = "|".join(re.escape(e) for e in end_labels)
        pattern = rf"(?i){re.escape(start_label)}\s*:\s*\n(.*?)(?={end_pattern}|$)"
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if not m:
            return []
        block = m.group(1)
        bullets = re.findall(r"[-•*]\s*(.+)", block)
        return [b.strip() for b in bullets if b.strip()]

    disease = _extract(r"SUSPECTED DISEASE\s*:\s*(.+?)(?:\n|$)")
    confidence = _extract(r"CONFIDENCE LEVEL\s*:\s*(.+?)(?:\n|$)")
    notes = _extract(r"ADDITIONAL NOTES\s*:\s*(.+?)(?:\n\n|$)", default="")

    symptoms = _extract_bullets(
        "OBSERVED SYMPTOMS",
        ["CONFIDENCE LEVEL", "POSSIBLE CAUSES", "ALTERNATIVE DIAGNOSES", "ADDITIONAL NOTES"],
    )
    causes = _extract_bullets(
        "POSSIBLE CAUSES",
        ["ALTERNATIVE DIAGNOSES", "ADDITIONAL NOTES"],
    )
    alternatives = _extract_bullets(
        "ALTERNATIVE DIAGNOSES",
        ["ADDITIONAL NOTES"],
    )
    # Filter out "None" placeholder entries
    alternatives = [a for a in alternatives if a.lower() not in ("none", "n/a")]

    return {
        "unclear_image": False,
        "disease": disease,
        "symptoms": symptoms,
        "confidence": confidence,
        "causes": causes,
        "alternative_diagnoses": alternatives,
        "additional_notes": notes,
        "raw_report": raw_text,
    }


def run_pathologist_agent(image_bytes: bytes, model: ModelInference, mime_type: str = "image/jpeg") -> dict:
    """
    Run the Pathologist Agent.

    Args:
        image_bytes: Raw bytes of the uploaded image file.
        model: A configured watsonx.ai ModelInference instance (vision model).
        mime_type: MIME type string — "image/jpeg" or "image/png".

    Returns:
        A dict with diagnosis fields, or an 'error' key on failure.
    """
    try:
        image_b64 = _encode_image(image_bytes)
        messages = _build_messages(image_b64, mime_type)

        # Call the vision LLM via the chat interface
        response = model.chat(messages=messages)

        # Extract the text response
        raw_text = ""
        if isinstance(response, dict):
            choices = response.get("choices", [])
            if choices:
                raw_text = choices[0].get("message", {}).get("content", "")
        if not raw_text:
            raw_text = str(response)

        return _parse_diagnosis(raw_text)

    except ValueError as exc:
        # Credential / configuration errors — surface directly
        return {"error": str(exc), "raw_report": ""}
    except Exception as exc:
        return {
            "error": (
                f"The Pathologist Agent encountered an error while analysing the image. "
                f"Details: {exc}"
            ),
            "raw_report": "",
        }
