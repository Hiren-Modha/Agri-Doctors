"""
Guardrail Category 3 — Tool Execution Guardrails.

Controls how AI agents interact with external tools and APIs:
  - API authorization validation (confirms credentials present before each call)
  - Model access permission (allowlist of permitted model IDs)
  - Function-call validation (only permitted agent functions may be invoked)
  - Prevention of unauthorized actions
"""

import os
import re

# ── Permitted model IDs ───────────────────────────────────────────────────────
_ALLOWED_VISION_MODELS = {
    "mistralai/mistral-small-3-1-24b-instruct-2503",
    "ibm/granite-vision-3-3-2b",
    "ibm/granite-vision-4-1-4b",
}
_ALLOWED_TEXT_MODELS = {
    "ibm/granite-4-h-small",
    "ibm/granite-13b-instruct-v2",
    "ibm/granite-3-8b-instruct",
    "ibm/granite-3-2b-instruct",
}

# ── Permitted agent function names ────────────────────────────────────────────
_PERMITTED_FUNCTIONS = {
    "run_pathologist_agent",
    "run_extension_officer_agent",
    "get_vision_model",
    "get_text_model",
    "format_full_report",
}


def validate_api_call(api_key: str, project_id: str, url: str) -> tuple[bool, str]:
    """
    Confirm that credentials are present and structurally valid before any API call.
    Does NOT make a network request — purely structural validation.

    Returns:
        (True, "")          — authorised to proceed
        (False, reason_str) — block the API call
    """
    if not api_key or len(api_key) < 10:
        return False, (
            "API call blocked: IBM watsonx.ai API key is missing or invalid. "
            "Check your .env file."
        )
    if not project_id or len(project_id) < 8:
        return False, (
            "API call blocked: IBM watsonx.ai Project ID is missing or invalid. "
            "Check your .env file."
        )
    if not url or not url.startswith("https://"):
        return False, (
            "API call blocked: Endpoint URL is missing or does not use HTTPS. "
            "Check your .env file."
        )
    # Detect placeholder values that were not filled in
    for val in (api_key, project_id):
        if re.search(r"(?i)(paste|your|placeholder|example|xxx)", val):
            return False, (
                "API call blocked: Credentials appear to be placeholder values. "
                "Please fill in real credentials in your .env file."
            )
    return True, ""


def validate_model_access(model_id: str, model_role: str) -> tuple[bool, str]:
    """
    Verify the requested model ID is on the allowlist for its role.

    Args:
        model_id:   The model identifier string from env var or config.
        model_role: Either "vision" or "text".

    Returns:
        (True, "")          — model access permitted
        (False, reason_str) — model is not on allowlist
    """
    allowed = _ALLOWED_VISION_MODELS if model_role == "vision" else _ALLOWED_TEXT_MODELS
    if model_id not in allowed:
        return False, (
            f"Model '{model_id}' is not on the permitted model list for the '{model_role}' role. "
            f"Permitted models: {', '.join(sorted(allowed))}. "
            "Update WATSONX_VISION_MODEL or WATSONX_TEXT_MODEL in your .env file."
        )
    return True, ""


def validate_function_call(function_name: str) -> tuple[bool, str]:
    """
    Confirm that only permitted agent functions are being invoked.
    Guards against dynamic code execution or unexpected tool calls.

    Returns:
        (True, "")          — function call is permitted
        (False, reason_str) — function is not authorised
    """
    if function_name not in _PERMITTED_FUNCTIONS:
        return False, (
            f"Function '{function_name}' is not authorised for execution in this application. "
            "Only the designated agent functions may be called."
        )
    return True, ""
