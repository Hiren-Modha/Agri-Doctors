"""
IBM watsonx.ai credential and model factory.

Credentials are read exclusively from environment variables (loaded from .env).
Nothing is hardcoded here.
"""

import os
from dotenv import load_dotenv
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

# Load .env file if present (no-op when running in production with real env vars)
load_dotenv()

# ── Defaults ──────────────────────────────────────────────────────────────────
_DEFAULT_VISION_MODEL = "mistralai/mistral-small-3-1-24b-instruct-2503"
_DEFAULT_TEXT_MODEL = "ibm/granite-4-h-small"


def get_credentials() -> dict:
    """
    Read and validate IBM watsonx.ai credentials from environment variables.

    Returns a dict with keys: api_key, project_id, url.
    Raises ValueError with a user-friendly message if any required value is missing.
    """
    api_key = os.environ.get("WATSONX_API_KEY", "").strip()
    project_id = os.environ.get("WATSONX_PROJECT_ID", "").strip()
    url = os.environ.get("WATSONX_URL", "https://us-south.ml.cloud.ibm.com").strip()

    missing = []
    if not api_key or api_key.startswith("PASTE_"):
        missing.append("WATSONX_API_KEY")
    if not project_id or project_id.startswith("PASTE_"):
        missing.append("WATSONX_PROJECT_ID")
    if not url:
        missing.append("WATSONX_URL")

    if missing:
        raise ValueError(
            f"Missing or placeholder IBM watsonx.ai credentials: {', '.join(missing)}. "
            f"On Streamlit Cloud, add them under Settings → Secrets. "
            f"Locally, copy .env.example to .env and fill in your real values."
        )

    return {"api_key": api_key, "project_id": project_id, "url": url}


def get_vision_model() -> ModelInference:
    """
    Return a configured watsonx.ai ModelInference instance for the vision LLM.

    Model ID is read from WATSONX_VISION_MODEL env var (falls back to default).
    """
    creds = get_credentials()
    model_id = os.environ.get("WATSONX_VISION_MODEL", _DEFAULT_VISION_MODEL).strip()

    credentials = Credentials(
        api_key=creds["api_key"],
        url=creds["url"],
    )

    params = {
        GenParams.MAX_NEW_TOKENS: 1024,
        GenParams.TEMPERATURE: 0.1,   # low temperature for factual diagnosis
    }

    return ModelInference(
        model_id=model_id,
        credentials=credentials,
        project_id=creds["project_id"],
        params=params,
    )


def get_text_model() -> ModelInference:
    """
    Return a configured watsonx.ai ModelInference instance for the text LLM.

    Model ID is read from WATSONX_TEXT_MODEL env var (falls back to default).
    """
    creds = get_credentials()
    model_id = os.environ.get("WATSONX_TEXT_MODEL", _DEFAULT_TEXT_MODEL).strip()

    credentials = Credentials(
        api_key=creds["api_key"],
        url=creds["url"],
    )

    params = {
        GenParams.MAX_NEW_TOKENS: 1500,
        GenParams.TEMPERATURE: 0.3,   # slightly more creative for advisory language
    }

    return ModelInference(
        model_id=model_id,
        credentials=credentials,
        project_id=creds["project_id"],
        params=params,
    )


def get_active_model_names() -> dict:
    """
    Return the model IDs currently configured via env vars.
    Safe to display in the UI — does not expose credentials.
    """
    return {
        "vision_model": os.environ.get("WATSONX_VISION_MODEL", _DEFAULT_VISION_MODEL),
        "text_model": os.environ.get("WATSONX_TEXT_MODEL", _DEFAULT_TEXT_MODEL),
    }
