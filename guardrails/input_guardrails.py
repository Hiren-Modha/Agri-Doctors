"""
Guardrail Category 1 — Input Guardrails.

Validates user inputs BEFORE they are passed to any AI agent:
  - Image file size, MIME type, and byte integrity
  - Filename sanity check (path traversal, executable extensions)
  - PII detection in filenames
  - Prompt injection detection
  - Inappropriate / malicious content signals in metadata
"""

import re
import os

# ── Constants ─────────────────────────────────────────────────────────────────
_MAX_IMAGE_BYTES = 10 * 1024 * 1024          # 10 MB
_MIN_IMAGE_BYTES = 1024                       # 1 KB — reject near-empty files
_ALLOWED_MIME_TYPES = {"image/jpeg", "image/png"}
_ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

# JPEG magic bytes: FF D8 FF
# PNG magic bytes:  89 50 4E 47 0D 0A 1A 0A
_MAGIC_BYTES = {
    "image/jpeg": b"\xff\xd8\xff",
    "image/png": b"\x89PNG",
}

# PII patterns to detect in filenames (simplified heuristics)
_PII_PATTERNS = [
    re.compile(r"\b\d{10,12}\b"),                        # phone / ID number
    re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),  # email
    re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),             # card-like
]

# Prompt injection keywords that should never appear in a filename
# Filenames use underscores/hyphens in place of spaces, so we match both
_INJECTION_PATTERNS = [
    re.compile(r"(?i)(ignore[\s_\-]+(all[\s_\-]+)?(previous|prior|above)|"
               r"system[\s_\-]+prompt|act[\s_\-]+as|jailbreak|"
               r"dan[\s_\-]+mode|pretend[\s_\-]+you|forget[\s_\-]+your|"
               r"bypass[\s_\-]+filter|<script|javascript:|eval\()"),
]

# Dangerous file extensions (guard against disguised uploads)
_DANGEROUS_EXTENSIONS = {
    "exe", "sh", "bat", "cmd", "ps1", "php", "py", "js", "vbs", "jar",
    "dll", "msi", "apk", "elf",
}


def validate_image_input(image_bytes: bytes, mime_type: str) -> tuple[bool, str]:
    """
    Validate uploaded image bytes before passing to the Pathologist Agent.

    Returns:
        (True, "")          — input is safe
        (False, reason_str) — input is blocked, reason explains why
    """
    # 1. MIME type allowed?
    if mime_type not in _ALLOWED_MIME_TYPES:
        return False, (
            f"File type '{mime_type}' is not allowed. "
            f"Please upload a JPG or PNG image."
        )

    # 2. File size checks
    size = len(image_bytes)
    if size < _MIN_IMAGE_BYTES:
        return False, (
            f"The uploaded file is too small ({size} bytes) to be a valid image. "
            "Please upload a clear crop photograph."
        )
    if size > _MAX_IMAGE_BYTES:
        mb = size / (1024 * 1024)
        return False, (
            f"The uploaded file is too large ({mb:.1f} MB). "
            "Maximum allowed size is 10 MB. Please resize and try again."
        )

    # 3. Magic-byte verification (confirms it really is the declared type)
    expected_magic = _MAGIC_BYTES.get(mime_type)
    if expected_magic and not image_bytes.startswith(expected_magic):
        return False, (
            "The file content does not match its declared type. "
            "Please upload a genuine JPG or PNG image file."
        )

    return True, ""


def validate_filename(filename: str) -> tuple[bool, str]:
    """
    Validate the uploaded filename for path traversal, dangerous extensions,
    PII content, and prompt injection attempts.

    Returns:
        (True, "")          — filename is safe
        (False, reason_str) — filename is blocked
    """
    if not filename:
        return False, "No filename provided."

    # Normalise — strip directory components (path traversal guard)
    basename = os.path.basename(filename)
    if basename != filename.replace("\\", "/").split("/")[-1]:
        return False, "Suspicious file path detected. Please upload directly."

    # Check extension
    parts = basename.rsplit(".", 1)
    if len(parts) < 2:
        return False, "File has no extension. Please upload a JPG or PNG file."
    ext = parts[-1].lower()
    if ext in _DANGEROUS_EXTENSIONS:
        return False, (
            f"File extension '.{ext}' is not permitted. "
            "Please upload a JPG or PNG image."
        )
    if ext not in _ALLOWED_EXTENSIONS:
        return False, (
            f"Extension '.{ext}' is not supported. "
            "Supported formats: JPG, JPEG, PNG."
        )

    # PII in filename
    name_no_ext = parts[0]
    for pattern in _PII_PATTERNS:
        if pattern.search(name_no_ext):
            return False, (
                "The filename appears to contain personal information. "
                "Please rename the file and try again."
            )

    # Prompt injection in filename
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(name_no_ext):
            return False, (
                "The filename contains disallowed content. "
                "Please rename the file using a simple descriptive name."
            )

    # Filename length guard
    if len(basename) > 200:
        return False, "Filename is too long. Please use a shorter filename."

    return True, ""
