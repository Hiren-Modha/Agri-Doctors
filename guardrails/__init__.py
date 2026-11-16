"""
guardrails package — nine categories of AI safety guardrails for Agri Doctor.

Categories
----------
1. input_guardrails      — image file validation, PII, prompt injection, content filtering
2. output_guardrails     — hallucination flags, toxicity, format, sensitive info
3. tool_guardrails       — API auth, function-call validation, restricted execution
4. data_guardrails       — data masking, privacy, PII scrubbing from text
5. security_guardrails   — jailbreak, injection, cross-agent integrity
6. operational_guardrails — token limits, rate limiting, retries, timeouts
7. hitl_guardrails       — human-in-the-loop escalation triggers
8. multiagent_guardrails — handoff validation, loop prevention, output consistency
9. ethical_guardrails    — bias flags, explainability notes, audit trail
"""

from .input_guardrails import validate_image_input, validate_filename
from .output_guardrails import validate_diagnosis_output, validate_advisory_output
from .tool_guardrails import validate_api_call, validate_model_access, validate_function_call
from .data_guardrails import mask_pii, check_data_privacy, sanitize_for_handoff
from .security_guardrails import (detect_jailbreak, detect_prompt_injection,
                                   check_cross_agent_integrity, compute_message_digest,
                                   detect_credential_exposure)
from .operational_guardrails import OperationalGuardrail, TokenBudget, with_retry
from .hitl_guardrails import hitl_escalation_check
from .multiagent_guardrails import validate_agent_handoff, detect_agent_loop, check_output_consistency
from .ethical_guardrails import AuditTrail, check_bias_flags, check_prohibited_content, add_explainability_note

__all__ = [
    # 1. Input
    "validate_image_input", "validate_filename",
    # 2. Output
    "validate_diagnosis_output", "validate_advisory_output",
    # 3. Tool
    "validate_api_call", "validate_model_access", "validate_function_call",
    # 4. Data
    "mask_pii", "check_data_privacy", "sanitize_for_handoff",
    # 5. Security
    "detect_jailbreak", "detect_prompt_injection", "check_cross_agent_integrity",
    "compute_message_digest", "detect_credential_exposure",
    # 6. Operational
    "OperationalGuardrail", "TokenBudget", "with_retry",
    # 7. HITL
    "hitl_escalation_check",
    # 8. Multi-agent
    "validate_agent_handoff", "detect_agent_loop", "check_output_consistency",
    # 9. Ethical
    "AuditTrail", "check_bias_flags", "check_prohibited_content", "add_explainability_note",
]
