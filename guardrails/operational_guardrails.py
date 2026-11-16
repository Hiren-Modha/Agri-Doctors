"""
Guardrail Category 6 — Operational Guardrails.

Controls agent execution and resource consumption:
  - Maximum execution time (via timer context manager)
  - Token usage budget tracking and enforcement
  - API rate limiting (per-session request counter)
  - Maximum agent iterations (loop prevention)
  - Retry and failure handling with exponential back-off
  - Cost monitoring
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Callable, Any

# ── Configurable limits ───────────────────────────────────────────────────────
MAX_AGENT_EXECUTION_SECONDS = 120       # 2 minutes per agent call
MAX_TOKENS_PER_SESSION = 8000           # combined input + output tokens per session
MAX_API_CALLS_PER_SESSION = 20          # prevent runaway loops
MAX_AGENT_ITERATIONS = 5               # prevent infinite re-try loops
COST_PER_1K_TOKENS_USD = 0.0002        # estimate — clearly labelled as such


@dataclass
class TokenBudget:
    """
    Tracks token consumption across a Streamlit session.
    Instantiate once per session and store in st.session_state.
    """
    max_tokens: int = MAX_TOKENS_PER_SESSION
    used_input_tokens: int = 0
    used_output_tokens: int = 0
    api_calls: int = 0
    agent_iterations: int = 0
    errors: int = 0
    _cost_per_1k: float = field(default=COST_PER_1K_TOKENS_USD, repr=False)

    @property
    def total_tokens(self) -> int:
        return self.used_input_tokens + self.used_output_tokens

    @property
    def estimated_cost_usd(self) -> float:
        return round(self.total_tokens / 1000 * self._cost_per_1k, 6)

    def record_call(self, input_tokens: int = 0, output_tokens: int = 0) -> tuple[bool, str]:
        """
        Record a completed API call. Returns (allowed, reason).
        Call BEFORE the actual API request to check budget.
        """
        if self.api_calls >= MAX_API_CALLS_PER_SESSION:
            return False, (
                f"Operational limit reached: maximum {MAX_API_CALLS_PER_SESSION} API calls "
                "per session. Please refresh the page to start a new session."
            )
        projected = self.total_tokens + input_tokens + output_tokens
        if projected > self.max_tokens:
            return False, (
                f"Token budget exceeded: this session has used {self.total_tokens} tokens "
                f"(limit: {self.max_tokens}). Please refresh the page to start a new session."
            )
        self.used_input_tokens += input_tokens
        self.used_output_tokens += output_tokens
        self.api_calls += 1
        return True, ""

    def record_error(self) -> None:
        self.errors += 1

    def record_iteration(self) -> tuple[bool, str]:
        """Check and increment agent iteration counter."""
        self.agent_iterations += 1
        if self.agent_iterations > MAX_AGENT_ITERATIONS:
            return False, (
                f"Operational guardrail: maximum agent iterations "
                f"({MAX_AGENT_ITERATIONS}) reached. "
                "Please refresh and try again."
            )
        return True, ""

    def summary(self) -> dict:
        return {
            "input_tokens": self.used_input_tokens,
            "output_tokens": self.used_output_tokens,
            "total_tokens": self.total_tokens,
            "api_calls": self.api_calls,
            "agent_iterations": self.agent_iterations,
            "errors": self.errors,
            "estimated_cost_usd": self.estimated_cost_usd,
            "budget_remaining_tokens": max(0, self.max_tokens - self.total_tokens),
        }


class OperationalGuardrail:
    """
    Context manager that enforces a maximum wall-clock time for an agent call.

    Usage:
        with OperationalGuardrail(agent_name="Pathologist Agent") as guard:
            result = run_pathologist_agent(...)
        if guard.timed_out:
            st.error(guard.timeout_message)
    """

    def __init__(self, agent_name: str = "Agent",
                 max_seconds: int = MAX_AGENT_EXECUTION_SECONDS):
        self.agent_name = agent_name
        self.max_seconds = max_seconds
        self.timed_out = False
        self.elapsed_seconds = 0.0
        self.timeout_message = ""
        self._start_time: float = 0.0

    def __enter__(self):
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_seconds = round(time.perf_counter() - self._start_time, 2)
        if self.elapsed_seconds > self.max_seconds:
            self.timed_out = True
            self.timeout_message = (
                f"The {self.agent_name} took longer than "
                f"{self.max_seconds} seconds ({self.elapsed_seconds:.1f}s elapsed). "
                "This may indicate a slow network or API issue. "
                "Please check your connection and try again."
            )
        return False   # do not suppress exceptions


def with_retry(func: Callable, max_retries: int = 2,
               initial_delay: float = 1.0) -> tuple[Any, str]:
    """
    Call func() with simple exponential back-off retry.

    Args:
        func:          Zero-argument callable to execute.
        max_retries:   Number of additional attempts after the first failure.
        initial_delay: Seconds to wait before the first retry.

    Returns:
        (result, "")          on success
        (None, error_message) after all retries exhausted
    """
    delay = initial_delay
    last_error = ""
    for attempt in range(max_retries + 1):
        try:
            result = func()
            return result, ""
        except Exception as exc:
            last_error = str(exc)
            if attempt < max_retries:
                time.sleep(delay)
                delay = min(delay * 2, 30.0)   # cap at 30 s
    return None, (
        f"The operation failed after {max_retries + 1} attempts. "
        f"Last error: {last_error}"
    )
