"""
Performance Tracker — LLM & Agent Evaluation Metrics.

Captures and stores all evaluation metrics for each workflow run:
  1. Accuracy & Quality Metrics
  2. RAG Evaluation Metrics
  3. Agentic AI Evaluation Metrics
  4. Safety & Guardrail Evaluation Metrics
  5. Performance & Cost Metrics
  6. Multi-Agent System Evaluation

Usage:
    tracker = RunMetrics(run_id=1)
    tracker.start_agent("Pathologist Agent")
    # ... run agent ...
    tracker.end_agent("Pathologist Agent", input_tokens=820, output_tokens=310,
                      success=True, response_text="...")
    session = SessionMetrics()
    session.add_run(tracker)
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

# ── Cost rate (configurable) ──────────────────────────────────────────────────
COST_PER_1K_INPUT_TOKENS_USD  = 0.00015   # estimate for vision model
COST_PER_1K_OUTPUT_TOKENS_USD = 0.00060   # estimate for text generation

# ── Quality scoring helpers ───────────────────────────────────────────────────
def _rouge1_f1(hypothesis: str, reference: str) -> float:
    """
    Compute a lightweight ROUGE-1 F1 score (unigram overlap).
    Returns 0.0–1.0. Used when a reference answer is available.
    """
    if not hypothesis or not reference:
        return 0.0
    hyp_tokens = set(hypothesis.lower().split())
    ref_tokens = set(reference.lower().split())
    if not ref_tokens:
        return 0.0
    overlap = hyp_tokens & ref_tokens
    precision = len(overlap) / len(hyp_tokens) if hyp_tokens else 0.0
    recall    = len(overlap) / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


def _bleu1(hypothesis: str, reference: str) -> float:
    """
    Compute a lightweight BLEU-1 score (unigram precision).
    Returns 0.0–1.0.
    """
    if not hypothesis or not reference:
        return 0.0
    hyp_tokens  = hypothesis.lower().split()
    ref_tokens  = reference.lower().split()
    ref_set     = set(ref_tokens)
    if not hyp_tokens:
        return 0.0
    hits = sum(1 for t in hyp_tokens if t in ref_set)
    return round(hits / len(hyp_tokens), 4)


def _response_length_score(text: str) -> float:
    """
    Heuristic quality proxy: responses with 50–1500 words score 1.0,
    very short or very long responses score lower.
    Returns 0.0–1.0.
    """
    words = len(text.split())
    if words < 10:
        return 0.0
    if words < 50:
        return round(words / 50, 2)
    if words <= 1500:
        return 1.0
    return round(1500 / words, 2)


def _has_required_sections(text: str, sections: list) -> float:
    """
    Returns the fraction of required sections found in the text.
    1.0 = all sections present.
    """
    import re
    if not sections or not text:
        return 0.0
    found = sum(1 for s in sections if re.search(rf"(?i){re.escape(s)}", text))
    return round(found / len(sections), 2)


# ── Per-agent timing record ───────────────────────────────────────────────────
@dataclass
class AgentMetrics:
    """Metrics captured for a single agent call within one workflow run."""
    agent_name: str
    start_time: float = field(default_factory=time.perf_counter)
    end_time: float   = 0.0

    # 5. Performance & Cost
    input_tokens:   int   = 0
    output_tokens:  int   = 0
    success:        bool  = False
    error_message:  str   = ""

    # 1. Quality (computed after response)
    response_length_score:   float = 0.0   # heuristic completeness
    format_compliance_score: float = 0.0   # fraction of required sections present
    rouge1_f1:               float = 0.0   # 0 when no reference available
    bleu1:                   float = 0.0

    # 3. Agentic
    task_completed:  bool  = False
    goal_achieved:   bool  = False
    handoff_success: bool  = False

    # 4. Safety
    guardrail_checks:        int   = 0
    guardrail_blocks:        int   = 0
    guardrail_warnings:      int   = 0
    hallucination_flagged:   bool  = False
    pii_detected:            bool  = False
    toxicity_flagged:        bool  = False
    prompt_injection_blocked:bool  = False

    @property
    def latency_seconds(self) -> float:
        if self.end_time == 0.0:
            return 0.0
        return round(self.end_time - self.start_time, 3)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def tokens_per_second(self) -> float:
        lat = self.latency_seconds
        if lat <= 0:
            return 0.0
        return round(self.output_tokens / lat, 1)

    @property
    def cost_usd(self) -> float:
        return round(
            self.input_tokens  / 1000 * COST_PER_1K_INPUT_TOKENS_USD +
            self.output_tokens / 1000 * COST_PER_1K_OUTPUT_TOKENS_USD,
            6
        )

    @property
    def hallucination_rate(self) -> float:
        return 1.0 if self.hallucination_flagged else 0.0

    @property
    def toxicity_score(self) -> float:
        return 1.0 if self.toxicity_flagged else 0.0

    @property
    def guardrail_intervention_rate(self) -> float:
        total = self.guardrail_checks
        return round((self.guardrail_blocks + self.guardrail_warnings) / total, 3) if total else 0.0

    def to_dict(self) -> dict:
        return {
            "Agent":               self.agent_name,
            "Latency (s)":         self.latency_seconds,
            "Input Tokens":        self.input_tokens,
            "Output Tokens":       self.output_tokens,
            "Total Tokens":        self.total_tokens,
            "Tokens/s":            self.tokens_per_second,
            "Cost (USD)":          self.cost_usd,
            "Success":             self.success,
            "Task Completed":      self.task_completed,
            "Goal Achieved":       self.goal_achieved,
            "Handoff Success":     self.handoff_success,
            "Format Score":        self.format_compliance_score,
            "Length Score":        self.response_length_score,
            "ROUGE-1 F1":          self.rouge1_f1,
            "BLEU-1":              self.bleu1,
            "Guardrail Blocks":    self.guardrail_blocks,
            "Guardrail Warnings":  self.guardrail_warnings,
            "Hallucination Flag":  self.hallucination_flagged,
            "PII Detected":        self.pii_detected,
            "Toxicity Flagged":    self.toxicity_flagged,
        }


# ── Per-workflow-run record ───────────────────────────────────────────────────
@dataclass
class RunMetrics:
    """
    Aggregated metrics for one complete diagnosis + advisory workflow run.
    """
    run_id:     int
    timestamp:  str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    image_name: str = ""

    pathologist:      Optional[AgentMetrics] = None
    extension_officer: Optional[AgentMetrics] = None

    # 6. Multi-agent system metrics
    workflow_success:           bool  = False
    end_to_end_latency_seconds: float = 0.0
    handoff_accuracy:           float = 0.0   # 1.0 = handoff succeeded
    inter_agent_consistency:    float = 0.0   # 1.0 = consistent output
    loop_detected:              bool  = False

    # 2. RAG-style faithfulness metrics (self-assessed)
    faithfulness_score:  float = 0.0   # advisory grounded in diagnosis?
    answer_relevancy:    float = 0.0   # advisory relevant to diagnosed disease?
    groundedness_score:  float = 0.0   # response grounded in image evidence?

    @property
    def total_input_tokens(self) -> int:
        a1 = self.pathologist.input_tokens if self.pathologist else 0
        a2 = self.extension_officer.input_tokens if self.extension_officer else 0
        return a1 + a2

    @property
    def total_output_tokens(self) -> int:
        a1 = self.pathologist.output_tokens if self.pathologist else 0
        a2 = self.extension_officer.output_tokens if self.extension_officer else 0
        return a1 + a2

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def total_cost_usd(self) -> float:
        a1 = self.pathologist.cost_usd if self.pathologist else 0.0
        a2 = self.extension_officer.cost_usd if self.extension_officer else 0.0
        return round(a1 + a2, 6)

    @property
    def total_guardrail_blocks(self) -> int:
        a1 = self.pathologist.guardrail_blocks if self.pathologist else 0
        a2 = self.extension_officer.guardrail_blocks if self.extension_officer else 0
        return a1 + a2

    @property
    def total_guardrail_warnings(self) -> int:
        a1 = self.pathologist.guardrail_warnings if self.pathologist else 0
        a2 = self.extension_officer.guardrail_warnings if self.extension_officer else 0
        return a1 + a2

    def to_summary_dict(self) -> dict:
        return {
            "Run":           self.run_id,
            "Timestamp":     self.timestamp,
            "Image":         self.image_name,
            "E2E Latency(s)":self.end_to_end_latency_seconds,
            "Total Tokens":  self.total_tokens,
            "Total Cost($)": self.total_cost_usd,
            "Workflow OK":   self.workflow_success,
            "Handoff OK":    round(self.handoff_accuracy, 2),
            "Consistency":   round(self.inter_agent_consistency, 2),
            "Faithfulness":  round(self.faithfulness_score, 2),
            "Relevancy":     round(self.answer_relevancy, 2),
            "Groundedness":  round(self.groundedness_score, 2),
            "GR Blocks":     self.total_guardrail_blocks,
            "GR Warnings":   self.total_guardrail_warnings,
        }


# ── Session-level accumulator ─────────────────────────────────────────────────
class SessionMetrics:
    """
    Accumulates RunMetrics objects across the Streamlit session.
    Stored in st.session_state["session_metrics"].
    """

    def __init__(self):
        self.runs: list[RunMetrics] = []

    def add_run(self, run: RunMetrics) -> None:
        self.runs.append(run)

    @property
    def run_count(self) -> int:
        return len(self.runs)

    @property
    def success_count(self) -> int:
        return sum(1 for r in self.runs if r.workflow_success)

    @property
    def task_completion_rate(self) -> float:
        return round(self.success_count / self.run_count, 3) if self.run_count else 0.0

    @property
    def error_rate(self) -> float:
        return round(1 - self.task_completion_rate, 3)

    @property
    def total_tokens(self) -> int:
        return sum(r.total_tokens for r in self.runs)

    @property
    def total_cost_usd(self) -> float:
        return round(sum(r.total_cost_usd for r in self.runs), 6)

    @property
    def avg_e2e_latency(self) -> float:
        completed = [r for r in self.runs if r.workflow_success]
        if not completed:
            return 0.0
        return round(sum(r.end_to_end_latency_seconds for r in completed) / len(completed), 2)

    @property
    def avg_faithfulness(self) -> float:
        vals = [r.faithfulness_score for r in self.runs if r.workflow_success]
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    @property
    def avg_answer_relevancy(self) -> float:
        vals = [r.answer_relevancy for r in self.runs if r.workflow_success]
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    @property
    def total_guardrail_blocks(self) -> int:
        return sum(r.total_guardrail_blocks for r in self.runs)

    @property
    def guardrail_intervention_rate(self) -> float:
        total_checks = sum(
            (r.pathologist.guardrail_checks if r.pathologist else 0) +
            (r.extension_officer.guardrail_checks if r.extension_officer else 0)
            for r in self.runs
        )
        total_interventions = sum(
            r.total_guardrail_blocks + r.total_guardrail_warnings
            for r in self.runs
        )
        return round(total_interventions / total_checks, 3) if total_checks else 0.0

    def get_runs_as_dicts(self) -> list:
        return [r.to_summary_dict() for r in self.runs]

    def get_agent_rows(self) -> list:
        """Flat list of per-agent dicts for the agent-level table."""
        rows = []
        for r in self.runs:
            if r.pathologist:
                d = r.pathologist.to_dict()
                d["Run"] = r.run_id
                rows.append(d)
            if r.extension_officer:
                d = r.extension_officer.to_dict()
                d["Run"] = r.run_id
                rows.append(d)
        return rows
