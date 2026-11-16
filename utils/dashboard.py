"""
Performance Dashboard — LLM & Agent Evaluation Dashboard.

Renders a full six-tab Streamlit performance dashboard using only native
Streamlit widgets (st.metric, st.dataframe, st.progress, st.table).

Tabs:
  1. Overview             — headline KPIs for the session
  2. Accuracy & Quality   — ROUGE, BLEU, format compliance, length score
  3. RAG Metrics          — faithfulness, relevancy, groundedness
  4. Agentic AI Metrics   — task completion, handoff, coordination, loop detection
  5. Safety & Guardrails  — hallucination rate, toxicity, PII, injection resistance
  6. Performance & Cost   — latency, tokens, cost, throughput, error rate

Call render_dashboard(st, session_metrics) after at least one run is recorded.
"""

import streamlit as st
from utils.performance_tracker import SessionMetrics, RunMetrics


# ── Colour helpers for metric deltas ─────────────────────────────────────────
def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _fmt_cost(usd: float) -> str:
    return f"${usd:.5f}"


def _progress_row(label: str, value: float, help_text: str = "") -> None:
    """Render a labelled progress bar for a 0.0–1.0 score."""
    cols = st.columns([3, 7])
    cols[0].markdown(f"**{label}**")
    cols[1].progress(min(max(value, 0.0), 1.0), text=f"{value:.3f}")
    if help_text:
        st.caption(help_text)


def render_dashboard(session: SessionMetrics) -> None:
    """
    Render the full LLM & Agent Performance Dashboard.

    Args:
        session: SessionMetrics instance from st.session_state.
    """
    st.header("📊 LLM & Agent Performance Dashboard")

    if session.run_count == 0:
        st.info(
            "No diagnosis runs have been completed yet. "
            "Upload a crop image and click **🔬 Diagnose Crop** to generate metrics."
        )
        return

    # Build data tables for display
    run_rows   = session.get_runs_as_dicts()
    agent_rows = session.get_agent_rows()

    latest: RunMetrics = session.runs[-1]
    p1 = latest.pathologist
    p2 = latest.extension_officer

    # ── Tab layout ────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "🏠 Overview",
        "🎯 Accuracy & Quality",
        "🔗 RAG Metrics",
        "🤖 Agentic AI",
        "🛡️ Safety & Guardrails",
        "⚡ Performance & Cost",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — Overview
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[0]:
        st.subheader("Session Overview")
        st.caption(f"Total workflow runs this session: **{session.run_count}**")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("✅ Task Completion Rate",   _pct(session.task_completion_rate),
                  help="Percentage of workflow runs that completed successfully")
        c2.metric("❌ Error Rate",             _pct(session.error_rate),
                  help="Percentage of runs that failed")
        c3.metric("⏱️ Avg E2E Latency",       f"{session.avg_e2e_latency}s",
                  help="Average end-to-end time per completed run")
        c4.metric("🪙 Total Tokens Used",      f"{session.total_tokens:,}",
                  help="Combined input + output tokens across all runs")
        c5.metric("💰 Total Est. Cost",        _fmt_cost(session.total_cost_usd),
                  help="Estimated cost (approximate) — verify with IBM Cloud billing")

        st.markdown("---")
        st.subheader("Latest Run — Agent Summary")

        if p1 and p2:
            la1, la2, la3 = st.columns(3)
            la1.metric("🔬 Agent 1 Latency",    f"{p1.latency_seconds}s")
            la2.metric("📋 Agent 2 Latency",    f"{p2.latency_seconds}s")
            la3.metric("🔄 E2E Latency",        f"{latest.end_to_end_latency_seconds}s")

            lb1, lb2, lb3 = st.columns(3)
            lb1.metric("Agent 1 Tokens",   f"{p1.total_tokens:,}")
            lb2.metric("Agent 2 Tokens",   f"{p2.total_tokens:,}")
            lb3.metric("Run Cost",         _fmt_cost(latest.total_cost_usd))

        st.markdown("---")
        st.subheader("All Runs Summary")
        if run_rows:
            st.dataframe(run_rows, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — Accuracy & Quality
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[1]:
        st.subheader("Accuracy & Quality Metrics — Latest Run")
        st.caption(
            "These metrics evaluate response completeness and format adherence. "
            "ROUGE and BLEU scores are 0.0 when no reference answer is available "
            "(self-diagnosis mode without a ground-truth label)."
        )

        if p1:
            st.markdown("#### 🔬 Pathologist Agent (Agent 1)")
            q1, q2, q3, q4 = st.columns(4)
            q1.metric("Format Compliance",  f"{p1.format_compliance_score:.2f}",
                      help="Fraction of required output sections present (0–1)")
            q2.metric("Length Score",       f"{p1.response_length_score:.2f}",
                      help="Heuristic: 1.0 for 50–1500 word responses")
            q3.metric("ROUGE-1 F1",         f"{p1.rouge1_f1:.4f}",
                      help="Unigram overlap with reference (0 when no reference)")
            q4.metric("BLEU-1",             f"{p1.bleu1:.4f}",
                      help="Unigram precision vs reference (0 when no reference)")

            _progress_row("Format Compliance",  p1.format_compliance_score,
                          "1.0 = all required sections (SUSPECTED DISEASE, OBSERVED SYMPTOMS, CONFIDENCE LEVEL) present")
            _progress_row("Response Length Quality", p1.response_length_score,
                          "1.0 = response is 50–1500 words (ideal for a structured diagnosis)")

        if p2:
            st.markdown("#### 📋 Extension Officer Agent (Agent 2)")
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Format Compliance",  f"{p2.format_compliance_score:.2f}")
            r2.metric("Length Score",       f"{p2.response_length_score:.2f}")
            r3.metric("ROUGE-1 F1",         f"{p2.rouge1_f1:.4f}")
            r4.metric("BLEU-1",             f"{p2.bleu1:.4f}")

            _progress_row("Format Compliance", p2.format_compliance_score,
                          "1.0 = all required sections (IMMEDIATE ACTIONS, SAFETY WARNINGS) present")
            _progress_row("Response Length Quality", p2.response_length_score)

        st.markdown("---")
        st.subheader("Per-Agent Quality — All Runs")
        if agent_rows:
            quality_cols = ["Run", "Agent", "Format Score", "Length Score", "ROUGE-1 F1", "BLEU-1"]
            filtered = [{k: row[k] for k in quality_cols if k in row} for row in agent_rows]
            st.dataframe(filtered, use_container_width=True)

        st.caption(
            "📌 Note: Precision, Recall, F1, Exact Match, and Semantic Similarity require "
            "a labelled ground-truth dataset, which is not available in self-service mode. "
            "ROUGE and BLEU are proxy metrics computed without a reference answer."
        )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — RAG Evaluation Metrics
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[2]:
        st.subheader("RAG Evaluation Metrics — Latest Run")
        st.caption(
            "These metrics measure whether the advisory is grounded in the diagnosis "
            "and relevant to the identified disease. Scores are heuristically estimated "
            "from the agent outputs without an external reference corpus."
        )

        if latest:
            rg1, rg2, rg3 = st.columns(3)
            rg1.metric("🎯 Faithfulness",    f"{latest.faithfulness_score:.2f}",
                       help="Advisory statements grounded in the diagnosis report (1.0=fully faithful)")
            rg2.metric("💡 Answer Relevancy", f"{latest.answer_relevancy:.2f}",
                       help="Advisory relevance to the diagnosed disease/symptoms")
            rg3.metric("⚓ Groundedness",    f"{latest.groundedness_score:.2f}",
                       help="Diagnosis grounded in image observations (not fabricated)")

            st.markdown("#### Score Breakdown")
            _progress_row("Faithfulness",
                          latest.faithfulness_score,
                          "High = Extension Officer advisory matches the Pathologist's findings")
            _progress_row("Answer Relevancy",
                          latest.answer_relevancy,
                          "High = Advisory actions are appropriate for the identified disease")
            _progress_row("Groundedness",
                          latest.groundedness_score,
                          "High = Diagnosis references visible symptoms; not speculative")

        st.markdown("---")
        st.subheader("RAG Metrics — All Runs")
        if run_rows:
            rag_cols = ["Run", "Timestamp", "Faithfulness", "Relevancy", "Groundedness", "Workflow OK"]
            filtered = [{k: row[k] for k in rag_cols if k in row} for row in run_rows]
            st.dataframe(filtered, use_container_width=True)

        with st.expander("ℹ️ How are RAG metrics computed?"):
            st.markdown("""
**Faithfulness** is estimated by checking how many key terms from the Pathologist Agent's
diagnosis report (disease name, symptoms) appear in the Extension Officer Agent's advisory.
Higher overlap = higher faithfulness.

**Answer Relevancy** measures whether the advisory sections (immediate actions, treatment options)
are logically connected to the identified disease. It is approximated by ROUGE-1 overlap
between the advisory and the diagnosis report.

**Groundedness** is estimated by checking whether the Pathologist Agent's response references
observable symptoms (uses structured output sections) rather than making unsupported claims.
Format compliance score is used as a proxy.

*These are heuristic estimates. For production evaluation, integrate a dedicated LLM-as-judge
or RAG evaluation framework such as RAGAS or DeepEval.*
            """)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4 — Agentic AI Metrics
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[3]:
        st.subheader("Agentic AI & Multi-Agent Evaluation — Latest Run")

        # Individual agent metrics
        if p1 or p2:
            ag1, ag2, ag3, ag4 = st.columns(4)
            ag1.metric("✅ Task Completion Rate", _pct(session.task_completion_rate),
                       help="Fraction of runs where both agents completed successfully")
            ag2.metric("🤝 Handoff Accuracy",
                       _pct(latest.handoff_accuracy) if latest else "N/A",
                       help="1.0 = Pathologist → Extension Officer handoff succeeded with valid data")
            ag3.metric("🔗 Inter-Agent Consistency",
                       _pct(latest.inter_agent_consistency) if latest else "N/A",
                       help="Advisory topics consistent with diagnosis findings")
            ag4.metric("🔄 Loop Detected",
                       "Yes ⚠️" if latest.loop_detected else "No ✅",
                       help="Whether an agent loop was detected this run")

        st.markdown("---")

        if p1:
            st.markdown("#### 🔬 Agent 1 — Pathologist Agent")
            pa1, pa2, pa3 = st.columns(3)
            pa1.metric("Task Completed",    "✅ Yes" if p1.task_completed  else "❌ No")
            pa2.metric("Goal Achieved",     "✅ Yes" if p1.goal_achieved   else "❌ No")
            pa3.metric("Tool Exec Success", "✅ Yes" if p1.success         else "❌ No")

        if p2:
            st.markdown("#### 📋 Agent 2 — Extension Officer Agent")
            pb1, pb2, pb3 = st.columns(3)
            pb1.metric("Task Completed",    "✅ Yes" if p2.task_completed  else "❌ No")
            pb2.metric("Goal Achieved",     "✅ Yes" if p2.goal_achieved   else "❌ No")
            pb3.metric("Handoff Received",  "✅ Yes" if p2.handoff_success else "❌ No")

        st.markdown("---")
        st.subheader("Multi-Agent System — All Runs")

        if run_rows:
            ma_cols = ["Run", "Workflow OK", "E2E Latency(s)", "Handoff OK",
                       "Consistency", "GR Blocks", "GR Warnings"]
            filtered = [{k: row[k] for k in ma_cols if k in row} for row in run_rows]
            st.dataframe(filtered, use_container_width=True)

        st.markdown("---")
        st.subheader("Agent Loop Detection History")
        loop_runs = [r for r in session.runs if r.loop_detected]
        if loop_runs:
            st.warning(f"⚠️ Agent loops were detected in {len(loop_runs)} run(s).")
            for r in loop_runs:
                st.markdown(f"- Run #{r.run_id} at {r.timestamp}")
        else:
            st.success("✅ No agent loops detected across all runs in this session.")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 5 — Safety & Guardrail Metrics
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[4]:
        st.subheader("Safety & Guardrail Evaluation — Session")

        sg1, sg2, sg3, sg4 = st.columns(4)
        sg1.metric("🚫 Total Guardrail Blocks",   session.total_guardrail_blocks,
                   help="Runs where guardrails hard-blocked a response or input")
        sg2.metric("⚠️ Guardrail Intervention Rate", _pct(session.guardrail_intervention_rate),
                   help="Fraction of guardrail checks that resulted in a block or warning")
        sg3.metric("Task Completion Rate",         _pct(session.task_completion_rate))
        sg4.metric("Error Rate",                   _pct(session.error_rate))

        st.markdown("---")
        st.subheader("Safety Flags — Latest Run")

        if p1 or p2:
            agents_to_check = [(p1, "Pathologist Agent"), (p2, "Extension Officer Agent")]
            for agent, name in agents_to_check:
                if agent is None:
                    continue
                st.markdown(f"**{name}**")
                sf1, sf2, sf3, sf4, sf5 = st.columns(5)
                sf1.metric("Hallucination Flagged", "⚠️ Yes" if agent.hallucination_flagged  else "✅ No")
                sf2.metric("Toxicity Flagged",      "⚠️ Yes" if agent.toxicity_flagged       else "✅ No")
                sf3.metric("PII Detected",          "⚠️ Yes" if agent.pii_detected           else "✅ No")
                sf4.metric("Injection Blocked",     "⚠️ Yes" if agent.prompt_injection_blocked else "✅ No")
                sf5.metric("Guardrail Blocks",      agent.guardrail_blocks)
                st.markdown("")

        st.markdown("---")
        st.subheader("Safety Metrics — All Runs (Agent Level)")
        if agent_rows:
            safe_cols = ["Run", "Agent", "Guardrail Blocks", "Guardrail Warnings",
                         "Hallucination Flag", "PII Detected", "Toxicity Flagged"]
            filtered = [{k: row[k] for k in safe_cols if k in row} for row in agent_rows]
            st.dataframe(filtered, use_container_width=True)

        with st.expander("ℹ️ Safety metric definitions"):
            st.markdown("""
| Metric | How it is measured in Agri Doctor |
|---|---|
| **Hallucination Rate** | Output guardrail detected over-confident language markers |
| **Toxicity Score** | Output guardrail matched toxic keyword patterns |
| **PII Leakage Rate** | Data guardrail detected and masked PII in cross-agent handoff |
| **Prompt Injection Resistance** | Security guardrail blocked injection attempts in filename/input |
| **Policy Violation Rate** | Ethical guardrail detected prohibited content |
| **Guardrail Intervention Rate** | (Blocks + Warnings) / Total guardrail checks |
            """)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 6 — Performance & Cost
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[5]:
        st.subheader("Performance & Cost Metrics — Latest Run")

        if p1 and p2:
            pc1, pc2, pc3 = st.columns(3)
            pc1.metric("🔬 Agent 1 Latency",    f"{p1.latency_seconds}s")
            pc2.metric("📋 Agent 2 Latency",    f"{p2.latency_seconds}s")
            pc3.metric("🔄 End-to-End Latency", f"{latest.end_to_end_latency_seconds}s")

            pt1, pt2, pt3 = st.columns(3)
            pt1.metric("Agent 1 — Input Tokens",  f"{p1.input_tokens:,}")
            pt2.metric("Agent 1 — Output Tokens", f"{p1.output_tokens:,}")
            pt3.metric("Agent 1 — Tokens/sec",    f"{p1.tokens_per_second}")

            pt4, pt5, pt6 = st.columns(3)
            pt4.metric("Agent 2 — Input Tokens",  f"{p2.input_tokens:,}")
            pt5.metric("Agent 2 — Output Tokens", f"{p2.output_tokens:,}")
            pt6.metric("Agent 2 — Tokens/sec",    f"{p2.tokens_per_second}")

            pc4, pc5, pc6 = st.columns(3)
            pc4.metric("💰 Agent 1 Cost (est.)",  _fmt_cost(p1.cost_usd))
            pc5.metric("💰 Agent 2 Cost (est.)",  _fmt_cost(p2.cost_usd))
            pc6.metric("💰 Run Total Cost (est.)", _fmt_cost(latest.total_cost_usd))

        st.markdown("---")
        st.subheader("Session Performance Summary")

        ps1, ps2, ps3, ps4 = st.columns(4)
        ps1.metric("Total Runs",           session.run_count)
        ps2.metric("Successful Runs",      session.success_count)
        ps3.metric("Session Total Tokens", f"{session.total_tokens:,}")
        ps4.metric("Session Total Cost",   _fmt_cost(session.total_cost_usd))

        pss1, pss2 = st.columns(2)
        pss1.metric("Avg E2E Latency",    f"{session.avg_e2e_latency}s")
        pss2.metric("Error Rate",         _pct(session.error_rate))

        st.markdown("---")
        st.subheader("Per-Run Performance Table")
        if run_rows:
            perf_cols = ["Run", "Timestamp", "E2E Latency(s)", "Total Tokens",
                         "Total Cost($)", "Workflow OK"]
            filtered = [{k: row[k] for k in perf_cols if k in row} for row in run_rows]
            st.dataframe(filtered, use_container_width=True)

        st.subheader("Per-Agent Performance Table — All Runs")
        if agent_rows:
            agent_perf_cols = ["Run", "Agent", "Latency (s)", "Input Tokens",
                               "Output Tokens", "Tokens/s", "Cost (USD)", "Success"]
            filtered = [{k: row[k] for k in agent_perf_cols if k in row} for row in agent_rows]
            st.dataframe(filtered, use_container_width=True)

        st.caption(
            "💡 Cost estimates use approximate token pricing and may differ from actual IBM Cloud billing. "
            "Time to First Token (TTFT) is not measurable via the synchronous watsonx.ai SDK. "
            "Throughput is calculated as completed runs per session duration."
        )
