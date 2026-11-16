"""
Agri Doctor — AI Crop Disease Advisor
Streamlit application entry point.

Run with:
    streamlit run app.py

Requires a .env file in this directory with:
    WATSONX_API_KEY, WATSONX_PROJECT_ID, WATSONX_URL,
    WATSONX_VISION_MODEL, WATSONX_TEXT_MODEL
"""

import time
import streamlit as st
from PIL import Image

from utils.ibm_client import get_vision_model, get_text_model, get_active_model_names, get_credentials
from agents.pathologist_agent import run_pathologist_agent
from agents.extension_officer_agent import run_extension_officer_agent
from utils.report_formatter import format_full_report
from utils.performance_tracker import (
    SessionMetrics, RunMetrics, AgentMetrics,
    _rouge1_f1, _bleu1, _response_length_score, _has_required_sections,
)
from utils.dashboard import render_dashboard

# ── Guardrails ────────────────────────────────────────────────────────────────
from guardrails import (
    validate_image_input, validate_filename,
    validate_diagnosis_output, validate_advisory_output,
    validate_api_call, validate_model_access,
    sanitize_for_handoff,
    detect_prompt_injection, detect_jailbreak,
    compute_message_digest, check_cross_agent_integrity, detect_credential_exposure,
    OperationalGuardrail, TokenBudget,
    hitl_escalation_check,
    validate_agent_handoff, detect_agent_loop, check_output_consistency,
    AuditTrail, check_bias_flags, check_prohibited_content, add_explainability_note,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agri Doctor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Session-state initialisation ──────────────────────────────────────────────
if "token_budget" not in st.session_state:
    st.session_state.token_budget = TokenBudget()
if "audit_trail" not in st.session_state:
    st.session_state.audit_trail = AuditTrail()
if "workflow_run_count" not in st.session_state:
    st.session_state.workflow_run_count = 0
if "session_metrics" not in st.session_state:
    st.session_state.session_metrics = SessionMetrics()
if "workflow_start_time" not in st.session_state:
    st.session_state.workflow_start_time = 0.0

budget: TokenBudget     = st.session_state.token_budget
audit: AuditTrail       = st.session_state.audit_trail
session: SessionMetrics = st.session_state.session_metrics

# ── Resolve model names once (used in credential check below) ─────────────────
try:
    model_names = get_active_model_names()
except Exception:
    model_names = {}

# ── Main tabs ─────────────────────────────────────────────────────────────────
main_tab, dashboard_tab = st.tabs(["🌿 Diagnosis", "📊 Performance Dashboard"])

# ══════════════════════════════════════════════════════════════════════════════
# MAIN DIAGNOSIS TAB
# ══════════════════════════════════════════════════════════════════════════════
with main_tab:
    st.title("🌿 Agri Doctor — AI Crop Disease Advisor")
    st.markdown(
        "Upload a photo of your crop or plant leaf and let the AI agents diagnose "
        "possible diseases and prepare a safe, practical advisory for you."
    )

    # AI disclaimer — always visible
    st.warning(
        "⚠️ **Important:** This tool provides an AI-generated preliminary assessment only. "
        "It is **not** a confirmed laboratory diagnosis. Always consult a qualified plant "
        "pathologist or your local agricultural extension officer before taking major action."
    )

    st.markdown("---")

    # ── Credential & tool guardrail check ────────────────────────────────────
    _credentials_ok = True
    try:
        creds = get_credentials()
        api_ok, api_reason = validate_api_call(
            creds["api_key"], creds["project_id"], creds["url"]
        )
        if not api_ok:
            st.error(f"🔑 **API Authorization Guardrail:** {api_reason}")
            audit.log("Tool", "validate_api_call", "BLOCK", api_reason)
            _credentials_ok = False
        else:
            audit.log("Tool", "validate_api_call", "PASS", "Credentials validated")
            vm_ok, vm_reason = validate_model_access(model_names.get("vision_model", ""), "vision")
            tm_ok, tm_reason = validate_model_access(model_names.get("text_model", ""), "text")
            if not vm_ok:
                st.warning(f"⚠️ **Model Access Guardrail:** {vm_reason}")
                audit.log("Tool", "validate_model_access", "WARN", vm_reason)
            else:
                audit.log("Tool", "validate_model_access", "PASS", "Vision model permitted")
            if not tm_ok:
                st.warning(f"⚠️ **Model Access Guardrail:** {tm_reason}")
                audit.log("Tool", "validate_model_access", "WARN", tm_reason)
            else:
                audit.log("Tool", "validate_model_access", "PASS", "Text model permitted")
    except ValueError as _cred_err:
        st.error(
            f"🔑 **Missing IBM watsonx.ai credentials**\n\n{_cred_err}\n\n"
            "Create a `.env` file in the `AgriDoctor/` folder using `.env.example` as a template."
        )
        audit.log("Tool", "validate_api_call", "BLOCK", str(_cred_err))
        _credentials_ok = False

    # ── Image upload ──────────────────────────────────────────────────────────
    col_upload, col_image = st.columns([1, 1])

    with col_upload:
        st.subheader("📷 Upload Crop Image")
        uploaded_file = st.file_uploader(
            "Choose a JPG, JPEG, or PNG photo of your affected crop or leaf",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear, well-lit photo of the affected plant part for best results.",
        )
        if uploaded_file is not None:
            st.success(f"✅ Image uploaded: **{uploaded_file.name}** ({uploaded_file.size:,} bytes)")

    with col_image:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded crop image", use_container_width=True)

    st.markdown("---")

    # ── Diagnose button ───────────────────────────────────────────────────────
    diagnose_button = st.button(
        "🔬 Diagnose Crop",
        type="primary",
        disabled=(not _credentials_ok),
        help="Upload a crop image above, then click to run the two-agent diagnosis workflow.",
    )

    # ── Main workflow ─────────────────────────────────────────────────────────
    if diagnose_button:
        if uploaded_file is None:
            st.error("❌ Please upload a crop image before running the diagnosis.")
            st.stop()

        uploaded_file.seek(0)
        image_bytes = uploaded_file.read()
        ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
        mime_type = "image/png" if ext == "png" else "image/jpeg"
        filename = uploaded_file.name

        # Initialise per-run metrics
        st.session_state.workflow_run_count += 1
        run_id = st.session_state.workflow_run_count
        current_run = RunMetrics(run_id=run_id, image_name=filename)
        workflow_start = time.perf_counter()

        # ── Guardrail 8: loop detection ───────────────────────────────────────
        loop_ok, loop_reason = detect_agent_loop(run_id)
        if not loop_ok:
            st.error(f"🔄 **Agent Loop Guardrail:** {loop_reason}")
            audit.log("Multi-Agent", "detect_agent_loop", "BLOCK", loop_reason)
            current_run.loop_detected = True
            session.add_run(current_run)
            st.stop()
        audit.log("Multi-Agent", "detect_agent_loop", "PASS", f"Run #{run_id}")

        # ── Guardrail 1: filename security ────────────────────────────────────
        fn_ok, fn_reason = validate_filename(filename)
        if not fn_ok:
            st.error(f"📁 **Input Guardrail (Filename):** {fn_reason}")
            audit.log("Input", "validate_filename", "BLOCK", fn_reason, filename)
            session.add_run(current_run)
            st.stop()
        audit.log("Input", "validate_filename", "PASS", "Filename OK", filename)

        # ── Guardrail 1: image bytes validation ───────────────────────────────
        img_ok, img_reason = validate_image_input(image_bytes, mime_type)
        if not img_ok:
            st.error(f"🖼️ **Input Guardrail (Image):** {img_reason}")
            audit.log("Input", "validate_image_input", "BLOCK", img_reason, image_bytes)
            session.add_run(current_run)
            st.stop()
        audit.log("Input", "validate_image_input", "PASS",
                  f"{mime_type}, {len(image_bytes):,} bytes", image_bytes)

        # ── Guardrail 5: security checks on filename ──────────────────────────
        inj_ok, inj_reason = detect_prompt_injection(filename)
        if not inj_ok:
            st.error(f"🔒 **Security Guardrail (Injection):** {inj_reason}")
            audit.log("Security", "detect_prompt_injection", "BLOCK", inj_reason, filename)
            session.add_run(current_run)
            st.stop()
        jb_ok, jb_reason = detect_jailbreak(filename)
        if not jb_ok:
            st.error(f"🔒 **Security Guardrail (Jailbreak):** {jb_reason}")
            audit.log("Security", "detect_jailbreak", "BLOCK", jb_reason, filename)
            session.add_run(current_run)
            st.stop()
        audit.log("Security", "detect_prompt_injection", "PASS", "No injection in filename")
        audit.log("Security", "detect_jailbreak", "PASS", "No jailbreak in filename")

        # ── Guardrail 6: token budget pre-flight for Agent 1 ─────────────────
        budget_ok, budget_reason = budget.record_call(input_tokens=800, output_tokens=0)
        if not budget_ok:
            st.error(f"⏱️ **Operational Guardrail (Budget):** {budget_reason}")
            audit.log("Operational", "token_budget", "BLOCK", budget_reason)
            session.add_run(current_run)
            st.stop()
        audit.log("Operational", "token_budget", "PASS", "Budget pre-flight OK for Agent 1")

        # ═════════════════════════════════════════════════════════════════════
        # AGENT 1 — Pathologist Agent
        # ═════════════════════════════════════════════════════════════════════
        p1_metrics = AgentMetrics(agent_name="Pathologist Agent")
        diagnosis = {}

        with OperationalGuardrail(agent_name="Pathologist Agent") as op_guard:
            with st.spinner("🔬 **Pathologist Agent** is analysing your crop image..."):
                try:
                    vision_model = get_vision_model()
                    diagnosis = run_pathologist_agent(image_bytes, vision_model, mime_type)
                    p1_metrics.success = True
                except ValueError as exc:
                    st.error(f"🔑 Credential error: {exc}")
                    audit.log("Tool", "run_pathologist_agent", "BLOCK", str(exc))
                    p1_metrics.error_message = str(exc)
                    budget.record_error()
                    current_run.pathologist = p1_metrics
                    session.add_run(current_run)
                    st.stop()
                except Exception as exc:
                    st.error(f"❌ Unexpected error loading the vision model: {exc}")
                    audit.log("Operational", "run_pathologist_agent", "BLOCK", str(exc))
                    p1_metrics.error_message = str(exc)
                    budget.record_error()
                    current_run.pathologist = p1_metrics
                    session.add_run(current_run)
                    st.stop()

        p1_metrics.end_time = time.perf_counter()
        # Estimate tokens from response length (actual tokens not returned by SDK in all versions)
        raw_len = len(diagnosis.get("raw_report", ""))
        p1_metrics.input_tokens  = 820          # typical vision call estimate
        p1_metrics.output_tokens = max(50, raw_len // 4)

        if op_guard.timed_out:
            st.warning(f"⏱️ **Operational Guardrail:** {op_guard.timeout_message}")
            audit.log("Operational", "execution_time", "WARN", op_guard.timeout_message)
        else:
            audit.log("Operational", "execution_time", "PASS",
                      f"Agent 1 completed in {op_guard.elapsed_seconds}s")

        if diagnosis.get("error"):
            st.error(f"❌ **Pathologist Agent error:** {diagnosis['error']}")
            audit.log("Multi-Agent", "run_pathologist_agent", "BLOCK", diagnosis["error"])
            p1_metrics.success = False
            p1_metrics.error_message = diagnosis["error"]
            current_run.pathologist = p1_metrics
            session.add_run(current_run)
            st.stop()

        if diagnosis.get("unclear_image"):
            st.warning(
                f"🖼️ **Image unclear or not a plant image.**\n\n"
                f"{diagnosis.get('unclear_reason', '')}\n\n"
                "Please upload a clearer, well-lit photograph and try again."
            )
            audit.log("Input", "unclear_image_check", "WARN",
                      diagnosis.get("unclear_reason", "Unclear image"))
            p1_metrics.task_completed = False
            current_run.pathologist = p1_metrics
            session.add_run(current_run)
            st.stop()

        p1_metrics.task_completed = True
        p1_metrics.goal_achieved  = True

        # ── Output guardrails on diagnosis ────────────────────────────────────
        raw_diag = diagnosis["raw_report"]
        diag_ok, diag_reason, diag_warnings = validate_diagnosis_output(raw_diag)
        p1_metrics.guardrail_checks += 1
        if not diag_ok:
            st.error(f"🔍 **Output Guardrail (Diagnosis):** {diag_reason}")
            audit.log("Output", "validate_diagnosis_output", "BLOCK", diag_reason)
            p1_metrics.guardrail_blocks += 1
            current_run.pathologist = p1_metrics
            session.add_run(current_run)
            st.stop()
        for w in diag_warnings:
            st.warning(f"⚠️ **Output Warning:** {w}")
            audit.log("Output", "validate_diagnosis_output", "WARN", w)
            p1_metrics.guardrail_warnings += 1
            if "hallucin" in w.lower() or "confident" in w.lower():
                p1_metrics.hallucination_flagged = True
        if not diag_warnings:
            audit.log("Output", "validate_diagnosis_output", "PASS", "Diagnosis output validated")

        # Compute quality metrics for Agent 1
        p1_metrics.format_compliance_score = _has_required_sections(
            raw_diag, ["SUSPECTED DISEASE", "OBSERVED SYMPTOMS", "CONFIDENCE LEVEL"]
        )
        p1_metrics.response_length_score = _response_length_score(raw_diag)

        # Ethical guardrail
        eth_ok, eth_reason = check_prohibited_content(raw_diag)
        p1_metrics.guardrail_checks += 1
        if not eth_ok:
            st.error(f"⚖️ **Ethical Guardrail:** {eth_reason}")
            audit.log("Ethical", "check_prohibited_content", "BLOCK", eth_reason)
            p1_metrics.guardrail_blocks += 1
            current_run.pathologist = p1_metrics
            session.add_run(current_run)
            st.stop()
        audit.log("Ethical", "check_prohibited_content", "PASS", "No prohibited content in diagnosis")

        # Security: credential leakage
        cred_ok, cred_reason = detect_credential_exposure(raw_diag)
        p1_metrics.guardrail_checks += 1
        if not cred_ok:
            st.error(f"🔒 **Security Guardrail:** {cred_reason}")
            audit.log("Security", "detect_credential_exposure", "BLOCK", cred_reason)
            p1_metrics.guardrail_blocks += 1
            current_run.pathologist = p1_metrics
            session.add_run(current_run)
            st.stop()
        audit.log("Security", "detect_credential_exposure", "PASS", "No credentials in diagnosis")

        # Multi-agent: handoff validation
        handoff_ok, handoff_reason = validate_agent_handoff(diagnosis)
        p1_metrics.guardrail_checks += 1
        if not handoff_ok:
            st.error(f"🔄 **Multi-Agent Guardrail (Handoff):** {handoff_reason}")
            audit.log("Multi-Agent", "validate_agent_handoff", "BLOCK", handoff_reason)
            p1_metrics.guardrail_blocks += 1
            current_run.pathologist = p1_metrics
            session.add_run(current_run)
            st.stop()
        audit.log("Multi-Agent", "validate_agent_handoff", "PASS", "Handoff validated")
        p1_metrics.handoff_success = True

        # Data: PII scrub before handoff
        sanitized_report, pii_found = sanitize_for_handoff(raw_diag)
        p1_metrics.guardrail_checks += 1
        if pii_found:
            st.info(
                f"🔐 **Data Guardrail:** {len(pii_found)} PII item(s) redacted before handoff."
            )
            audit.log("Data", "sanitize_for_handoff", "WARN", f"PII: {pii_found}")
            p1_metrics.pii_detected = True
            p1_metrics.guardrail_warnings += 1
        else:
            audit.log("Data", "sanitize_for_handoff", "PASS", "No PII in handoff")

        handoff_digest = compute_message_digest(sanitized_report)
        audit.log("Security", "compute_message_digest", "PASS",
                  f"Digest: {handoff_digest[:12]}...")

        current_run.pathologist = p1_metrics
        current_run.handoff_accuracy = 1.0

        st.success("✅ Pathologist Agent completed the diagnosis.")

        # ── Guardrail 6: token budget for Agent 2 ─────────────────────────────
        budget_ok2, budget_reason2 = budget.record_call(input_tokens=1500, output_tokens=0)
        if not budget_ok2:
            st.error(f"⏱️ **Operational Guardrail (Budget):** {budget_reason2}")
            audit.log("Operational", "token_budget", "BLOCK", budget_reason2)
            session.add_run(current_run)
            st.stop()
        audit.log("Operational", "token_budget", "PASS", "Budget pre-flight OK for Agent 2")

        # ═════════════════════════════════════════════════════════════════════
        # AGENT 2 — Extension Officer Agent
        # ═════════════════════════════════════════════════════════════════════
        p2_metrics = AgentMetrics(agent_name="Extension Officer Agent")
        p2_metrics.handoff_success = True
        advisory = {}

        with OperationalGuardrail(agent_name="Extension Officer Agent") as op_guard2:
            with st.spinner("📋 **Extension Officer Agent** is preparing your advisory..."):
                try:
                    text_model = get_text_model()
                    advisory = run_extension_officer_agent(sanitized_report, text_model)
                    p2_metrics.success = True
                except ValueError as exc:
                    st.error(f"🔑 Credential error: {exc}")
                    audit.log("Tool", "run_extension_officer_agent", "BLOCK", str(exc))
                    p2_metrics.error_message = str(exc)
                    budget.record_error()
                    current_run.extension_officer = p2_metrics
                    session.add_run(current_run)
                    st.stop()
                except Exception as exc:
                    st.error(f"❌ Unexpected error loading the text model: {exc}")
                    audit.log("Operational", "run_extension_officer_agent", "BLOCK", str(exc))
                    p2_metrics.error_message = str(exc)
                    budget.record_error()
                    current_run.extension_officer = p2_metrics
                    session.add_run(current_run)
                    st.stop()

        p2_metrics.end_time = time.perf_counter()
        raw_adv = advisory.get("raw_advisory", "")
        p2_metrics.input_tokens  = max(100, len(sanitized_report) // 4)
        p2_metrics.output_tokens = max(50, len(raw_adv) // 4)

        if op_guard2.timed_out:
            st.warning(f"⏱️ **Operational Guardrail:** {op_guard2.timeout_message}")
            audit.log("Operational", "execution_time", "WARN", op_guard2.timeout_message)
        else:
            audit.log("Operational", "execution_time", "PASS",
                      f"Agent 2 completed in {op_guard2.elapsed_seconds}s")

        if advisory.get("error"):
            st.error(f"❌ **Extension Officer Agent error:** {advisory['error']}")
            audit.log("Multi-Agent", "run_extension_officer_agent", "BLOCK", advisory["error"])
            p2_metrics.success = False
            p2_metrics.error_message = advisory["error"]
            current_run.extension_officer = p2_metrics
            session.add_run(current_run)
            st.stop()

        p2_metrics.task_completed = True
        p2_metrics.goal_achieved  = True

        # Security: cross-agent integrity check
        integrity_ok, integrity_reason = check_cross_agent_integrity(
            sanitized_report, handoff_digest
        )
        p2_metrics.guardrail_checks += 1
        if not integrity_ok:
            st.error(f"🔒 **Security Guardrail (Integrity):** {integrity_reason}")
            audit.log("Security", "check_cross_agent_integrity", "BLOCK", integrity_reason)
            p2_metrics.guardrail_blocks += 1
            current_run.extension_officer = p2_metrics
            session.add_run(current_run)
            st.stop()
        audit.log("Security", "check_cross_agent_integrity", "PASS",
                  "Cross-agent message integrity verified")

        # Output guardrails on advisory
        adv_ok, adv_reason, adv_warnings = validate_advisory_output(raw_adv)
        p2_metrics.guardrail_checks += 1
        if not adv_ok:
            st.error(f"🔍 **Output Guardrail (Advisory):** {adv_reason}")
            audit.log("Output", "validate_advisory_output", "BLOCK", adv_reason)
            p2_metrics.guardrail_blocks += 1
            current_run.extension_officer = p2_metrics
            session.add_run(current_run)
            st.stop()
        for w in adv_warnings:
            st.warning(f"⚠️ **Output Warning:** {w}")
            audit.log("Output", "validate_advisory_output", "WARN", w)
            p2_metrics.guardrail_warnings += 1
            if "hallucin" in w.lower() or "confident" in w.lower():
                p2_metrics.hallucination_flagged = True
        if not adv_warnings:
            audit.log("Output", "validate_advisory_output", "PASS", "Advisory output validated")

        # Quality metrics for Agent 2
        p2_metrics.format_compliance_score = _has_required_sections(
            raw_adv, ["IMMEDIATE ACTIONS", "SAFETY WARNINGS"]
        )
        p2_metrics.response_length_score = _response_length_score(raw_adv)
        # RAG: ROUGE between advisory and diagnosis report as faithfulness proxy
        p2_metrics.rouge1_f1 = _rouge1_f1(raw_adv, raw_diag)
        p2_metrics.bleu1     = _bleu1(raw_adv, raw_diag)

        # Ethical: prohibited content in advisory
        eth_ok2, eth_reason2 = check_prohibited_content(raw_adv)
        p2_metrics.guardrail_checks += 1
        if not eth_ok2:
            st.error(f"⚖️ **Ethical Guardrail:** {eth_reason2}")
            audit.log("Ethical", "check_prohibited_content", "BLOCK", eth_reason2)
            p2_metrics.guardrail_blocks += 1
            current_run.extension_officer = p2_metrics
            session.add_run(current_run)
            st.stop()
        audit.log("Ethical", "check_prohibited_content", "PASS", "No prohibited content in advisory")

        # Security: credential leakage in advisory
        cred_ok2, cred_reason2 = detect_credential_exposure(raw_adv)
        p2_metrics.guardrail_checks += 1
        if not cred_ok2:
            st.error(f"🔒 **Security Guardrail:** {cred_reason2}")
            audit.log("Security", "detect_credential_exposure", "BLOCK", cred_reason2)
            p2_metrics.guardrail_blocks += 1
            current_run.extension_officer = p2_metrics
            session.add_run(current_run)
            st.stop()
        audit.log("Security", "detect_credential_exposure", "PASS", "No credentials in advisory")

        # Ethical: bias
        bias_found, bias_warnings = check_bias_flags(raw_adv)
        p2_metrics.guardrail_checks += 1
        for bw in bias_warnings:
            st.warning(f"⚖️ **Ethical Guardrail (Bias):** {bw}")
            audit.log("Ethical", "check_bias_flags", "WARN", bw)
            p2_metrics.guardrail_warnings += 1
        if not bias_found:
            audit.log("Ethical", "check_bias_flags", "PASS", "No bias flags in advisory")

        # Multi-agent: output consistency
        consistent, consistency_warnings = check_output_consistency(diagnosis, advisory)
        p2_metrics.guardrail_checks += 1
        for cw in consistency_warnings:
            st.warning(f"🔄 **Multi-Agent Guardrail (Consistency):** {cw}")
            audit.log("Multi-Agent", "check_output_consistency", "WARN", cw)
            p2_metrics.guardrail_warnings += 1
        if consistent:
            audit.log("Multi-Agent", "check_output_consistency", "PASS",
                      "Advisory consistent with diagnosis")

        current_run.inter_agent_consistency = 1.0 if consistent else 0.5

        # HITL escalation
        hitl = hitl_escalation_check(diagnosis, advisory)
        audit.log("HITL", "hitl_escalation_check",
                  "PASS" if not hitl["requires_human_review"] else "WARN",
                  f"Escalation level: {hitl['escalation_level']}")

        # Finalise run metrics
        current_run.extension_officer       = p2_metrics
        current_run.workflow_success         = True
        current_run.end_to_end_latency_seconds = round(time.perf_counter() - workflow_start, 2)

        # RAG metric estimates
        current_run.faithfulness_score  = p2_metrics.rouge1_f1
        current_run.answer_relevancy    = p2_metrics.format_compliance_score
        current_run.groundedness_score  = p1_metrics.format_compliance_score

        session.add_run(current_run)

        st.success("✅ Extension Officer Agent completed the advisory.")
        st.markdown("---")

        # ── HITL banner ───────────────────────────────────────────────────────
        if hitl["requires_human_review"]:
            if hitl["escalation_level"] == "mandatory":
                st.error(f"🚨 **HUMAN REVIEW REQUIRED**\n\n{hitl['expert_message']}")
                for reason in hitl["escalation_reasons"]:
                    st.error(f"• {reason}")
            else:
                st.warning(f"👨‍🌾 **Expert Consultation Recommended**\n\n{hitl['expert_message']}")
                for reason in hitl["escalation_reasons"]:
                    st.warning(f"• {reason}")

        # ── Explainability note ───────────────────────────────────────────────
        with st.expander("🔍 Explainability — How did the AI reach this conclusion?",
                         expanded=False):
            expl = add_explainability_note(diagnosis, advisory)
            st.info(expl)
            audit.log("Ethical", "add_explainability_note", "PASS", "Explainability note generated")

        # ── Results display ───────────────────────────────────────────────────
        st.header("📊 Diagnosis & Advisory Results")

        with st.expander("🔬 Agent 1 — Pathologist: Disease Diagnosis", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 🦠 Suspected Disease")
                disease    = diagnosis.get("disease", "Not identified")
                confidence = diagnosis.get("confidence", "Not stated")
                if "unable to determine" in disease.lower():
                    st.error(f"**{disease}**")
                    st.info("⚠️ Unable to identify disease reliably. Please consult a local expert.")
                else:
                    st.success(f"**{disease}**")
                st.markdown(f"**Confidence Level:** {confidence}")

                causes = diagnosis.get("causes", [])
                if causes:
                    st.markdown("#### 🔍 Possible Causes")
                    for cause in causes:
                        st.markdown(f"- {cause}")

            with col2:
                symptoms = diagnosis.get("symptoms", [])
                if symptoms:
                    st.markdown("#### 🌿 Observed Symptoms")
                    for symptom in symptoms:
                        st.markdown(f"- {symptom}")
                alternatives = diagnosis.get("alternative_diagnoses", [])
                if alternatives:
                    st.markdown("#### 🔄 Alternative Diagnoses")
                    for alt in alternatives:
                        st.markdown(f"- {alt}")

            notes = diagnosis.get("additional_notes", "").strip()
            if notes:
                st.markdown("#### 📝 Additional Notes")
                st.info(notes)

        with st.expander("📋 Agent 2 — Extension Officer: Farmer Advisory", expanded=True):
            col3, col4 = st.columns(2)
            with col3:
                immediate = advisory.get("immediate_actions", [])
                if immediate:
                    st.markdown("#### ⚡ Immediate Actions")
                    for action in immediate:
                        st.markdown(f"- {action}")
                crop_care = advisory.get("crop_care", [])
                if crop_care:
                    st.markdown("#### 🌱 Crop Care Recommendations")
                    for rec in crop_care:
                        st.markdown(f"- {rec}")

            with col4:
                preventive = advisory.get("preventive_measures", [])
                if preventive:
                    st.markdown("#### 🛡️ Preventive Measures")
                    for measure in preventive:
                        st.markdown(f"- {measure}")
                treatment = advisory.get("treatment_options", [])
                if treatment:
                    st.markdown("#### 💊 Treatment Options")
                    for option in treatment:
                        st.markdown(f"- {option}")
                    st.caption(
                        "⚠️ Always read the product label for correct dosage. "
                        "Follow local agricultural regulations."
                    )

            safety = advisory.get("safety_warnings", [])
            if safety:
                st.markdown("#### ⚠️ Safety Warnings")
                for warning in safety:
                    st.warning(f"🔶 {warning}")

            expert_rec = advisory.get("expert_recommendation", "").strip()
            if expert_rec:
                st.markdown("#### 👨‍🌾 Expert Recommendation")
                st.info(expert_rec)

        # ── Guardrails activity summary ───────────────────────────────────────
        with st.expander("🛡️ Guardrails Activity — This Diagnosis Run", expanded=False):
            g_summary = audit.get_summary()
            gc1, gc2, gc3, gc4 = st.columns(4)
            gc1.metric("Total Checks", g_summary["total"])
            gc2.metric("✅ Passed",    g_summary["pass"])
            gc3.metric("⚠️ Warnings",  g_summary["warn"])
            gc4.metric("🚫 Blocked",   g_summary["block"])
            st.markdown("**Session Budget**")
            bs = budget.summary()
            bc1, bc2, bc3 = st.columns(3)
            bc1.metric("API Calls",    bs["api_calls"])
            bc2.metric("Total Tokens", bs["total_tokens"])
            bc3.metric("Est. Cost",    f"${bs['estimated_cost_usd']:.4f} USD*")
            st.caption("*Cost is an estimate based on approximate token pricing.")

        st.markdown("---")

        # ── Full report ───────────────────────────────────────────────────────
        st.header("📄 Full Advisory Report")
        report_text = format_full_report(diagnosis, advisory, audit)

        st.text_area(
            label="Full report (read-only)",
            value=report_text,
            height=400,
            disabled=True,
            label_visibility="collapsed",
        )

        report_filename = f"agri_doctor_report_{uploaded_file.name.rsplit('.', 1)[0]}.txt"
        st.download_button(
            label="⬇️ Download Advisory Report (.txt)",
            data=report_text.encode("utf-8"),
            file_name=report_filename,
            mime="text/plain",
            type="primary",
        )

        st.markdown("---")
        st.caption(
            "🌿 Agri Doctor — AI Crop Disease Advisor | Powered by IBM watsonx.ai  \n"
            "AI-generated preliminary assessment only. Not a confirmed laboratory diagnosis."
        )

# ══════════════════════════════════════════════════════════════════════════════
# PERFORMANCE DASHBOARD TAB
# ══════════════════════════════════════════════════════════════════════════════
with dashboard_tab:
    render_dashboard(session)
