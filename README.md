# 🌿 Agri Doctor — AI Crop Disease Advisor

An **Agentic AI** web application built with **Python + Streamlit**, powered by **IBM watsonx.ai**.  
Farmers upload a photo of an affected crop or leaf and two specialised AI agents diagnose the disease and generate a safe, practical advisory.

---

## 🏆 Hackathon Project

**Inter-Institute IBM Bob Hackathon with IBM SkillsBuild**  
Problem Statement: *Design and develop an Agentic AI application called Agri Doctor using Python and Streamlit, integrated with the IBM API.*

---

## 🤖 Two-Agent Architecture

```
Farmer uploads image
        │
        ▼
┌─────────────────────────────┐
│  Agent 1 — Pathologist      │  IBM watsonx.ai Vision LLM
│  mistralai/mistral-small-   │  → Disease diagnosis
│  3-1-24b-instruct-2503      │  → Symptoms, confidence, alternatives
└──────────────┬──────────────┘
               │  Sanitised diagnostic report
               ▼
┌─────────────────────────────┐
│  Agent 2 — Extension        │  IBM watsonx.ai Granite LLM
│  Officer                    │  → Farmer-friendly advisory
│  ibm/granite-4-h-small      │  → Remedies, prevention, safety
└──────────────┬──────────────┘
               │
               ▼
     Combined report + Download (.txt)
     Performance Dashboard (6 metric categories)
```

---

## ✨ Features

| Feature | Detail |
|---|---|
| 🔬 **Crop Disease Diagnosis** | Vision LLM analyses uploaded JPG/PNG for disease, symptoms, confidence |
| 📋 **Farmer Advisory** | Text LLM generates immediate actions, prevention tips, safe treatment options |
| 🛡️ **9-Category Guardrails** | Input, Output, Tool, Data, Security, Operational, HITL, Multi-Agent, Ethical |
| 📊 **Performance Dashboard** | 6 metric categories: Accuracy, RAG, Agentic AI, Safety, Performance & Cost |
| ⬇️ **Downloadable Report** | Full advisory + guardrail audit trail as `.txt` file |
| 🔐 **Secure by design** | Credentials only from `.env`; PII scrubbing; SHA-256 cross-agent integrity |

---

## 🗂️ Project Structure

```
AgriDoctor/
├── app.py                            # Streamlit entry point
├── requirements.txt                  # Python dependencies
├── .env.example                      # Credential template (copy to .env)
├── README.md
│
├── agents/
│   ├── pathologist_agent.py          # Agent 1 — Vision LLM crop diagnosis
│   └── extension_officer_agent.py   # Agent 2 — Text LLM farmer advisory
│
├── guardrails/                       # 9-category AI safety guardrails
│   ├── input_guardrails.py           # 1. Input validation, PII, injection detection
│   ├── output_guardrails.py          # 2. Hallucination, toxicity, dosage checks
│   ├── tool_guardrails.py            # 3. API auth, model allowlist
│   ├── data_guardrails.py            # 4. PII masking, data privacy
│   ├── security_guardrails.py        # 5. Jailbreak, SQL injection, integrity
│   ├── operational_guardrails.py     # 6. Token budget, timeouts, retries
│   ├── hitl_guardrails.py            # 7. Human-in-the-loop escalation
│   ├── multiagent_guardrails.py      # 8. Handoff validation, loop detection
│   └── ethical_guardrails.py        # 9. Bias, audit trail, explainability
│
└── utils/
    ├── ibm_client.py                 # IBM watsonx.ai credential + model factory
    ├── performance_tracker.py        # RunMetrics + SessionMetrics dataclasses
    ├── dashboard.py                  # 6-tab Streamlit performance dashboard
    └── report_formatter.py          # Plain-text advisory report generator
```

---

## ⚡ Quick Start

### 1. Prerequisites

- Python 3.10 or higher
- IBM Cloud account with Watson Studio / watsonx.ai service
- IBM watsonx.ai API Key and Project ID

### 2. Clone the repository

```bash
git clone https://github.com/Hiren-Modha/Agri-Doctors.git
cd Agri-Doctors/AgriDoctor
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure credentials

```bash
cp .env.example .env
```

Edit `.env` and fill in your real credentials:

```env
WATSONX_API_KEY=your_ibm_api_key_here
WATSONX_PROJECT_ID=your_project_id_here
WATSONX_URL=https://us-south.ml.cloud.ibm.com
WATSONX_VISION_MODEL=mistralai/mistral-small-3-1-24b-instruct-2503
WATSONX_TEXT_MODEL=ibm/granite-4-h-small
```

> ⚠️ **Never commit your `.env` file.** It is listed in `.gitignore`.

### 5. Run the application

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 🛡️ Guardrail Categories

| # | Category | What it protects |
|---|---|---|
| 1 | **Input** | File MIME type, magic bytes, size limits, path traversal, PII/injection in filenames |
| 2 | **Output** | Min-length check, toxicity keywords, hallucination markers, fabricated dosage hard-block |
| 3 | **Tool** | API credential structural validation, model ID allowlist before every agent call |
| 4 | **Data** | PII masking (email, phone, ID) on cross-agent handoff message |
| 5 | **Security** | Prompt injection + jailbreak detection, SHA-256 cross-agent integrity, credential leakage scan |
| 6 | **Operational** | 120-second agent timeout, 8,000-token session budget, 20-call API limit, exponential retry |
| 7 | **HITL** | Mandatory/advisory escalation on low confidence, chemical treatment, or severe crop damage |
| 8 | **Multi-Agent** | Handoff structure validation, 5-run loop detection, advisory ↔ diagnosis consistency |
| 9 | **Ethical** | Bias/fairness language flags, prohibited content hard-block, AuditTrail, explainability note |

---

## 📊 Performance Dashboard Metrics

| Tab | Key Metrics |
|---|---|
| 🏠 Overview | Task completion rate, error rate, avg latency, total tokens/cost |
| 🎯 Accuracy & Quality | Format compliance, response length, ROUGE-1 F1, BLEU-1 |
| 🔗 RAG Metrics | Faithfulness, answer relevancy, groundedness |
| 🤖 Agentic AI | Task completion, handoff accuracy, inter-agent consistency, loop detection |
| 🛡️ Safety & Guardrails | Hallucination rate, toxicity, PII leakage, injection resistance, guardrail intervention rate |
| ⚡ Performance & Cost | Per-agent latency, tokens/sec, cost per request, session totals |

---

## 🔧 Configuration

All model names and limits are configurable via environment variables:

| Variable | Default | Description |
|---|---|---|
| `WATSONX_VISION_MODEL` | `mistralai/mistral-small-3-1-24b-instruct-2503` | Vision LLM for crop image analysis |
| `WATSONX_TEXT_MODEL` | `ibm/granite-4-h-small` | Text LLM for advisory generation |
| `WATSONX_URL` | `https://us-south.ml.cloud.ibm.com` | IBM watsonx.ai endpoint |

---

## ⚠️ Safety & Disclaimer

- All diagnoses are **AI-generated preliminary assessments only** — not confirmed laboratory diagnoses.
- Chemical treatment dosages are **never invented** by the AI — the system prompt and output guardrails explicitly prohibit this.
- When disease cannot be identified reliably, the application directs farmers to consult a local **agricultural extension officer** or **certified plant pathologist**.
- All agent outputs are screened for toxic content, credential leakage, and prohibited recommendations before display.

---

## 🧰 Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| UI Framework | Streamlit |
| LLM Provider | IBM watsonx.ai |
| Vision LLM | Mistral Small 3.1 (via IBM watsonx.ai) |
| Text LLM | IBM Granite 4 H Small |
| Guardrails | Custom Python (rule-based, no external library) |
| Credential Management | python-dotenv |
| Image Processing | Pillow |

---

## 📄 License

This project was developed as part of the IBM SkillsBuild Hackathon. All IBM watsonx.ai services are subject to [IBM Cloud Terms of Service](https://www.ibm.com/legal).

---

*🌿 Agri Doctor — Powered by IBM watsonx.ai | AI-generated preliminary assessment only*
