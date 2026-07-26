# Navneet ContractAI v6
## Evidence-Grounded Enterprise Contract Intelligence

**Navneet Education Limited | 2026** — engine: LexForge Hybrid Intelligence (LangGraph + OpenRouter + evidence-grounded hybrid RAG)

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Put the single workspace key in .streamlit/secrets.toml
# OPENROUTER_API_KEY = "sk-or-v1-your-key-here"

# Optional managed vector backend (ChromaDB remains the local default)
export PINECONE_API_KEY=your-pinecone-key
export PINECONE_INDEX_HOST=https://your-index.svc.region.pinecone.io
export PINECONE_NAMESPACE=contracts

# 3. Run
streamlit run app.py
```

Get your free OpenRouter key at: https://openrouter.ai/keys

## Sign In

The public landing page (Home / About / Contact) is open. All AI tools live behind login.

| Field    | Value      |
|----------|------------|
| Username | `snehal1`  |
| Password | `snehal123`|

New users can self-register from the **Create Account** tab. After login you get a sidebar
dashboard with: Dashboard, AI Analyzer, AI Contract Writer, AI Chatbot, AI Agent, RAG Chat,
Compare, History, Settings and Logout. The light/dark toggle appears only after sign-in.

### Demo Workspace

The database is seeded once with realistic team usage: 8 analyzed demo contracts, 5 complete
editable drafts, contract Q&A history, reusable legal clauses, owners and a recent-activity feed.
The seed is idempotent and does not overwrite user-created data or recreate deleted demo records.

| Demo role | Username | Password |
|-----------|----------|----------|
| Legal Manager | `priya.shah` | `demo123` |
| Procurement | `arjun.mehta` | `demo123` |
| Compliance | `neha.kulkarni` | `demo123` |
| Business User | `rohan.patel` | `demo123` |

Demo contracts are illustrative only and are not legal advice or execution-ready without review.

### Branding
- Logo lives in `assets/`. Drop your official `navneet_logo.png` there and it is used
  automatically (falls back to the bundled `navneet_logo.svg`).

---

## Pages

| Page | Description |
|------|-------------|
| Home | Landing page with hero, capabilities and system architecture |
| Architecture | Public engineering overview for analysis, retrieval, MCP and evidence controls |
| Knowledge and Retrieval | PDF/TXT/Markdown ingestion, vector indexing and scored retrieval diagnostics |
| AI Infrastructure | ChromaDB/Pinecone configuration and MCP Streamable HTTP registry |
| Features | Full technical specification of AI/ML stack |
| Analyze | Upload PDF/TXT → instant eight-dimension contract review |
| AI Writer | Generate 12+ contract types with full CRUD |
| RAG Chat | Contract-grounded Q&A with ChromaDB retrieval |
| History | Full CRUD for all analyzed contracts |
| Compare | Side-by-side two-contract comparison |
| About | Developer credits — SNEHAL LAXMAN JADHAV |
| Contact | Support and contact form |
| Settings | Fixed AI connection status, appearance and data management |

---

## AI Architecture

### Observable Hybrid Workflow

Every new analysis runs through a stateful four-stage pipeline:

1. **Deterministic extraction** — contract type, parties, duties, dates, clause coverage and control gaps
2. **LLM legal review** — GPT-OSS or Nemotron corrects and prioritizes the local screening
3. **Evidence grounding** — each material finding receives a section/page route, excerpt and grounding score
4. **Guardrail validation** — schema, score bounds, evidence coverage and cross-section consistency are checked

The UI exposes the execution trace, analysis fingerprint, calibrated confidence, risk decision matrix,
evidence ledger, cross-clause conflicts, human approval gate and a ranked negotiation playbook. If the
provider is unavailable, the same workflow returns a transparent local fallback instead of a fake AI result.

No model call is made during application startup, local parsing, evidence grounding, unit tests, or hybrid
retrieval. OpenRouter is called only when a signed-in user explicitly starts an AI workflow.

### Eight-Domain Analysis
1. **NER Node** — parties, dates, amounts, jurisdiction
2. **Obligation Node** — all obligations with priority/conditions
3. **Risk Node** — severity/likelihood/financial exposure/mitigation
4. **Deadline Node** — dates, terms, notice periods, auto-renewals
5. **Clause Node** — 20+ clause types, favorable/concerning rating
6. **Compliance Node** — GDPR/CCPA/HIPAA/SOC2 scoring
7. **Anomaly Node** — unusual/hidden/contradictory provisions
8. **Summary Node** — executive summary + PROCEED/DO NOT SIGN

With an OpenRouter key configured, the default action automatically combines local extraction with an
AI review. **Deep** mode asks the model for the complete eight-domain structured report; Quick and Standard
use a compact review over the deterministic baseline. Provider failures never replace the locally validated
report with a misleading `0/10` result.

### RAG Engine
- ChromaDB vector store (cosine HNSW)
- Hybrid retrieval: hashed dense similarity + exact legal-term/bigram reranking
- Local deterministic retrieval (no second AI model or API request)
- Pinecone managed vector index support through the REST data plane
- Namespace and document metadata filtering
- Retrieval lab with visible combined, semantic and lexical scores plus source/chunk metadata

### MCP Connectivity
- Session-scoped Streamable HTTP MCP server registry
- Optional bearer authentication
- JSON-RPC initialization health check
- Clear configured/connected status reporting

### Models (via OpenRouter)
- Default generation, drafting and chat: `openai/gpt-oss-20b:free`
- AI-enhanced contract analysis: choose `openai/gpt-oss-20b:free` or `nvidia/nemotron-3-super-120b-a12b:free`.
- OpenRouter reasoning is enabled per model, and returned `reasoning_details` are preserved unchanged for follow-up turns.
- Retrieval uses cost-free local hash embeddings with exact legal-term reranking and does not call another AI model.
- All remote AI calls use the same `OPENROUTER_API_KEY` from `.streamlit/secrets.toml`.
- Only the approved analysis models can be selected in the UI; API-key replacement remains unavailable.

### Database (CRUD)
- SQLite local DB (lexforge_v5.db)
- Tables: contracts, chat_history, generated_contracts, clause_snippets
- Full CRUD: create/read/update/delete + search + tagging

---

## File Structure

```
lexforge-v5/
├── app.py                  # Main Streamlit app (12 pages)
├── requirements.txt
├── run.sh
├── README.md
├── .streamlit/
│   └── config.toml         # Light mode default
├── core/
│   ├── agent.py            # LangChain + OpenRouter agents
│   ├── pdf_parser.py       # PDF extraction
│   └── rag_engine.py       # ChromaDB RAG
└── database/
    └── db.py               # SQLite CRUD layer
```

---

## Developer

```
╔══════════════════════════════════════════════════════════════╗
║  LexForge AI v5.0 — Enterprise Contract Intelligence        ║
║  Developer  : SNEHAL LAXMAN JADHAV                          ║
║  Role       : AI Engineer                                   ║
║  Company    : Navneet Education Limited                     ║
║  Year       : 2026                                          ║
║  AI analysis does not constitute legal advice.              ║
╚══════════════════════════════════════════════════════════════╝
```
