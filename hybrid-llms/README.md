# Hybrid LLM Routing & Governance POC

A small POC showing how different LLMs can be assigned work based on their core specialization, followed
by independent safety and quality validation. The codebase is organized as four layers — **data**,
**execution**, **evaluation** and **governance** — plus a shared **settings** layer, so each concern lives
in exactly one place and can be reasoned about independently.

## Flow

DOCX/PPTX
→ **Data**: Extract
→ **Execution**: Route by task type → Specialist LLM (with fallback)
→ **Governance**: Guardrail
→ **Evaluation**: Quality scoring
→ **Governance**: PASS / BLOCK / MANUAL REVIEW → Audit + Excel report + LLM call ledger

## Architecture

```text
hybrid-llms/
├── settings/                  # Shared config & logging — used by every layer
│   ├── config.json
│   ├── config.py
│   └── logger.py
│
├── data/                      # Data layer: turns input files into evaluable text
│   ├── document_processor.py    (extract_text)
│   ├── input_loader.py          (find_input_files, load_expected_answer, hash_text)
│   └── create_demo_inputs.py
│
├── execution/                 # Execution layer: routing + generation + LLM clients
│   ├── router.py                 (route_request)
│   ├── generator.py              (generate_response, incl. Gemini → server-model fallback)
│   └── clients/
│       ├── ollama_client.py
│       └── gemini_client.py
│
├── evaluation/                # Evaluation layer: independent quality scoring
│   └── evaluator.py              (evaluate_response)
│
├── governance/                # Governance layer: safety rules, final decision, reporting
│   ├── guardrail.py              (run_guardrail)
│   ├── decision.py               (make_final_decision)
│   └── report.py                 (write_audit_log, write_excel_report, append_call_ledger)
│
├── tests/
│   └── test_core.py
├── input/
├── logs/
├── output/
├── main.py                    # Orchestrator — wires the four layers together
├── requirements.txt
├── .env.example
└── README.md
```

Each layer only depends on the layers "below" it (governance and evaluation take an `ollama_client`
instance as a parameter rather than importing `execution.clients` themselves), which keeps the dependency
direction obvious: `main.py` is the only place that knows about every layer at once.

## Model specialization

- General/document reasoning → Gemini 3.8 Flash (falls back to a server Ollama model — see below)
- Coding/automation → Qwen3-Coder 30B
- SQL/database → SQLCoder 15B
- Independent guardrail → GPT-OSS 20B
- Independent quality evaluation → GPT-OSS 20B

The goal is not to use one model for every task. The router assigns each request to the model configured
for that task type.

## Gemini fallback

The GENERAL route normally calls Gemini. If Gemini is unavailable — quota exhaustion, a 503/UNAVAILABLE
response, or any other request failure — `generate_response()` in `execution/generator.py` catches the
error and retries the same prompt against a server-hosted Ollama model instead of failing the request
outright.

The fallback model is configured in `settings/config.json` under `models.general_fallback` and currently
points at `llama3:instruct` on the same internal Ollama server (`http://192.168.1.81:11434/`) already used
for the coding/SQL/guardrail/evaluation routes:

```json
"models": {
  "general_fallback": "llama3:instruct"
}
```

`llama3:instruct` was chosen because it's the fastest warm-latency model on that server (~1.9s) and is
already validated as a general-purpose baseline, so falling back adds the least possible latency on top of
an already-degraded path. Any of the other server-hosted generators can be swapped in by changing this one
config value — no code change required:

| Model | Family | Why it fits as a fallback |
|---|---|---|
| `llama3:instruct` (default) | Llama | Fastest warm latency; general-purpose baseline |
| `qwen2.5-coder:14b` | Qwen2 | Code-capable and general-QA verified |
| `gpt-oss:20b` | GPT-OSS | General-purpose; independent from the judge family |
| `qwen3-coder:30b` | Qwen3-MoE | Same model already used as guardrail/evaluation judge |

The audit record's `generation_provider` field records which path actually served the request
(`gemini`, `ollama`, or `ollama_fallback`), and a fallback is always logged as a warning so it's visible in
`logs/hybrid_llm_poc.log` and not silently absorbed. The failed Gemini attempt's own duration is still
captured in the LLM call ledger (see below) rather than discarded. If `models.general_fallback` is left
empty, a Gemini failure is reported as `TECHNICAL_ERROR` as before.

## Setup

### 1. Create a virtual environment

Windows:

```powershell
python -m venv .venv
.venv\\Scripts\\activate
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Configure Gemini

Copy `.env.example` to `.env` (at the project root) and add your Gemini API key:

```text
GEMINI_API_KEY=your_key
GEMINI_MODEL=gemini-3.8-flash
```

Do not commit `.env`.

### 4. Check Ollama

Make sure the configured internal Ollama URL (`settings/config.json` → `ollama.base_url`) is reachable and
these models are available:

- qwen3-coder:30b
- sqlcoder:15b
- gpt-oss:20b
- llama3:instruct (Gemini fallback)

### 5. Create demo inputs

Run from the project root so the `input/` folder resolves correctly:

```powershell
python data/create_demo_inputs.py
```

This creates four requests:

1. General architecture summary → Gemini
2. Python automation requirement → Qwen3-Coder
3. SQL read-only query → SQLCoder
4. Destructive SQL requirement → SQLCoder, then guardrail BLOCK/REVIEW

### 6. Run

```powershell
python main.py
```

Results:

```text
output/hybrid_llm_governance_report.xlsx   # Summary, Configuration, Audit, LLM Calls sheets
output/audit_log.jsonl                     # one JSON record per document
output/llm_call_ledger.csv                 # one row per individual LLM call, appended across runs
logs/hybrid_llm_poc.log
```

A per-phase timing table is also printed to the console at the end of every run (see below).

## Routing

The router is intentionally rule-based for this POC, using word-boundary keyword matching so a substring
inside an unrelated word (e.g. "programmatic") can't trigger a false-positive route:

- SQL/database wording → SQLCoder
- coding/automation wording → Qwen3-Coder
- both SQL and coding wording matched → `MANUAL_REVIEW` (ambiguous; no generation call is spent on it)
- everything else → Gemini (with server-model fallback)

This keeps routing deterministic and makes the specialization decision easy to explain and test.

## Guardrail

The guardrail has two layers:

- deterministic security rules for obvious dangerous patterns (destructive SQL, DELETE without WHERE,
  hardcoded secrets, private key material)
- GPT-OSS 20B as an independent LLM safety/policy reviewer

A rule match blocks instantly without spending a guardrail or evaluation LLM call — and is correctly
excluded from the LLM call ledger, since no LLM was actually called.

## Evaluation

GPT-OSS 20B independently evaluates, using an optional `<input-name>.json` sidecar's `expected_answer`
as a reference when one is provided (otherwise it evaluates against the source document only):

- correctness
- relevancy
- completeness
- hallucination

The application calculates the overall score instead of trusting the LLM's supplied overall score.

## Final decision

Safety always wins:

```text
Guardrail BLOCK → BLOCK
Guardrail REVIEW → MANUAL_REVIEW
Guardrail PASS + Evaluation PASS → PASS
Guardrail PASS + Evaluation REVIEW/FAIL → MANUAL_REVIEW
```

## Observability & LLM call timing

Every audit record carries a `run_id`, per-phase latency (`extraction_latency_seconds`,
`generation_latency_seconds`, `guardrail_latency_seconds`, `evaluation_latency_seconds`,
`total_latency_seconds`), and `input_hash`/`response_hash` content hashes so a result can be traced and
correlated without storing raw document/response text a second time. A technical/infrastructure failure
(network error, invalid LLM response, etc.) is recorded as `status: TECHNICAL_ERROR` /
`final_decision: TECHNICAL_ERROR`, kept separate from the `PASS` / `BLOCK` / `MANUAL_REVIEW` quality and
policy verdicts so the two can never be confused in the Excel report.

On top of the per-document summary, every **individual** LLM call (generation, guardrail, evaluation —
including a failed Gemini attempt before a fallback) is tracked as its own row with `run_id`, `input_file`,
`phase`, `provider`, `model`, `started_at` and `duration_seconds`:

- `output/llm_call_ledger.csv` — appended to on every run, so it builds a permanent, growing record of
  every call's timing that you can open in Excel or load with pandas to track speed over time.
- The `LLM Calls` sheet in the Excel report — the same rows, scoped to just the current run.
- A console table printed at the end of every run, e.g.:

  ```text
  LLM call timing summary (this run):
  Phase                 Calls   Min (s)   Avg (s)   P50 (s)   Max (s)
  ------------------------------------------------------------------
  generation                 3      6.01     28.65     12.65     67.59
  generation_fallback         1      1.89      1.89      1.89      1.89
  guardrail                   4     13.30     14.28     13.83     15.55
  evaluation                  3      8.38     14.06     13.29     18.13
  ```

## Scalability & performance

- `execution.file_concurrency` in `settings/config.json` controls how many input files are processed in
  parallel via a `ThreadPoolExecutor`. It defaults to `1` (sequential) — increase it only after confirming
  the target Ollama endpoint tolerates concurrent requests without a large per-call latency penalty.
- `execution.max_document_chars` caps how much extracted document text is sent to any single prompt,
  bounding worst-case latency and avoiding model context-length failures on large inputs.
- Both `OllamaClient` and `GeminiClient` reuse a single `requests.Session()` instead of opening a new
  connection per call.
- `ollama.keep_alive` (default `"30m"`) is sent on every Ollama call so the server keeps a model resident
  in memory between back-to-back calls — guardrail and evaluation call the same model on every single
  document, so this avoids repeated model reloads within a run.
- Gemini requests set `generationConfig.temperature = 0` for reproducible output across runs.

## Code structure

```text
hybrid-llms/
├── settings/
│   ├── __init__.py
│   ├── config.json
│   ├── config.py
│   └── logger.py
├── data/
│   ├── __init__.py
│   ├── document_processor.py
│   ├── input_loader.py
│   └── create_demo_inputs.py
├── execution/
│   ├── __init__.py
│   ├── router.py
│   ├── generator.py
│   └── clients/
│       ├── __init__.py
│       ├── ollama_client.py
│       └── gemini_client.py
├── evaluation/
│   ├── __init__.py
│   └── evaluator.py
├── governance/
│   ├── __init__.py
│   ├── guardrail.py
│   ├── decision.py
│   └── report.py
├── tests/
│   └── test_core.py
├── input/
├── logs/
├── output/
├── main.py
├── .env.example
├── requirements.txt
└── README.md
```
