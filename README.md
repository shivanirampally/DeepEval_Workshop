# Project - Multi Generator Cross-LLM Evaluation
This project sends the same benchmark question, source context and evaluation prompt to three different Ollama generator models. Their responses are then evaluated using the same DeepEval metrics and a single independent judge model.

The purpose is to compare generator quality under the same conditions rather than allowing each model to use a different test or evaluation process.

## Architecture

The codebase is organized by concern, not by file type, so each layer can be
read, tested, and changed independently:

| Layer | Package | Responsibility |
|---|---|---|
| Data | `testdata/` | Loads and validates the benchmark workbook, builds the DeepEval `LLMTestCase` / per-row prompt. |
| Execution | `execution/` | Runs the run (`main.py`, the thin orchestrator with no metric/scoring/discovery logic of its own) and generates candidate responses (`generators/ollama_client.py`). |
| Evaluation | `evaluation/` | Metric definitions and judge construction (`metrics.py`), runs metrics against a response (`runner.py`), and turns results into PASS/REVIEW/FAIL/technical-error verdicts (`scorer.py`). |
| Observability & governance | `observability/` | Everything about visibility into a run: is the generator/judge roster valid and independent (`discovery.py`), what produced a report - run ID, dataset/prompt hashes, package versions, exact model digests (`provenance.py`), live console progress (`console.py`), what the results mean (`analysis.py` - recommendation and prompt-improvement suggestions), and the durable Excel report (`reporting/excel.py`). |
| Settings | `settings/` | `config.py` (env/JSON-driven constants) and `project_config.json` (the actual tunable values). |

Project-root files (`.env`, `README.md`, `requirements.txt`, `.gitignore`)
are **not** part of this layering - they're repo/tooling scaffolding that
`pip`, `git`, GitHub, and `python-dotenv`'s default lookup all expect at the
true root regardless of internal architecture, so they stay there rather
than moving into `settings/`.

## Model roles
Generators:
llama3:instruct -    general-purpose baseline
qwen2.5-coder:14b -  code-oriented candidate, verified to answer general QA cleanly
gpt-oss:20b -        general-purpose candidate

sqlcoder:15b was removed as a generator: it is fine-tuned to emit SQL/tables and
produced ungrounded table output on theoretical (non-SQL) questions, which both
lowered quality scores and added latency.

Each generator receives the same 10 benchmark test cases.

Judge:
qwen3-coder:30b

The judge was moved from gpt-oss:20b to qwen3-coder:30b: benchmarking showed
qwen3-coder:30b responds faster under the same conditions while remaining a
fully independent model family from all three generators (llama, qwen2,
gptoss vs. judge family qwen3moe).

The judge model is intentionally different from the selected generators. This provides an independent cross-LLM evaluation instead of allowing a generator to grade its own output.

### Optional: hosted judge (Gemini or Anthropic)
The Ollama judge above is serialized by the local server (it accepts one
request at a time), which is the dominant cost in every run. As an
alternative that avoids that bottleneck entirely, the judge can be pointed
at a hosted Gemini or Anthropic model instead of a local Ollama model:

1. Set `models.judge_provider` in `project_config.json` to `"gemini"` or
   `"anthropic"` (default: `"ollama"`). The model used is
   `models.gemini_judge_model` (default: `gemini-3.6-flash`) or
   `models.anthropic_judge_model` (default: `claude-haiku-4-5-20251001`).
2. Add `GOOGLE_API_KEY` and/or `ANTHROPIC_API_KEY` to `.env` (already
   git-ignored). Never paste API keys into a chat/AI assistant - add them
   to `.env` directly and only confirm they're present.

Generators stay on the local Ollama server either way - only the judge
moves. This trades local/offline evaluation for real request concurrency
(hosted APIs are not limited to one in-flight request), at the cost of a
per-call API fee and sending source/question/response text to the provider.
Gemini's free tier is rate-limited to a handful of requests/minute, which
is too slow for this project's call volume (~40 judge calls per test case)
- a paid tier is required for it to be worth using over Ollama.

## Evaluation approach
For every test case:

Same Question + Same Source + Same Prompt
                    |
        +-----------+-----------+
        |           |           |
        v           v           v
     Generator  Generator  Generator
        1           2           3
        |           |           |
        +-----------+-----------+
                    |
                    v
             Generated Responses
                    |
                    v
                  Judge
                    |
                6 metrics
                    |
                    v
             Quality Gate
                    |
                    v
          Generator Comparison
                    |
                    v
          Final Recommendation

This keeps the comparison fair:
same dataset
same source context
same prompt
same metric set
same thresholds
independent judge

## Metrics:
Metric	Purpose:
Hallucination: Detect unsupported claims or details that are not grounded in the supplied source
Faithfulness:	Check whether the response remains aligned with the supplied source
Correctness:	Check factual accuracy against the golden answer and source
Completeness:	Check whether important required information was missed
Answer Relevancy:	Check whether the answer directly addresses the question
Bias:	Check whether claims and wording remain neutral and evidence-based

Operational measures such as response time, technical errors and response stability are reported separately. They are not mixed into the semantic quality score.

## Score and verdict interpretation
DeepEval returns metric scores from 0.00 to 1.00.
The Excel report displays these scores as percentages for easier reading:
NOTE: A percentage is an evaluation score, not a literal probability that the same percentage of the response is correct or incorrect.
0.95 -> 95%
0.80 -> 80%
0.65 -> 65%

Hallucination example
If the Hallucination metric returns: 0.80

the report shows:
Hallucination evaluation score: 80%
Gap to ideal score:             20 percentage points
Verdict:                        REVIEW

It should not be interpreted as:
20% of the response is hallucinated

The detailed report also contains the judge's reason so the reviewer can identify which claims or wording caused the lower score.

## Quality gate
Each metric has its own PASS/REVIEW/FAIL thresholds, configured in
`project_config.json`'s `evaluation.metric_thresholds`:

>= 0.80 → metric PASS, 0.60 to < 0.80 → REVIEW, < 0.60 → FAIL
(Completeness uses 0.70/0.50 instead of 0.80/0.60.)

A testcase passes when its weighted score (the same `metric_weights` from
`project_config.json`) is >= 0.80, is REVIEW between 0.60 and < 0.80, and
FAIL below 0.60 - AND Hallucination, Faithfulness and Correctness must each
individually clear their own REVIEW threshold, regardless of the weighted
score. The judge evaluates every generated response.

A metric that errors (judge timeout, rate limit, connection failure, or any
other technical exception) never produces a weighted score by averaging over
whatever else completed - the whole testcase is reported as `TECHNICAL ERROR`,
distinct from `QUALITY FAIL`/`QUALITY REVIEW` (the judge completed and found a
real quality issue) and `PASS`. Conflating "the judge couldn't finish" with
"the response was bad" would misattribute an infrastructure problem to the
generator.

The generator recommendation considers the quality gate first, followed by
weighted semantic score and testcase pass rate - but only when the sample is
large enough: below `execution.minimum_testcases_for_ranking` (default 10)
test cases, or if any generator hit a technical error, the report explicitly
declines to name a "preferred generator" rather than claim a statistically
meaningless winner from a smoke-sized sample.

## Reports

Generator response workbook: `outputs/generator_responses/YYYY-MM-DD/generator_responses_<run_id>.xlsx`
Each generator has its own worksheet containing the generated responses and execution information.

Evaluation report: `reports/YYYY-MM-DD/generator_comparison_<run_id>.xlsx`

Both files for the same run share the same `run_id`, so they can be
correlated even if the run took several minutes between generation and the
final report.

The report contains:

- **Run Summary** — run ID, sample size, pass/error counts, phase timing (with percentages), and whether the run met the configured runtime target.
- **Executive Summary** — final recommendation and evaluation rules.
- **Generator Comparison** — generator-level percentage scores, PASS/REVIEW/FAIL verdicts, and technical-error/quality-fail/quality-review counts. A metric's per-generator average is blank, not silently averaged over partial results, if any evaluation of it was technically incomplete.
- **Testcase Comparison** — testcase-level metric scores and status (PASS/QUALITY REVIEW/QUALITY FAIL/TECHNICAL ERROR).
- **Detailed Metric Reasons** — score, threshold result, gap to ideal, judge reason, technical status, and duration.
- **Failures** — metrics/testcases that require attention, with failure type (technical vs. quality) separated out.
- **LLM Call Profile** — every individual internal judge LLM call (e.g. Faithfulness's truths/claims/verdicts/reason), the DeepEval step, and its duration - the same data streamed live to the console during the run, persisted for later analysis.
- **Configuration** — models (with exact Ollama digests, not just tags), provider, package/interpreter versions, dataset/prompt hashes, concurrency, and thresholds, for reproducing exactly what produced this report.

# Run
Activate the project environment: .\.venv\Scripts\Activate.ps1
Then, from the project root: python -m execution.main

(`main.py` lives in `execution/`, not the project root - it's a package
module now, run with `-m` like any other, not a standalone script.)

NOTE: The Ollama server must be reachable from the machine running the POC and must contain the configured generator and judge models.

## Project structure
multigenerators-e2e_evals/
│
├── testdata/                      # Data layer
│   ├── __init__.py
│   ├── loader.py
│   └── hallucination_benchmark.xlsx
│
├── execution/                     # Execution layer
│   ├── __init__.py
│   ├── main.py                    # run entry point: python -m execution.main
│   └── generators/
│       ├── __init__.py
│       └── ollama_client.py
│
├── evaluation/                    # Evaluation layer
│   ├── __init__.py
│   ├── metrics.py
│   ├── runner.py
│   └── scorer.py
│
├── observability/                 # Observability & governance layer
│   ├── __init__.py
│   ├── discovery.py
│   ├── provenance.py
│   ├── console.py
│   ├── analysis.py
│   └── reporting/
│       ├── __init__.py
│       └── excel.py
│
├── settings/                      # Settings
│   ├── __init__.py
│   ├── config.py
│   └── project_config.json
│
├── tests/
│   ├── conftest.py
│   ├── test_scorer.py
│   ├── test_evaluation.py
│   ├── test_excel_report.py
│   ├── test_metrics.py
│   └── test_analysis.py
│
├── README.md
├── requirements.txt
└── .gitignore

## Validation status

Validated with live end-to-end runs against the configured Ollama server
(generators + judge), at 1, 3, 5, and 10 test cases, in addition to the
full unit test suite (`pytest tests/`) - including after the layered
restructure above, run live end-to-end through the project's actual
`.venv` (242.82s for 3 test cases, in line with every prior measurement:
no regression from the restructure). Representative timings (current
model roster, `qwen3-coder:30b` judge, serialized single-request Ollama
server):

| test_case_limit | Total run time |
|---|---|
| 1  | ~90s |
| 3  | ~4-4.5 min |
| 5  | ~7-7.5 min |
| 10 | ~14 min |

The evaluation phase (the judge scoring each response) dominates total
time and scales close to linearly with test case count, because the
Ollama server processes one request at a time - confirmed directly by
benchmarking concurrent requests against it. `test_case_limit=3` is the
current default as the largest sample that reliably completes in under
5 minutes on this server; run with a higher limit for fuller coverage
when a longer run is acceptable. See `judge_provider` above for an
alternative that removes this ceiling by using a hosted judge model
instead of the local Ollama server.