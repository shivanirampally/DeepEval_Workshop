# DeepEval Lifecycle Component Evaluation POC

This POC demonstrates a human-gated prompt improvement lifecycle: evaluate a baseline
prompt (V1) against a curated 10-testcase hallucination benchmark, generate a targeted
synthetic testcase from whatever failed, optimize the prompt (V2), and compare. One
testcase (`T010`) is an intentional "V1 trap" -- designed so a weak prompt fails it
even when everything else passes, guaranteeing there's always something to improve.

## Project layout

```
config.py                     every tunable setting, in one file -- edit this directly, no .env

testdata/                     everything about the data going into the lifecycle
    dataset/                  the source benchmark workbook
    prompts.py                BASE_PROMPT and the optimizer's meta-prompt
    synthetic_data.py         targeted synthetic testcase generation + validation

execution/                    everything that drives an actual run
    main.py                   the lifecycle entrypoint (stages, resume, human gates)
    ollama_client.py          thin HTTP client for Ollama text generation
    prompt_optimizer.py       asks the optimizer model for a V2 prompt candidate

evaluation/                   everything that scores a response
    evaluator.py               DeepEval metrics (hallucination, faithfulness, ...)
    failure_analysis.py        turns failing metric rows into a PASS/REVIEW/FAIL verdict + fix suggestions

governance_observability/     everything about audit trail, approvals, and logging
    excel_reporting.py         the workbook: evaluations, approvals, comparisons, audit log
    logging_config.py          console/file logging setup + the shared LLM-call timing helper

reports/                       generated per run: workbook, run.log, run_state.json, prompt snapshots
tests/                         pytest suite
```

`reports/<date>/run_<time>/` holds everything a single run produces, including its own `prompts/`
subfolder with the exact prompt text used for V1 and the V2 candidate -- there's no separate
top-level prompts folder to cross-reference.

## Performance / scalability controls

- `GENERATOR_MODEL`/`JUDGE_MODEL`/`OPTIMIZER_MODEL`/`SYNTHETIC_MODEL` default to the fastest
  models benchmarked on this project's Ollama server.
- `GENERATOR_CONCURRENCY` bounds concurrent generation requests.
- `JUDGE_CONCURRENCY` bounds ONE pool shared across every testcase's metrics combined (not
  6 threads per testcase), so it stays a fixed resource cap as `TEST_CASE_LIMIT` grows.
  Default is 3, not 6 -- a controlled sweep (1/3/6/12, 2026-09-17) measured essentially flat
  wall-clock time from 3 upward (87.5s/87.8s/92.2s), meaning this server has one real
  throughput ceiling regardless of concurrency; 3 reaches it with half the connections.
  Re-run the sweep if the backend changes.
- Correctness and Completeness (GEval) use static `evaluation_steps`, skipping the judge
  call DeepEval would otherwise make every time to turn `criteria` into steps.
- `SYNTHETIC_MODE=direct` (default) asks for synthetic testcases as one JSON request;
  `SYNTHETIC_MODE=deepeval` falls back to the original DeepEval Synthesizer. Controlled
  benchmark (2026-09-17, same source/model/SYNTHETIC_TEST_CASE_COUNT=1, one run each):
  direct = 6.40s vs deepeval = 24.71s (~3.9x faster) -- the Synthesizer's own
  context/golden/evolution calls are the difference, not the model or server.
- Excel timing/audit/approval rows are appended in place instead of a full workbook
  read-rewrite-reformat per row; a one-time formatting pass runs at the end of each run.
- `DEEPEVAL_TELEMETRY_OPT_OUT=YES` skips DeepEval's ~10s shutdown telemetry flush.
- These changes optimize application-side work. Actual backend parallelism is an
  infrastructure/server capability -- the sweep above shows this Ollama endpoint has a
  fixed throughput ceiling (not full serialization: per-call latency scales with
  concurrency, but wall-clock time doesn't improve past concurrency=3), so client-side
  concurrency past that point gives diminishing returns rather than added throughput.

## Lifecycle

1. V1 baseline evaluation
2. Failure analysis
3. Human approval of failure analysis
4. Targeted synthetic testcase generation
5. Human review/edit/approval of synthetic testcase
6. Synthetic V1 evaluation
7. Prompt optimization
8. Human approval of V2 prompt
9. V2 evaluation on the same original testcase
10. Synthetic V2 evaluation on the same synthetic testcase
11. V1 vs V2 comparison
12. STOP - no V3 and no automatic prompt promotion

## Configuration

Every tunable lives directly in `config.py` as a plain constant -- there is no `.env` file, so
there's exactly one place to look and one place to edit (a committed `.env` would otherwise
silently override these documented values for anyone following this README). The settings that
matter most for this POC:

```text
TEST_CASE_LIMIT = 10
SYNTHETIC_TEST_CASE_COUNT = 1
GENERATOR_CONCURRENCY = 3
JUDGE_CONCURRENCY = 3
SYNTHETIC_MODE = direct
```

`TEST_CASE_LIMIT` caps how many rows of `testdata/dataset/hallucination_benchmark.xlsx` are
used (defaults to all 10). The synthetic dataset generated from V1's failures is validated to
contain exactly `SYNTHETIC_TEST_CASE_COUNT` testcase(s).

If V1 passes every metric on every testcase (rare, given T010), the lifecycle stops cleanly
right after failure analysis instead of erroring out on synthetic data generation -- there's
nothing to target for V2, and that's a legitimate outcome, not a failure.

## Governance fixes (2026-09-17)

- **A real judge/server error is not a prompt-quality failure.** If any V1 metric call
  technically errors out (network blip, malformed response), the lifecycle stops right after
  failure analysis instead of feeding that noise into synthetic data generation / the
  optimizer -- see `11_Audit_Log` for the technical error count, then re-run V1.
- **`--resume` cannot bypass an explicit human rejection.** Answering `N` at any approval
  gate now permanently blocks `--resume` on that run folder (a real gap in the original
  design where resuming after a rejection would silently continue the pipeline anyway).
  Start a new run once the concern is addressed.
- **Synthetic data must carry a recorded human approval** before Synthetic V1/V2 evaluation
  or the optimizer will use it -- enforced by `validate_human_approved_synthetic_dataset`,
  separate from the basic schema check used while the sheet is still pending review.
- **Transient Ollama failures (429/5xx/timeout/connection) retry** with exponential backoff
  (`OLLAMA_MAX_RETRIES`, `OLLAMA_RETRY_BACKOFF_SECONDS`) instead of killing a 10+ minute run
  over one blip.
- Dataset and config values are validated up front (blank required fields, threshold
  ordering, temperature range) so bad input fails fast with a clear message.
- `10_V1_vs_V2` adds an `Overall Regression` / `V1 vs V2 Pass-Rate Change` summary on top of
  the existing per-metric `Regression Flag` column.

## Run

```powershell
python -m execution.main
```

Run it as a module (`-m`) from the project root, not `python execution/main.py` directly --
`main.py` imports `config`, `testdata`, `evaluation`, and `governance_observability` as top-level
packages, which only resolve correctly when the project root is on `sys.path`.

Override the testcase count for a single run (e.g. a demo that needs to fit in 5 minutes)
without editing `config.py`:

```powershell
python -m execution.main --test-cases 3
```

## Resume

If the process stops unexpectedly, use the run folder printed by the application:

```powershell
python -m execution.main --resume "reports\YYYY-MM-DD\run_HHMMSS"
```

The lifecycle stores `run_state.json` after each completed stage, so completed DeepEval stages
are not rerun. `--resume` restores the same testcase count the run was started with, and refuses
to continue past a run a human explicitly stopped (see Governance fixes above).

## Important

Close the Excel workbook before answering `Y` at a human gate. The program needs to write the approval/checkpoint to the workbook.
