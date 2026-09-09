# Project 2 - Prompt Improvement POC

## Objective

Demonstrate how DeepEval findings can drive prompt improvement:

**Same Project 1 benchmark → V1 baseline → clear report → failure analysis → targeted synthetic test data → prompt improvement**

This POC intentionally runs **V1 only**. It does not run V2/V3.

## Input

`dataset/hallucination_benchmark.xlsx` is the **same Project 1 benchmark** and is kept unchanged.

The baseline evaluation uses this file directly.

## V1 baseline

The baseline prompt is intentionally simple:

```text
Answer the question using the provided source.
```

Generator:

`qwen2.5:14b`

Independent judge:

`gpt-oss:20b`

Temperature:

`0`

DeepEval metrics:

- Hallucination
- Faithfulness
- Correctness
- Completeness
- Answer Relevancy
- Bias

Metric thresholds:

- PASS: >= 90%
- REVIEW: 70% to <90%
- FAIL: <70%

A testcase passes only when all its metrics pass.

## Failure analysis

The report connects:

**Test Case → Metric → Score → Verdict → Reason → Recommended Action**

For each testcase, all six metrics are submitted together and run in parallel. The analysis groups failed/reviewed metrics and turns them into practical prompt recommendations.

## Synthetic test data

After reviewing the V1 failures, `dataset/create_synthetic_data.py` creates targeted synthetic cases designed around common source-grounding gaps.

The original Project 1 benchmark is never modified.

Run:

```powershell
python dataset/create_synthetic_data.py
```

This creates:

`dataset/synthetic_test_data.xlsx`

In a real iteration, the synthetic scenarios should be aligned to the actual failure patterns observed in the V1 report.

## Prompt improvement

The improved prompt is implemented in `prompts.py`.

The change adds:

- explicit source grounding
- no invention or unsupported inference
- missing-information handling
- partial-answer handling
- concise direct answers
- no outside knowledge

The POC documents the improvement but does **not** claim a score improvement because the improved prompt is not rerun.

## Run

Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Run V1:

```powershell
python main.py
```

Run tests:

```powershell
pytest -q
```

The Ollama server must be reachable from the machine running the project.

## Report

Reports are saved under:

`reports/YYYY-MM-DD/v1_baseline_evaluation_HHMMSS.xlsx`

Sheets:

1. **Executive Summary** - what was tested and the baseline result
2. **Testcase Results** - testcase-level result
3. **Metric Summary** - metric-level performance
4. **Detailed Results** - score, verdict, reason, response and status
5. **Failure Analysis** - failed/reviewed cases and recommended actions
6. **Recommendations** - grouped recommendations
7. **Prompt Improvement** - baseline prompt, improved prompt and rationale
8. **Baseline Input Data** - exact input used for V1

The report is designed for human reading rather than DeepEval internals.

## Project structure

```text
project2/
├── dataset/
│   ├── hallucination_benchmark.xlsx
│   └── create_synthetic_data.py
├── analysis/
│   ├── __init__.py
│   └── failure_analysis.py
├── deepeval_framework/
│   ├── __init__.py
│   └── evaluator.py
├── reporting/
│   ├── __init__.py
│   └── excel.py
├── tests/
│   └── test_analysis.py
├── config.py
├── prompts.py
├── main.py
├── README.md
├── requirements.txt
└── .gitignore
```
