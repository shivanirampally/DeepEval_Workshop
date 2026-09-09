# Project 3 - Project 1 Trajectory Validation

This project validates the **multi-generator decision trajectory used by Project 1**.

## What it validates

Project 1 runs the same three generators:

1. `llama3:instruct`
2. `qwen2.5-coder:14b`
3. `sqlcoder:15b`

Project 3 traces those generator calls, records the selection step, and compares the selected generator with the preferred generator reported by Project 1.

```text
Project 1 benchmark
       |
       +--> Generator 1
       +--> Generator 2
       +--> Generator 3
       |
       +--> Selection
       |
       +--> Final response
```

## Trajectory metrics

- **Task Completion** - did the workflow complete the requested task?
- **Step Efficiency** - did it avoid unnecessary or redundant steps?
- **Plan Adherence** - did it follow the configured comparison and selection path?

These are DeepEval trajectory metrics. They require a traced agent and are evaluated through `evals_iterator()` against the complete trace.

## Project 1 connection

Project 3 reads the latest Project 1 `generator_comparison_*.xlsx` report and extracts its `Preferred generator` verdict. The trajectory report then shows whether the traced workflow selected the same generator.

This does **not** replace Project 1's quality evaluation. Project 1 evaluates response quality; Project 3 validates the execution path that leads to the model-selection outcome.

## Run

Run Project 1 first so its report exists. Then:

```powershell
.\setup.ps1
.\.venv\Scripts\Activate.ps1
python -m pytest -q
python main.py
```

The report is written under:

```text
reports/YYYY-MM-DD/trajectory_evaluation_*.xlsx
```

Use:

```powershell
deepeval inspect
```

after the run to inspect the captured trace and metric reasons.
