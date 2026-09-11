# DeepEval Lifecycle Component Evaluation - One Testcase POC

This POC demonstrates a human-gated prompt improvement lifecycle using **one original testcase** and **one targeted synthetic testcase**.

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

## One-testcase configuration

`config.py` contains:

```text
TEST_CASE_LIMIT = 1
SYNTHETIC_TEST_CASE_COUNT = 1
```

These values can be overridden through `.env`.

The original dataset is limited to exactly one testcase before any V1/V2 processing. The synthetic dataset is validated to contain exactly one testcase.

## Run

```powershell
python main.py
```

## Resume

If the process stops unexpectedly, use the run folder printed by the application:

```powershell
python main.py --resume "reports\YYYY-MM-DD\run_HHMMSS"
```

The lifecycle stores `run_state.json` after each completed stage, so completed DeepEval stages are not rerun.

## Important

Close the Excel workbook before answering `Y` at a human gate. The program needs to write the approval/checkpoint to the workbook.
