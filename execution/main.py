from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from time import perf_counter
import argparse
import json
import logging
from pathlib import Path
import pandas as pd

from config import (
    DATASET_PATH,
    GENERATOR_CONCURRENCY,
    GENERATOR_MODEL,
    JUDGE_CONCURRENCY,
    JUDGE_MODEL,
    OLLAMA_BASE_URL,
    OPTIMIZER_MODEL,
    REPORT_ROOT,
    SYNTHETIC_MODE,
    SYNTHETIC_TEST_CASE_COUNT,
    TEMPERATURE,
    TEST_CASE_LIMIT,
)
from evaluation.evaluator import evaluate_responses
from evaluation.failure_analysis import analyze
from execution.ollama_client import ollama_generate
from execution.prompt_optimizer import optimize_prompt
from governance_observability.excel_reporting import (
    append_audit,
    append_llm_call_timing,
    append_llm_call_timings,
    format_workbook,
    read_sheet,
    save_approval,
    save_comparison,
    save_evaluation,
    save_failure_analysis,
    save_prompt_version,
    save_summary,
)
from governance_observability.logging_config import configure_logging
from testdata.prompts import BASE_PROMPT
from testdata.synthetic_data import (
    generate_synthetic_dataset,
    validate_human_approved_synthetic_dataset,
    validate_synthetic_dataset,
)

STAGES = [
    "V1",
    "FAILURE_ANALYSIS",
    "HUMAN_APPROVAL_FAILURE_ANALYSIS",
    "SYNTHETIC_DATA",
    "HUMAN_APPROVAL_SYNTHETIC_DATA",
    "SYNTHETIC_V1",
    "OPTIMIZER",
    "HUMAN_APPROVAL_V2_PROMPT",
    "V2",
    "SYNTHETIC_V2",
    "COMPARISON",
    "COMPLETE",
]


def state_path(run_dir):
    return run_dir / "run_state.json"


def save_state(run_dir, run_id, last_completed_stage, next_stage, status="RUNNING", error="", test_case_limit=None):
    state = {
        "run_id": run_id,
        "last_completed_stage": last_completed_stage,
        "next_stage": next_stage,
        "status": status,
    }
    if error:
        state["error"] = error
    if test_case_limit is not None:
        # So a later --resume uses the SAME testcase count this run started
        # with, not whatever config.TEST_CASE_LIMIT happens to be at resume
        # time -- otherwise a run started with --test-cases 3 could resume
        # against 10 testcases and silently misalign V1 vs V2 comparisons.
        state["test_case_limit"] = test_case_limit
    state_path(run_dir).write_text(json.dumps(state, indent=2), encoding="utf-8")


def load_state(run_dir):
    path = state_path(run_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def log_stage(workbook, stage, action, status, details=""):
    logging.info("[%s] %s - %s", stage, action, status)
    append_audit(
        workbook,
        datetime.now().isoformat(timespec="seconds"),
        stage,
        action,
        status,
        details,
    )


def human_gate(workbook, stage, message):
    # Blocks here on purpose -- this is the whole point of a human-gated
    # lifecycle. No timeout, no auto-approve.
    print()
    print("=" * 68)
    print(f"HUMAN ACTION REQUIRED: {stage}")
    print("=" * 68)
    print(message)
    print()
    print("IMPORTANT: Close the Excel workbook before entering Y. The lifecycle must write to it after approval.")

    while True:
        answer = input("Enter Y to approve and continue, N to stop: ").strip().lower()
        if answer in {"y", "yes"}:
            try:
                save_approval(workbook, stage, "APPROVED", "CONTINUE", "Human approved the stage.")
            except PermissionError:
                print("Workbook is still open/locked. Close Excel and enter Y again.")
                continue
            logging.info("[HUMAN] %s - APPROVED", stage)
            return True
        if answer in {"n", "no"}:
            try:
                save_approval(workbook, stage, "REJECTED", "STOP", "Human rejected the stage.")
            except PermissionError:
                logging.warning("[HUMAN] %s rejected, but workbook is locked; approval record was not written.", stage)
            logging.info("[HUMAN] %s - REJECTED. Lifecycle stopped.", stage)
            return False
        print("Please enter Y or N.")


def load_data(path, limit=None):
    """limit overrides config.TEST_CASE_LIMIT for a single run -- lets --test-cases
    pick a smaller demo-sized slice without editing config.py."""
    if limit is None:
        limit = TEST_CASE_LIMIT
    dataset = pd.read_excel(path)
    required = {"Test_ID", "Source", "Question", "Golden_Answer"}
    missing = required - set(dataset.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if dataset.empty:
        raise ValueError("Dataset is empty.")
    if dataset["Test_ID"].fillna("").astype(str).str.strip().eq("").any():
        raise ValueError("Dataset contains blank Test_ID values.")
    if dataset["Test_ID"].duplicated().any():
        raise ValueError("Dataset contains duplicate Test_ID values.")
    for column in ["Source", "Question", "Golden_Answer"]:
        if dataset[column].fillna("").astype(str).str.strip().eq("").any():
            raise ValueError(f"Dataset contains blank {column} values.")

    dataset = dataset.head(limit).copy()
    if len(dataset) != limit:
        raise ValueError(
            f"Requested {limit} testcase(s), but the dataset contains only {len(dataset)}."
        )
    return dataset


def generate(row, prompt_template):
    prompt = prompt_template.format(
        source=str(row["Source"]),
        question=str(row["Question"]),
    )
    started = perf_counter()
    response = ollama_generate(GENERATOR_MODEL, prompt, temperature=TEMPERATURE)
    return response, round(perf_counter() - started, 3)


def run_generation_and_evaluation(dataset, prompt_template, label):
    logging.info("[%s] Generating %d responses...", label, len(dataset))
    generated = {}
    generation_latency = {}
    generation_errors = {}

    # Capped at GENERATOR_CONCURRENCY -- the dataset is tiny in this POC, but we
    # still don't want to hammer the Ollama box with one request per row, and a
    # server with a low throughput ceiling can dial this down in config.py.
    with ThreadPoolExecutor(max_workers=min(GENERATOR_CONCURRENCY, max(1, len(dataset)))) as pool:
        futures = {
            pool.submit(generate, row, prompt_template): str(row["Test_ID"])
            for _, row in dataset.iterrows()
        }
        for future in as_completed(futures):
            test_id = futures[future]
            try:
                generated[test_id], generation_latency[test_id] = future.result()
            except Exception as exc:
                generation_errors[test_id] = f"{type(exc).__name__}: {exc}"

    logging.info("[%s] Generation complete.", label)
    rows = []

    # One evaluation call for the whole dataset -- evaluate_responses runs every
    # testcase's metrics through a single bounded pool instead of us looping
    # testcase-by-testcase and spinning up a fresh pool each time.
    evaluable = [
        (row, generated[str(row["Test_ID"])])
        for _, row in dataset.iterrows()
        if str(row["Test_ID"]) not in generation_errors
    ]
    logging.info("[%s] Evaluating %d testcase(s)...", label, len(evaluable))
    evaluated = evaluate_responses(evaluable) if evaluable else {}

    for _, row in dataset.iterrows():
        test_id = str(row["Test_ID"])
        response = generated.get(test_id, "")

        if test_id in generation_errors:
            rows.append({
                "test_id": test_id,
                "metric": "generation",
                "score": None,
                "threshold": None,
                "passed": False,
                "status": "GENERATION_ERROR",
                "reason": "",
                "error": generation_errors[test_id],
                "response": "",
                "source": str(row["Source"]),
                "question": str(row["Question"]),
                "expected_output": str(row["Golden_Answer"]),
                "evaluation_model": JUDGE_MODEL,
                "evaluation_method": "Not evaluated",
                "generation_latency_seconds": None,
                "latency_seconds": None,
            })
            continue

        results = evaluated[test_id]
        testcase_passed = all(bool(result.get("passed")) for result in results.values())

        for metric, result in results.items():
            rows.append({
                "test_id": test_id,
                "metric": metric,
                "score": result["score"],
                "threshold": result["threshold"],
                "passed": result["passed"],
                "testcase_passed": testcase_passed,
                "status": result["status"],
                "reason": result["reason"],
                "error": result["error"],
                "response": response,
                "source": str(row["Source"]),
                "question": str(row["Question"]),
                "expected_output": str(row["Golden_Answer"]),
                "evaluation_model": result["evaluation_model"],
                "evaluation_method": result["evaluation_method"],
                "generation_latency_seconds": generation_latency.get(test_id),
                "latency_seconds": result["latency_seconds"],
            })

    result_frame = pd.DataFrame(rows)
    completed = int((result_frame.get("status", pd.Series(dtype=str)) == "COMPLETED").sum()) if not result_frame.empty else 0
    errors = int((result_frame.get("status", pd.Series(dtype=str)) != "COMPLETED").sum()) if not result_frame.empty else 0
    logging.info("[%s] DeepEval evaluation complete. Metric results=%d, non-completed=%d.", label, completed, errors)
    return result_frame


def record_call_timings(workbook, stage_label, details):
    """Turn one stage's per-row latency columns into rows on the 12_LLM_Call_Timings
    sheet, so the whole run's LLM timing can be read as one table instead of grepping run.log.

    Collected into one list and written with a single append_llm_call_timings
    call -- one workbook open/save for the whole stage instead of one per row.
    """
    if details.empty:
        return
    timing_rows = []
    logged_generation_for = set()
    for _, row in details.iterrows():
        test_id = row.get("test_id")
        gen_seconds = row.get("generation_latency_seconds")
        if test_id not in logged_generation_for and pd.notna(gen_seconds):
            timing_rows.append({
                "Stage": stage_label,
                "Call": f"generate:{test_id}",
                "Model": GENERATOR_MODEL,
                "Time Taken (s)": gen_seconds,
            })
            logged_generation_for.add(test_id)
        metric_seconds = row.get("latency_seconds")
        if pd.notna(metric_seconds):
            timing_rows.append({
                "Stage": stage_label,
                "Call": f"metric={row.get('metric')}:{test_id}",
                "Model": row.get("evaluation_model", JUDGE_MODEL),
                "Time Taken (s)": metric_seconds,
            })
    append_llm_call_timings(workbook, timing_rows)


def update_summary(workbook, run_id, testcase_count, extra=None):
    rows = [
        {"Item": "Run ID", "Value": run_id},
        {"Item": "Original Testcase Count", "Value": testcase_count},
        {"Item": "Synthetic Testcase Count", "Value": SYNTHETIC_TEST_CASE_COUNT},
        {"Item": "Generator", "Value": GENERATOR_MODEL},
        {"Item": "Judge", "Value": JUDGE_MODEL},
        {"Item": "Ollama Endpoint", "Value": OLLAMA_BASE_URL},
        {"Item": "Temperature", "Value": TEMPERATURE},
        {"Item": "Generator Concurrency", "Value": GENERATOR_CONCURRENCY},
        {"Item": "Judge Concurrency", "Value": JUDGE_CONCURRENCY},
        {"Item": "Synthetic Mode", "Value": SYNTHETIC_MODE},
        {"Item": "Lifecycle", "Value": f"Human-gated; {testcase_count} testcase(s); stops after V2 comparison"},
    ]
    if extra:
        rows.extend(extra)
    save_summary(workbook, rows)


def finalize_lifecycle(workbook, run_dir, run_id, checkpoint, dataset_count, last_stage,
                        summary_extra, audit_details, log_path, title, result_line, next_step_line="STOP"):
    """Shared "we're done" sequence: update the summary sheet, log the final
    audit entry, run one full workbook formatting pass, checkpoint as
    COMPLETE, and print the closing banner. Used both when V1 passes cleanly
    (nothing to improve) and after a full V1 vs V2 comparison, so those two
    "lifecycle finished" paths can't quietly drift out of sync with each other.
    """
    update_summary(workbook, run_id, dataset_count, summary_extra)
    append_audit(workbook, datetime.now().isoformat(timespec="seconds"), "RUN",
                 "Lifecycle stopped", "COMPLETE", audit_details)
    # One final full formatting pass -- the lightweight appends used
    # throughout the run skip this for speed, so catch it up here on
    # the finished workbook a human is actually going to open.
    format_workbook(workbook)
    checkpoint(last_stage, "COMPLETE", "COMPLETE")
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)
    print(f"Master Excel : {workbook}")
    print(f"Run Log      : {log_path}")
    print(f"Result       : {result_line}")
    print(f"Next step    : {next_step_line}")
    print("=" * 68)


def stop_for_technical_errors(workbook, run_dir, run_id, checkpoint, dataset_count,
                               stage_label, technical_errors, retry_stage):
    """Shared "don't build anything on top of broken judge/generation data"
    stop. A network blip or malformed response is a technical failure, not a
    verdict on the prompt -- feeding it into failure analysis, synthetic data,
    the optimizer, or the final comparison would treat noise as a real
    quality signal. retry_stage is the exact stage --resume should redo;
    everything before it stays cached since it's unaffected.
    """
    log_stage(workbook, stage_label, "Technical evaluation errors detected", "ERROR",
              f"{len(technical_errors)} metric result(s) did not complete.")
    update_summary(workbook, run_id, dataset_count, [
        {"Item": f"{stage_label} Status", "Value": "TECHNICAL ERROR - blocked"},
        {"Item": "Technical Error Count", "Value": len(technical_errors)},
        {"Item": "Lifecycle", "Value": f"STOPPED - resolve technical errors and re-run {retry_stage}"},
    ])
    append_audit(workbook, datetime.now().isoformat(timespec="seconds"), "RUN",
                 "Lifecycle stopped", "TECHNICAL_ERROR",
                 f"{stage_label}: {len(technical_errors)} judge/generation call(s) did not complete.")
    format_workbook(workbook)
    checkpoint("UNKNOWN", retry_stage, "ERROR",
               f"{len(technical_errors)} technical metric error(s) in {stage_label}.")
    print("\n" + "=" * 68)
    print(f"LIFECYCLE STOPPED -- TECHNICAL ERROR ({stage_label})")
    print("=" * 68)
    print(f"Master Excel : {workbook}")
    print(f"{len(technical_errors)} judge/generation call(s) did not complete -- see 11_Audit_Log.")
    print("This is not a prompt-quality finding; re-run once the underlying issue is fixed:")
    print(f"  python -m execution.main --resume \"{run_dir}\"")
    print("=" * 68)


def read_approved_v2_prompt(workbook):
    prompt_versions = read_sheet(workbook, "07_Prompt_Versions")
    candidates = prompt_versions[
        prompt_versions["Version"].astype(str).str.upper().eq("V2 CANDIDATE")
    ]
    if candidates.empty:
        raise ValueError("No V2 Candidate prompt found in 07_Prompt_Versions.")
    prompt = str(candidates.iloc[-1]["Prompt"]).strip()
    if "{source}" not in prompt or "{question}" not in prompt:
        raise ValueError("Approved V2 prompt must contain {source} and {question} placeholders.")
    return prompt


def find_workbook(run_dir):
    # File names embed a sortable timestamp, so the last one alphabetically is the latest.
    workbooks = sorted(run_dir.glob("AI_validation_run_*.xlsx"))
    if not workbooks:
        raise FileNotFoundError(f"No lifecycle workbook found in {run_dir}")
    return workbooks[-1]


def infer_test_case_limit_from_workbook(workbook):
    """Best-effort recovery of the original testcase count for a run whose
    checkpoint predates test_case_limit being persisted, or has none at all.
    Falls back to None (meaning: use config.TEST_CASE_LIMIT) if V1 never
    completed, since there's nothing in the workbook to count yet."""
    try:
        v1_details = read_sheet(workbook, "02_V1_Evaluation")
    except Exception:
        return None
    if v1_details.empty or "test_id" not in v1_details.columns:
        return None
    return int(v1_details["test_id"].nunique())


def infer_resume_stage(workbook):
    """Create a checkpoint for a legacy run that predates run_state.json."""
    with pd.ExcelFile(workbook) as xls:
        sheets = set(xls.sheet_names)

    def approved(stage):
        if "04_Human_Approval" not in sheets:
            return False
        approvals = read_sheet(workbook, "04_Human_Approval")
        if approvals.empty:
            return False
        rows = approvals[approvals["Stage"].astype(str).str.upper() == stage.upper()]
        return not rows.empty and str(rows.iloc[-1]["Status"]).upper() == "APPROVED"

    if "10_V1_vs_V2" in sheets:
        return "COMPLETE"
    if "09_Synthetic_V2" in sheets:
        return "COMPARISON"
    if "08_V2_Evaluation" in sheets:
        return "SYNTHETIC_V2"
    if approved("V2 Prompt"):
        return "V2"
    if "07_Prompt_Versions" in sheets:
        return "HUMAN_APPROVAL_V2_PROMPT"
    if "06_Synthetic_V1" in sheets:
        return "OPTIMIZER"
    if approved("Synthetic Data"):
        return "SYNTHETIC_V1"
    if "05_Synthetic_Data" in sheets:
        return "HUMAN_APPROVAL_SYNTHETIC_DATA"
    if approved("Failure Analysis"):
        return "SYNTHETIC_DATA"
    if "03_Failure_Analysis" in sheets:
        return "HUMAN_APPROVAL_FAILURE_ANALYSIS"
    if "02_V1_Evaluation" in sheets:
        return "FAILURE_ANALYSIS"
    return "V1"


def should_run(next_stage, stage):
    # Stages before the resume point are re-read from the workbook instead
    # of recomputed, so DeepEval calls that already ran don't run twice.
    if next_stage in {"STOP", "COMPLETE"}:
        return False
    if next_stage not in STAGES:
        raise ValueError(f"Unknown resume stage: {next_stage}")
    return STAGES.index(stage) >= STAGES.index(next_stage)


def main_new(test_case_limit=None):
    start = datetime.now()
    date_part = start.strftime("%Y-%m-%d")
    time_part = start.strftime("%H%M%S")
    run_id = f"{date_part}_{time_part}"

    run_dir = REPORT_ROOT / date_part / f"run_{time_part}"
    run_dir.mkdir(parents=True, exist_ok=True)
    workbook = run_dir / f"AI_validation_run_{start.strftime('%Y%m%d_%H%M%S')}.xlsx"
    log_path = run_dir / "run.log"
    configure_logging(log_path)
    return run_lifecycle(run_id, run_dir, workbook, log_path, "V1", test_case_limit=test_case_limit)


def main_resume(run_dir):
    run_dir = Path(run_dir).resolve()
    workbook = find_workbook(run_dir)
    log_path = run_dir / "run.log"
    configure_logging(log_path)

    state = load_state(run_dir)
    if state and str(state.get("status", "")).upper() == "STOPPED":
        # A human answered N at an approval gate. Without this check, --resume
        # would happily pick up next_stage (the stage right after the gate)
        # and carry on -- silently overriding an explicit human rejection,
        # which defeats the entire point of a human-gated lifecycle.
        raise RuntimeError(
            "This run was explicitly stopped by a human reviewer (answered N at a gate). "
            "--resume cannot bypass that rejection -- address the concern and start a new run."
        )
    if state:
        next_stage = state.get("next_stage", "V1")
        run_id = state.get("run_id", run_dir.name.replace("run_", ""))
        # Restore the SAME testcase count this run was started with (e.g. via
        # --test-cases), not whatever config.TEST_CASE_LIMIT is by the time
        # someone resumes. Older checkpoints that predate this field fall back
        # to counting distinct test_id values already saved in the workbook.
        test_case_limit = state.get("test_case_limit") or infer_test_case_limit_from_workbook(workbook)
        # A terminal checkpoint is trusted only when the workbook's own audit
        # log confirms a real completion. There are two legitimate ways to
        # finish (full V1->V2->comparison, or V1 passing cleanly with nothing
        # to improve) and both funnel through finalize_lifecycle's identical
        # "Lifecycle stopped" / COMPLETE audit entry -- checking that directly
        # is more robust than hard-coding one specific set of sheets, which
        # incorrectly flagged the "V1 passed cleanly" completion shape as an
        # inconsistent checkpoint and re-ran parts of an already-finished run.
        if str(next_stage).upper() in {"STOP", "COMPLETE"}:
            with pd.ExcelFile(workbook) as xls:
                workbook_sheets = set(xls.sheet_names)
            actually_complete = False
            if "11_Audit_Log" in workbook_sheets:
                audit = read_sheet(workbook, "11_Audit_Log")
                actually_complete = (
                    (audit["Action"] == "Lifecycle stopped") & (audit["Status"] == "COMPLETE")
                ).any()
            if actually_complete:
                next_stage = "COMPLETE"
            else:
                recovered_stage = infer_resume_stage(workbook)
                logging.warning(
                    "Terminal checkpoint is inconsistent with workbook artifacts. "
                    "Recovering resume stage from workbook: %s -> %s",
                    state.get("next_stage"),
                    recovered_stage,
                )
                next_stage = recovered_stage
                save_state(run_dir, run_id, "RECOVERED", next_stage, "PAUSED", test_case_limit=test_case_limit)
    else:
        next_stage = infer_resume_stage(workbook)
        run_id = run_dir.parent.name + "_" + run_dir.name.replace("run_", "")
        test_case_limit = infer_test_case_limit_from_workbook(workbook)
        save_state(run_dir, run_id, "NONE", next_stage, "PAUSED", test_case_limit=test_case_limit)
        logging.info("No run_state.json found. Created legacy checkpoint; resume starts at %s.", next_stage)

    return run_lifecycle(run_id, run_dir, workbook, log_path, next_stage, test_case_limit=test_case_limit)


def run_lifecycle(run_id, run_dir, workbook, log_path, next_stage, test_case_limit=None):
    start = datetime.now()
    dataset = load_data(DATASET_PATH, test_case_limit)
    # The count this run actually loaded, not the raw (possibly None) parameter --
    # every checkpoint below records this so --resume can restore the exact same
    # testcase count later regardless of what config.TEST_CASE_LIMIT is by then.
    effective_test_case_limit = len(dataset)

    def checkpoint(last_completed_stage, next_stage_, status="RUNNING", error=""):
        save_state(run_dir, run_id, last_completed_stage, next_stage_, status, error, effective_test_case_limit)

    # Prompt snapshots live inside the run's own folder, next to its workbook
    # and log -- one place per run instead of a separate top-level tree.
    prompt_dir = run_dir / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 68)
    print("AI VALIDATION LIFECYCLE - HALLUCINATION EVALUATION POC")
    print("=" * 68)
    print(f"Run ID        : {run_id}")
    print(f"Workbook      : {workbook}")
    print(f"Original cases: {len(dataset)}")
    print(f"Synthetic cases: {SYNTHETIC_TEST_CASE_COUNT}")
    print(f"Generator     : {GENERATOR_MODEL}")
    print(f"Judge         : {JUDGE_MODEL}")
    print(f"Next stage    : {next_stage}")

    if next_stage == "COMPLETE":
        print("\nLifecycle already completed. Nothing to resume.")
        print(f"Result       : {len(dataset)} original testcase(s) -> {SYNTHETIC_TEST_CASE_COUNT} synthetic testcase(s) -> V1 -> V2 -> comparison")
        print("Next step    : STOP")
        return

    v1_details = None
    failures = None
    recommendations = None
    synthetic_df = None
    synthetic_v1 = None
    v2_details = None
    synthetic_v2 = None

    try:
        if should_run(next_stage, "V1"):
            log_stage(workbook, "1/8 V1", "Baseline evaluation started", "STARTED")
            time_part = datetime.now().strftime("%H%M%S")
            prompt_v1_path = prompt_dir / f"prompt_v1_{time_part}.txt"
            prompt_v1_path.write_text(BASE_PROMPT, encoding="utf-8")
            save_prompt_version(workbook, "V1", BASE_PROMPT, "BASELINE")
            v1_details = run_generation_and_evaluation(dataset, BASE_PROMPT, "V1")
            save_evaluation(workbook, "02_V1_Evaluation", v1_details)
            record_call_timings(workbook, "V1", v1_details)
            log_stage(workbook, "1/8 V1", "Baseline evaluation completed", "DONE")
            checkpoint("V1", "FAILURE_ANALYSIS")
        else:
            v1_details = read_sheet(workbook, "02_V1_Evaluation")

        if should_run(next_stage, "FAILURE_ANALYSIS"):
            log_stage(workbook, "2/8 Analysis", "Failure analysis started", "STARTED")
            failures, recommendations = analyze(v1_details)
            save_failure_analysis(workbook, failures, recommendations)
            log_stage(workbook, "2/8 Analysis", "Failure analysis completed", "DONE",
                      f"{len(failures)} failed/review results; {len(recommendations)} recurring patterns.")
            checkpoint("FAILURE_ANALYSIS", "HUMAN_APPROVAL_FAILURE_ANALYSIS")
        else:
            failures, recommendations = analyze(v1_details)

        technical_errors = failures.attrs.get("technical_errors", [])
        if technical_errors:
            # next_stage="V1" because the only trustworthy recovery is to redo
            # V1 from scratch; these rows can't be selectively retried.
            stop_for_technical_errors(workbook, run_dir, run_id, checkpoint, len(dataset),
                                       "2/8 Analysis", technical_errors, "V1")
            return

        if failures.empty:
            # V1 passed every metric cleanly -- there's nothing to target for
            # synthetic data generation or prompt optimization. That's a
            # success outcome (the baseline prompt already works), not an
            # error, so stop here cleanly instead of asking for an approval
            # gate over an empty failure list and crashing later when
            # generate_synthetic_dataset() finds no failure sources to use.
            log_stage(workbook, "2/8 Analysis", "V1 passed every metric; nothing to target for improvement", "COMPLETE")
            finalize_lifecycle(
                workbook, run_dir, run_id, checkpoint, len(dataset), "FAILURE_ANALYSIS",
                summary_extra=[
                    {"Item": "V1 Status", "Value": "COMPLETED - ALL METRICS PASSED"},
                    {"Item": "Lifecycle", "Value": "STOPPED AFTER V1 - no failures to target, baseline prompt already passes"},
                ],
                audit_details="V1 passed every metric; no synthetic/optimizer stages needed.",
                log_path=log_path,
                title="LIFECYCLE COMPLETE -- NO IMPROVEMENT NEEDED",
                result_line="V1 passed every metric on the baseline prompt.",
                next_step_line="STOP (no synthetic data, no V2, nothing to compare)",
            )
            return

        if should_run(next_stage, "HUMAN_APPROVAL_FAILURE_ANALYSIS"):
            if not human_gate(
                workbook,
                "Failure Analysis",
                f"Review '03_Failure_Analysis' in:\n{workbook}\n\nConfirm that the findings are valid and prompt improvement is appropriate.",
            ):
                format_workbook(workbook)  # the human is about to look at this workbook right now
                checkpoint("HUMAN_APPROVAL_FAILURE_ANALYSIS", "SYNTHETIC_DATA", "STOPPED")
                return
            checkpoint("HUMAN_APPROVAL_FAILURE_ANALYSIS", "SYNTHETIC_DATA")

        if should_run(next_stage, "SYNTHETIC_DATA"):
            log_stage(workbook, "3/8 Synthesis", "Generating targeted synthetic data", "STARTED")
            synthetic_working = run_dir / "synthetic_working.xlsx"
            _, synth_gen_elapsed = generate_synthetic_dataset(dataset, failures, synthetic_working)
            append_llm_call_timing(workbook, "SYNTHETIC_DATA", "synthetic_data_generation", OPTIMIZER_MODEL, synth_gen_elapsed)
            synthetic_df = pd.read_excel(synthetic_working)
            validate_synthetic_dataset(synthetic_df)
            save_evaluation(workbook, "05_Synthetic_Data", synthetic_df)
            log_stage(workbook, "3/8 Synthesis", "Targeted synthetic data generated", "DONE",
                      f"{len(synthetic_df)} cases created.")
            checkpoint("SYNTHETIC_DATA", "HUMAN_APPROVAL_SYNTHETIC_DATA")
        else:
            synthetic_df = read_sheet(workbook, "05_Synthetic_Data")
            if should_run(next_stage, "HUMAN_APPROVAL_SYNTHETIC_DATA"):
                # Resuming right at (or before) the approval gate -- the sheet
                # is legitimately still Human_Reviewed=PENDING at this point,
                # so only the base schema/blank-value checks apply here.
                validate_synthetic_dataset(synthetic_df)
            else:
                # Resuming from a stage strictly after the gate -- approval
                # must already be on record, or something's wrong.
                validate_human_approved_synthetic_dataset(synthetic_df)

        if should_run(next_stage, "HUMAN_APPROVAL_SYNTHETIC_DATA"):
            if not human_gate(
                workbook,
                "Synthetic Data",
                f"Review and edit '05_Synthetic_Data' in:\n{workbook}\n\nCorrect questions/golden answers if required. When you are satisfied, enter Y to approve the reviewed synthetic testcase(s). The system will record Human_Reviewed = APPROVED.",
            ):
                format_workbook(workbook)
                checkpoint("HUMAN_APPROVAL_SYNTHETIC_DATA", "SYNTHETIC_V1", "STOPPED")
                return
            synthetic_df = read_sheet(workbook, "05_Synthetic_Data")
            validate_synthetic_dataset(synthetic_df)

            # The console Y is the human approval gate. The workbook column is
            # the persisted audit status, so record the approval here rather
            # than requiring the reviewer to perform two separate approvals.
            synthetic_df["Human_Reviewed"] = "APPROVED"
            save_evaluation(workbook, "05_Synthetic_Data", synthetic_df)
            validate_human_approved_synthetic_dataset(synthetic_df)

            checkpoint("HUMAN_APPROVAL_SYNTHETIC_DATA", "SYNTHETIC_V1")

        if should_run(next_stage, "SYNTHETIC_V1"):
            log_stage(workbook, "4/8 Synthetic V1", "Validating V1 on synthetic data", "STARTED")
            synthetic_v1 = run_generation_and_evaluation(synthetic_df, BASE_PROMPT, "Synthetic V1")
            save_evaluation(workbook, "06_Synthetic_V1", synthetic_v1)
            record_call_timings(workbook, "SYNTHETIC_V1", synthetic_v1)
            log_stage(workbook, "4/8 Synthetic V1", "Synthetic V1 validation completed", "DONE")
            checkpoint("SYNTHETIC_V1", "OPTIMIZER")
        else:
            synthetic_v1 = read_sheet(workbook, "06_Synthetic_V1")

        synthetic_v1_failures, _ = analyze(synthetic_v1)
        synthetic_v1_technical_errors = synthetic_v1_failures.attrs.get("technical_errors", [])
        if synthetic_v1_technical_errors:
            stop_for_technical_errors(workbook, run_dir, run_id, checkpoint, len(dataset),
                                       "4/8 Synthetic V1", synthetic_v1_technical_errors, "SYNTHETIC_V1")
            return

        if should_run(next_stage, "OPTIMIZER"):
            log_stage(workbook, "5/8 Optimizer", "Generating candidate V2 prompt", "STARTED")
            time_part = datetime.now().strftime("%H%M%S")
            v2_prompt_path = prompt_dir / f"prompt_v2_candidate_{time_part}.txt"
            v2_prompt, optimizer_elapsed = optimize_prompt(BASE_PROMPT, failures, synthetic_df, v2_prompt_path)
            append_llm_call_timing(workbook, "OPTIMIZER", "optimize_prompt", OPTIMIZER_MODEL, optimizer_elapsed)
            save_prompt_version(workbook, "V2 Candidate", v2_prompt, "PENDING HUMAN APPROVAL")
            log_stage(workbook, "5/8 Optimizer", "Candidate V2 prompt generated", "DONE",
                      f"Saved as {v2_prompt_path.name}.")
            checkpoint("OPTIMIZER", "HUMAN_APPROVAL_V2_PROMPT")

        if should_run(next_stage, "HUMAN_APPROVAL_V2_PROMPT"):
            v2_prompt_path = sorted(prompt_dir.glob("prompt_v2_candidate_*.txt"))[-1]
            if not human_gate(
                workbook,
                "V2 Prompt",
                f"Review '07_Prompt_Versions' and:\n{v2_prompt_path}\n\nConfirm the changes address V1 failures without changing the original task. You may edit the V2 Candidate prompt in the workbook before approving.",
            ):
                format_workbook(workbook)
                checkpoint("HUMAN_APPROVAL_V2_PROMPT", "V2", "STOPPED")
                return
            v2_prompt = read_approved_v2_prompt(workbook)
            save_prompt_version(workbook, "V2 Candidate", v2_prompt, "APPROVED FOR EXECUTION")
            checkpoint("HUMAN_APPROVAL_V2_PROMPT", "V2")
        else:
            v2_prompt = read_approved_v2_prompt(workbook)

        if should_run(next_stage, "V2"):
            log_stage(workbook, "6/8 V2", "Running approved V2 prompt", "STARTED")
            v2_details = run_generation_and_evaluation(dataset, v2_prompt, "V2")
            save_evaluation(workbook, "08_V2_Evaluation", v2_details)
            record_call_timings(workbook, "V2", v2_details)
            log_stage(workbook, "6/8 V2", "V2 evaluation completed", "DONE")
            checkpoint("V2", "SYNTHETIC_V2")
        else:
            v2_details = read_sheet(workbook, "08_V2_Evaluation")

        v2_failures, _ = analyze(v2_details)
        v2_technical_errors = v2_failures.attrs.get("technical_errors", [])
        if v2_technical_errors:
            # Building the V1-vs-V2 comparison on top of incomplete V2 data
            # would produce a misleading "Declined"/regression verdict that
            # looks like a real prompt-quality problem.
            stop_for_technical_errors(workbook, run_dir, run_id, checkpoint, len(dataset),
                                       "6/8 V2", v2_technical_errors, "V2")
            return

        if should_run(next_stage, "SYNTHETIC_V2"):
            log_stage(workbook, "7/8 Synthetic V2", "Validating V2 on synthetic data", "STARTED")
            synthetic_v2 = run_generation_and_evaluation(synthetic_df, v2_prompt, "Synthetic V2")
            save_evaluation(workbook, "09_Synthetic_V2", synthetic_v2)
            record_call_timings(workbook, "SYNTHETIC_V2", synthetic_v2)
            log_stage(workbook, "7/8 Synthetic V2", "Synthetic V2 validation completed", "DONE")
            checkpoint("SYNTHETIC_V2", "COMPARISON")
        else:
            synthetic_v2 = read_sheet(workbook, "09_Synthetic_V2")

        synthetic_v2_failures, _ = analyze(synthetic_v2)
        synthetic_v2_technical_errors = synthetic_v2_failures.attrs.get("technical_errors", [])
        if synthetic_v2_technical_errors:
            stop_for_technical_errors(workbook, run_dir, run_id, checkpoint, len(dataset),
                                       "7/8 Synthetic V2", synthetic_v2_technical_errors, "SYNTHETIC_V2")
            return

        if should_run(next_stage, "COMPARISON"):
            log_stage(workbook, "8/8 Comparison", "Building V1 vs V2 comparison", "STARTED")
            save_comparison(workbook, v1_details, v2_details, synthetic_v1, synthetic_v2)
            log_stage(workbook, "8/8 Comparison", "V1 vs V2 comparison completed", "DONE")
            finalize_lifecycle(
                workbook, run_dir, run_id, checkpoint, len(dataset), "COMPARISON",
                summary_extra=[
                    {"Item": "V1 Status", "Value": "COMPLETED"},
                    {"Item": "Failure Analysis", "Value": "COMPLETED + HUMAN APPROVED"},
                    {"Item": "Synthetic Data", "Value": "GENERATED + HUMAN APPROVED"},
                    {"Item": "Synthetic V1", "Value": "COMPLETED"},
                    {"Item": "V2 Prompt", "Value": "GENERATED + HUMAN APPROVED"},
                    {"Item": "V2 Status", "Value": "COMPLETED"},
                    {"Item": "Comparison", "Value": "COMPLETED"},
                    {"Item": "Lifecycle", "Value": "STOPPED AFTER V2"},
                ],
                audit_details="No V3 or automatic prompt promotion.",
                log_path=log_path,
                title="LIFECYCLE COMPLETE",
                result_line=f"{len(dataset)} original testcase(s) -> {SYNTHETIC_TEST_CASE_COUNT} synthetic testcase(s) -> V1 -> V2 -> comparison",
            )

    except Exception as exc:
        error_text = f"{type(exc).__name__}: {exc}"
        logging.exception("Lifecycle stopped unexpectedly: %s", error_text)
        # `next_stage` here is still the stage run_lifecycle() was *entered*
        # at, not wherever the pipeline actually got to before failing --
        # every stage that completed successfully already checkpointed its
        # own next_stage via save_state() above. Reload that checkpoint
        # instead of overwriting it with the stale entry-point value, or a
        # resume would restart the whole lifecycle (and re-pay for every
        # already-completed judge call) instead of picking up where it broke.
        current_state = load_state(run_dir) or {}
        resume_from = current_state.get("next_stage", next_stage)
        last_done = current_state.get("last_completed_stage", "UNKNOWN")
        checkpoint(last_done, resume_from, "ERROR", error_text)
        try:
            append_audit(workbook, datetime.now().isoformat(timespec="seconds"), "RUN",
                         "Lifecycle stopped unexpectedly", "ERROR", error_text)
            format_workbook(workbook)  # the human is about to open this workbook to see what broke
        except Exception:
            logging.exception("Could not write the final error to the workbook.")
        print("\nLifecycle stopped because of an error.")
        print(f"Resume the run with: python -m execution.main --resume \"{run_dir}\"")
        raise


def parse_args():
    parser = argparse.ArgumentParser(description="Run the human-gated AI validation lifecycle.")
    parser.add_argument("--resume", type=str, help="Resume an existing run folder.")
    parser.add_argument(
        "--test-cases", type=int, default=None,
        help="Override TEST_CASE_LIMIT for this new run only, e.g. --test-cases 3 for a "
             "quick demo. Ignored with --resume (a resumed run keeps whatever count it started with).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.resume:
        main_resume(args.resume)
    else:
        main_new(test_case_limit=args.test_cases)
