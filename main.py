from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import argparse
import json
import logging
import sys
from pathlib import Path
import pandas as pd

from analysis.failure_analysis import analyze
from config import (
    DATASET_PATH,
    GENERATOR_MODEL,
    JUDGE_MODEL,
    OLLAMA_BASE_URL,
    PROMPT_ROOT,
    REPORT_ROOT,
    SYNTHETIC_TEST_CASE_COUNT,
    TEMPERATURE,
    TEST_CASE_LIMIT,
)
from deepeval_framework.evaluator import evaluate_response
from llm_client import ollama_generate
from prompt_optimizer import optimize_prompt
from prompts import BASE_PROMPT
from reporting.excel import (
    append_audit,
    read_sheet,
    save_approval,
    save_comparison,
    save_evaluation,
    save_failure_analysis,
    save_prompt_version,
    save_summary,
)
from synthetic_data import generate_synthetic_dataset, validate_synthetic_dataset

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


def configure_logging(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def state_path(run_dir):
    return run_dir / "run_state.json"


def save_state(run_dir, run_id, last_completed_stage, next_stage, status="RUNNING", error=""):
    state = {
        "run_id": run_id,
        "last_completed_stage": last_completed_stage,
        "next_stage": next_stage,
        "status": status,
    }
    if error:
        state["error"] = error
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


def load_data(path):
    dataset = pd.read_excel(path)
    required = {"Test_ID", "Source", "Question", "Golden_Answer"}
    missing = required - set(dataset.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if dataset.empty:
        raise ValueError("Dataset is empty.")
    if dataset["Test_ID"].duplicated().any():
        raise ValueError("Dataset contains duplicate Test_ID values.")

    dataset = dataset.head(TEST_CASE_LIMIT).copy()
    if len(dataset) != TEST_CASE_LIMIT:
        raise ValueError(
            f"TEST_CASE_LIMIT is {TEST_CASE_LIMIT}, but the dataset contains only {len(dataset)} testcase(s)."
        )
    return dataset


def generate(row, prompt_template):
    prompt = prompt_template.format(
        source=str(row["Source"]),
        question=str(row["Question"]),
    )
    return ollama_generate(GENERATOR_MODEL, prompt, temperature=TEMPERATURE)


def run_generation_and_evaluation(dataset, prompt_template, label):
    logging.info("[%s] Generating %d responses...", label, len(dataset))
    generated = {}
    generation_errors = {}

    with ThreadPoolExecutor(max_workers=min(3, max(1, len(dataset)))) as pool:
        futures = {
            pool.submit(generate, row, prompt_template): str(row["Test_ID"])
            for _, row in dataset.iterrows()
        }
        for future in as_completed(futures):
            test_id = futures[future]
            try:
                generated[test_id] = future.result()
            except Exception as exc:
                generation_errors[test_id] = f"{type(exc).__name__}: {exc}"

    logging.info("[%s] Generation complete.", label)
    rows = []

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
                "latency_seconds": None,
            })
            continue

        logging.info("[%s] Evaluating %s...", label, test_id)
        results = evaluate_response(row, response)
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
                "latency_seconds": result["latency_seconds"],
            })

    result_frame = pd.DataFrame(rows)
    completed = int((result_frame.get("status", pd.Series(dtype=str)) == "COMPLETED").sum()) if not result_frame.empty else 0
    errors = int((result_frame.get("status", pd.Series(dtype=str)) != "COMPLETED").sum()) if not result_frame.empty else 0
    logging.info("[%s] DeepEval evaluation complete. Metric results=%d, non-completed=%d.", label, completed, errors)
    return result_frame


def update_summary(workbook, run_id, extra=None):
    rows = [
        {"Item": "Run ID", "Value": run_id},
        {"Item": "Original Testcase Count", "Value": TEST_CASE_LIMIT},
        {"Item": "Synthetic Testcase Count", "Value": SYNTHETIC_TEST_CASE_COUNT},
        {"Item": "Generator", "Value": GENERATOR_MODEL},
        {"Item": "Judge", "Value": JUDGE_MODEL},
        {"Item": "Ollama Endpoint", "Value": OLLAMA_BASE_URL},
        {"Item": "Temperature", "Value": TEMPERATURE},
        {"Item": "Lifecycle", "Value": "Human-gated; one testcase; stops after V2 comparison"},
    ]
    if extra:
        rows.extend(extra)
    save_summary(workbook, rows)


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
    workbooks = sorted(run_dir.glob("AI_validation_run_*.xlsx"))
    if not workbooks:
        raise FileNotFoundError(f"No lifecycle workbook found in {run_dir}")
    return workbooks[-1]


def infer_resume_stage(workbook):
    """Create a checkpoint for a legacy run that predates run_state.json."""
    sheets = set(pd.ExcelFile(workbook).sheet_names)

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
    if next_stage in {"STOP", "COMPLETE"}:
        return False
    if next_stage not in STAGES:
        raise ValueError(f"Unknown resume stage: {next_stage}")
    return STAGES.index(stage) >= STAGES.index(next_stage)


def main_new():
    start = datetime.now()
    date_part = start.strftime("%Y-%m-%d")
    time_part = start.strftime("%H%M%S")
    run_id = f"{date_part}_{time_part}"

    run_dir = REPORT_ROOT / date_part / f"run_{time_part}"
    run_dir.mkdir(parents=True, exist_ok=True)
    workbook = run_dir / f"AI_validation_run_{start.strftime('%Y%m%d_%H%M%S')}.xlsx"
    log_path = run_dir / "run.log"
    configure_logging(log_path)
    return run_lifecycle(run_id, run_dir, workbook, log_path, "V1")


def main_resume(run_dir):
    run_dir = Path(run_dir).resolve()
    workbook = find_workbook(run_dir)
    log_path = run_dir / "run.log"
    configure_logging(log_path)

    state = load_state(run_dir)
    if state:
        next_stage = state.get("next_stage", "V1")
        run_id = state.get("run_id", run_dir.name.replace("run_", ""))
        # A terminal checkpoint is trusted only when the workbook contains
        # the final comparison sheet. Older buggy checkpoints could say STOP
        # even though the lifecycle stopped earlier, so recover from the
        # workbook when the final artifacts are missing.
        if str(next_stage).upper() in {"STOP", "COMPLETE"}:
            workbook_sheets = set(pd.ExcelFile(workbook).sheet_names)
            required_final_sheets = {
                "08_V2_Evaluation",
                "09_Synthetic_V2",
                "10_V1_vs_V2",
                "11_Audit_Log",
            }
            if required_final_sheets.issubset(workbook_sheets):
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
                save_state(run_dir, run_id, "RECOVERED", next_stage, "PAUSED")
    else:
        next_stage = infer_resume_stage(workbook)
        run_id = run_dir.parent.name + "_" + run_dir.name.replace("run_", "")
        save_state(run_dir, run_id, "NONE", next_stage, "PAUSED")
        logging.info("No run_state.json found. Created legacy checkpoint; resume starts at %s.", next_stage)

    return run_lifecycle(run_id, run_dir, workbook, log_path, next_stage)


def run_lifecycle(run_id, run_dir, workbook, log_path, next_stage):
    start = datetime.now()
    dataset = load_data(DATASET_PATH)
    prompt_dir = PROMPT_ROOT / start.strftime("%Y-%m-%d")
    prompt_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 68)
    print("AI VALIDATION LIFECYCLE - ONE TESTCASE POC")
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
        print("Result       : 1 original testcase -> 1 synthetic testcase -> V1 -> V2 -> comparison")
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
            log_stage(workbook, "1/8 V1", "Baseline evaluation completed", "DONE")
            save_state(run_dir, run_id, "V1", "FAILURE_ANALYSIS")
        else:
            v1_details = read_sheet(workbook, "02_V1_Evaluation")

        if should_run(next_stage, "FAILURE_ANALYSIS"):
            log_stage(workbook, "2/8 Analysis", "Failure analysis started", "STARTED")
            failures, recommendations = analyze(v1_details)
            save_failure_analysis(workbook, failures, recommendations)
            log_stage(workbook, "2/8 Analysis", "Failure analysis completed", "DONE",
                      f"{len(failures)} failed/review results; {len(recommendations)} recurring patterns.")
            save_state(run_dir, run_id, "FAILURE_ANALYSIS", "HUMAN_APPROVAL_FAILURE_ANALYSIS")
        else:
            failures, recommendations = analyze(v1_details)

        if should_run(next_stage, "HUMAN_APPROVAL_FAILURE_ANALYSIS"):
            if not human_gate(
                workbook,
                "Failure Analysis",
                f"Review '03_Failure_Analysis' in:\n{workbook}\n\nConfirm that the findings are valid and prompt improvement is appropriate.",
            ):
                save_state(run_dir, run_id, "HUMAN_APPROVAL_FAILURE_ANALYSIS", "SYNTHETIC_DATA", "STOPPED")
                return
            save_state(run_dir, run_id, "HUMAN_APPROVAL_FAILURE_ANALYSIS", "SYNTHETIC_DATA")

        if should_run(next_stage, "SYNTHETIC_DATA"):
            log_stage(workbook, "3/8 Synthesis", "Generating targeted synthetic data", "STARTED")
            synthetic_working = run_dir / "synthetic_working.xlsx"
            generate_synthetic_dataset(dataset, failures, synthetic_working)
            synthetic_df = pd.read_excel(synthetic_working)
            validate_synthetic_dataset(synthetic_df)
            save_evaluation(workbook, "05_Synthetic_Data", synthetic_df)
            log_stage(workbook, "3/8 Synthesis", "Targeted synthetic data generated", "DONE",
                      f"{len(synthetic_df)} cases created.")
            save_state(run_dir, run_id, "SYNTHETIC_DATA", "HUMAN_APPROVAL_SYNTHETIC_DATA")
        else:
            synthetic_df = read_sheet(workbook, "05_Synthetic_Data")
            validate_synthetic_dataset(synthetic_df)

        if should_run(next_stage, "HUMAN_APPROVAL_SYNTHETIC_DATA"):
            if not human_gate(
                workbook,
                "Synthetic Data",
                f"Review and edit '05_Synthetic_Data' in:\n{workbook}\n\nCorrect questions/golden answers if required. When you are satisfied, enter Y to approve the reviewed synthetic testcase(s). The system will record Human_Reviewed = APPROVED.",
            ):
                save_state(run_dir, run_id, "HUMAN_APPROVAL_SYNTHETIC_DATA", "SYNTHETIC_V1", "STOPPED")
                return
            synthetic_df = read_sheet(workbook, "05_Synthetic_Data")
            validate_synthetic_dataset(synthetic_df)

            # The console Y is the human approval gate. The workbook column is
            # the persisted audit status, so record the approval here rather
            # than requiring the reviewer to perform two separate approvals.
            synthetic_df["Human_Reviewed"] = "APPROVED"
            save_evaluation(workbook, "05_Synthetic_Data", synthetic_df)

            save_state(run_dir, run_id, "HUMAN_APPROVAL_SYNTHETIC_DATA", "SYNTHETIC_V1")

        if should_run(next_stage, "SYNTHETIC_V1"):
            log_stage(workbook, "4/8 Synthetic V1", "Validating V1 on synthetic data", "STARTED")
            synthetic_v1 = run_generation_and_evaluation(synthetic_df, BASE_PROMPT, "Synthetic V1")
            save_evaluation(workbook, "06_Synthetic_V1", synthetic_v1)
            log_stage(workbook, "4/8 Synthetic V1", "Synthetic V1 validation completed", "DONE")
            save_state(run_dir, run_id, "SYNTHETIC_V1", "OPTIMIZER")
        else:
            synthetic_v1 = read_sheet(workbook, "06_Synthetic_V1")

        if should_run(next_stage, "OPTIMIZER"):
            log_stage(workbook, "5/8 Optimizer", "Generating candidate V2 prompt", "STARTED")
            time_part = datetime.now().strftime("%H%M%S")
            v2_prompt_path = prompt_dir / f"prompt_v2_candidate_{time_part}.txt"
            v2_prompt = optimize_prompt(BASE_PROMPT, failures, synthetic_df, v2_prompt_path)
            save_prompt_version(workbook, "V2 Candidate", v2_prompt, "PENDING HUMAN APPROVAL")
            log_stage(workbook, "5/8 Optimizer", "Candidate V2 prompt generated", "DONE",
                      f"Saved as {v2_prompt_path.name}.")
            save_state(run_dir, run_id, "OPTIMIZER", "HUMAN_APPROVAL_V2_PROMPT")

        if should_run(next_stage, "HUMAN_APPROVAL_V2_PROMPT"):
            v2_prompt_path = sorted(prompt_dir.glob("prompt_v2_candidate_*.txt"))[-1]
            if not human_gate(
                workbook,
                "V2 Prompt",
                f"Review '07_Prompt_Versions' and:\n{v2_prompt_path}\n\nConfirm the changes address V1 failures without changing the original task. You may edit the V2 Candidate prompt in the workbook before approving.",
            ):
                save_state(run_dir, run_id, "HUMAN_APPROVAL_V2_PROMPT", "V2", "STOPPED")
                return
            v2_prompt = read_approved_v2_prompt(workbook)
            save_prompt_version(workbook, "V2 Candidate", v2_prompt, "APPROVED FOR EXECUTION")
            save_state(run_dir, run_id, "HUMAN_APPROVAL_V2_PROMPT", "V2")
        else:
            v2_prompt = read_approved_v2_prompt(workbook)

        if should_run(next_stage, "V2"):
            log_stage(workbook, "6/8 V2", "Running approved V2 prompt", "STARTED")
            v2_details = run_generation_and_evaluation(dataset, v2_prompt, "V2")
            save_evaluation(workbook, "08_V2_Evaluation", v2_details)
            log_stage(workbook, "6/8 V2", "V2 evaluation completed", "DONE")
            save_state(run_dir, run_id, "V2", "SYNTHETIC_V2")
        else:
            v2_details = read_sheet(workbook, "08_V2_Evaluation")

        if should_run(next_stage, "SYNTHETIC_V2"):
            log_stage(workbook, "7/8 Synthetic V2", "Validating V2 on synthetic data", "STARTED")
            synthetic_v2 = run_generation_and_evaluation(synthetic_df, v2_prompt, "Synthetic V2")
            save_evaluation(workbook, "09_Synthetic_V2", synthetic_v2)
            log_stage(workbook, "7/8 Synthetic V2", "Synthetic V2 validation completed", "DONE")
            save_state(run_dir, run_id, "SYNTHETIC_V2", "COMPARISON")
        else:
            synthetic_v2 = read_sheet(workbook, "09_Synthetic_V2")

        if should_run(next_stage, "COMPARISON"):
            log_stage(workbook, "8/8 Comparison", "Building V1 vs V2 comparison", "STARTED")
            save_comparison(workbook, v1_details, v2_details, synthetic_v1, synthetic_v2)
            update_summary(workbook, run_id, [
                {"Item": "V1 Status", "Value": "COMPLETED"},
                {"Item": "Failure Analysis", "Value": "COMPLETED + HUMAN APPROVED"},
                {"Item": "Synthetic Data", "Value": "GENERATED + HUMAN APPROVED"},
                {"Item": "Synthetic V1", "Value": "COMPLETED"},
                {"Item": "V2 Prompt", "Value": "GENERATED + HUMAN APPROVED"},
                {"Item": "V2 Status", "Value": "COMPLETED"},
                {"Item": "Comparison", "Value": "COMPLETED"},
                {"Item": "Lifecycle", "Value": "STOPPED AFTER V2"},
            ])
            log_stage(workbook, "8/8 Comparison", "V1 vs V2 comparison completed", "DONE")
            append_audit(workbook, datetime.now().isoformat(timespec="seconds"), "RUN",
                         "Lifecycle stopped", "COMPLETE", "No V3 or automatic prompt promotion.")
            save_state(run_dir, run_id, "COMPARISON", "COMPLETE", "COMPLETE")

        print("\n" + "=" * 68)
        print("LIFECYCLE COMPLETE")
        print("=" * 68)
        print(f"Master Excel : {workbook}")
        print(f"Run Log      : {log_path}")
        print("Result       : 1 original testcase -> 1 synthetic testcase -> V1 -> V2 -> comparison")
        print("Next step    : STOP")
        print("=" * 68)

    except Exception as exc:
        error_text = f"{type(exc).__name__}: {exc}"
        logging.exception("Lifecycle stopped unexpectedly: %s", error_text)
        save_state(run_dir, run_id, "UNKNOWN", next_stage, "ERROR", error_text)
        try:
            append_audit(workbook, datetime.now().isoformat(timespec="seconds"), "RUN",
                         "Lifecycle stopped unexpectedly", "ERROR", error_text)
        except Exception:
            logging.exception("Could not write the final error to the workbook.")
        print("\nLifecycle stopped because of an error.")
        print(f"Resume the run with: python main.py --resume \"{run_dir}\"")
        raise


def parse_args():
    parser = argparse.ArgumentParser(description="Run the human-gated AI validation lifecycle.")
    parser.add_argument("--resume", type=str, help="Resume an existing run folder.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.resume:
        main_resume(args.resume)
    else:
        main_new()
