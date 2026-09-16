"""Run orchestration.

Sequences the layers for one evaluation run: testdata -> execution
(generation) -> evaluation (metrics + scorer) -> observability (discovery,
provenance, console, reporting), narrating progress as it goes. Contains no
metric, scoring, discovery, or reporting logic of its own - those live in
their own layers.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import sys
import time

from evaluation.runner import TruthsCache, evaluate_one
from evaluation.scorer import testcase_status, testcase_verdict, weighted_score
from execution.generators.ollama_client import OllamaClient
from observability import console, provenance
from observability.analysis import improvement_suggestions, recommend
from observability.discovery import discover_models
from observability.reporting.excel import save_generator_responses, save_report
from settings import config
from testdata.loader import build_prompt, load_dataset, load_prompt


def _fix_console_encoding():
    """Force UTF-8 on stdout/stderr.

    Windows defaults them to the legacy cp1252 codec whenever they are not
    attached to an interactive console (redirected to a file/pipe, as in CI
    or `python -m execution.main > log.txt`). That codec cannot encode the U+2713/
    U+2717 progress markers this module prints, which crashes the print
    call itself - including inside the exception handler meant to report
    that same failure.
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


TOTAL_STEPS = 3


def _error_rows(model, dataset, error, run_id):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return [
        {
            "run_id": run_id,
            "test_id": row["Test_ID"],
            "generator": model,
            "model": model,
            "run_timestamp": timestamp,
            "generated_at": timestamp,
            "source": row["Source"],
            "question": row["Question"],
            "prompt": "",
            "response": "",
            "status": "ERROR",
            "error": error,
            "duration_seconds": "",
        }
        for _, row in dataset.iterrows()
    ]


def generate_for_model(model, dataset, prompt_template, run_id):
    client = OllamaClient(
        config.OLLAMA_BASE_URL,
        config.REQUEST_TIMEOUT,
        config.RETRIES,
        config.TEMPERATURE,
    )
    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []

    for _, row in dataset.iterrows():
        started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prompt = ""

        try:
            prompt = build_prompt(prompt_template, row)
            result = client.generate(model, prompt)
            rows.append({
                "run_id": run_id,
                "test_id": row["Test_ID"],
                "generator": model,
                "model": model,
                "run_timestamp": run_timestamp,
                "generated_at": started,
                "source": row["Source"],
                "question": row["Question"],
                "prompt": prompt,
                "response": result.get("response", ""),
                "status": result.get("status", "ERROR"),
                "error": result.get("error", ""),
                "duration_seconds": result.get("duration_seconds", ""),
            })
        except Exception as exc:
            rows.append({
                "run_id": run_id,
                "test_id": row["Test_ID"],
                "generator": model,
                "model": model,
                "run_timestamp": run_timestamp,
                "generated_at": started,
                "source": row["Source"],
                "question": row["Question"],
                "prompt": prompt,
                "response": "",
                "status": "ERROR",
                "error": f"{type(exc).__name__}: {exc}",
                "duration_seconds": "",
            })

    return rows


def _avg_duration(rows):
    durations = [
        row["duration_seconds"]
        for row in rows
        if isinstance(row.get("duration_seconds"), (int, float))
    ]
    return sum(durations) / len(durations) if durations else None


def generate_responses(models, dataset, prompt, run_id):
    responses = {}
    workers = min(config.GENERATOR_CONCURRENCY, len(models))

    console.phase_banner(1, TOTAL_STEPS, "Generating responses")
    console.generator_concurrency_note(workers, len(models))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                generate_for_model, model, dataset, prompt, run_id
            ): model
            for model in models
        }
        for future in as_completed(futures):
            model = futures[future]
            try:
                responses[model] = future.result()
                completed = sum(
                    row["status"] == "COMPLETED"
                    for row in responses[model]
                )
                console.generator_completed(
                    model, completed, len(dataset),
                    _avg_duration(responses[model]),
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                responses[model] = _error_rows(model, dataset, error, run_id)
                console.generator_failed(model, error)

    return responses


def _failure_type(result):
    if result.get("status") != "COMPLETED" or result.get("score") is None:
        return "TECHNICAL ERROR"
    return "QUALITY ISSUE" if result.get("verdict") != "PASS" else ""


def evaluate_one_generator(model, judge, responses, source_by_id, truths_cache):
    detail_rows = []
    failures = []
    testcase_rows = []
    model_scores = []

    for test_index, response in enumerate(responses[model], start=1):
        test_id = str(response["test_id"])
        metrics = evaluate_one(
            response,
            source_by_id[test_id],
            judge,
            config.OLLAMA_BASE_URL,
            truths_cache,
        )
        score = weighted_score(metrics)
        verdict = testcase_verdict(metrics)
        status = testcase_status(metrics)

        problem_metrics = []
        for name, result in metrics.items():
            detail_rows.append({
                "generator": model,
                "judge": judge,
                "test_id": test_id,
                "metric": name,
                "score": result["score"],
                "verdict": result["verdict"],
                "passed": result["passed"],
                "status": result["status"],
                "reason": result["reason"],
                "error": result["error"],
                "duration_seconds": result["duration_seconds"],
                "llm_call_timings": result.get("llm_call_timings", []),
            })

            if (
                result["status"] != "COMPLETED"
                or result["verdict"] != "PASS"
            ):
                failures.append({
                    "generator": model,
                    "judge": judge,
                    "test_id": test_id,
                    "metric": name,
                    "score": result["score"],
                    "verdict": result["verdict"],
                    "reason": result["reason"],
                    "status": result["status"],
                    "error": result["error"],
                    "failure_type": _failure_type(result),
                })
                problem_metrics.append(
                    (name, result["verdict"], result["score"], result["reason"])
                )

        testcase_rows.append({
            "generator": model,
            "test_id": test_id,
            "testcase_verdict": verdict,
            "testcase_status": status,
            **{
                f"{name}_score": result["score"]
                for name, result in metrics.items()
            },
        })
        model_scores.append(score)

        console.testcase_result(
            model, test_index, len(responses[model]), test_id, status
        )
        for name, metric_verdict_value, metric_score, reason in problem_metrics:
            console.metric_failure_detail(
                name, metric_verdict_value, metric_score, reason
            )

    # A metric's per-generator average is None, not silently computed from
    # only the successful rows, if any evaluation of that metric for this
    # generator was technically incomplete - same rule as weighted_score().
    metric_scores = {}
    for name in config.METRIC_NAMES:
        rows_for_metric = [
            row for row in detail_rows if row["metric"] == name
        ]
        if not rows_for_metric or any(
            row["status"] != "COMPLETED" or row["score"] is None
            for row in rows_for_metric
        ):
            metric_scores[name] = None
        else:
            metric_scores[name] = round(
                sum(row["score"] for row in rows_for_metric)
                / len(rows_for_metric),
                4,
            )

    technical_error_count = sum(
        row["testcase_status"] == "TECHNICAL ERROR" for row in testcase_rows
    )
    quality_fail_count = sum(
        row["testcase_status"] == "QUALITY FAIL" for row in testcase_rows
    )
    quality_review_count = sum(
        row["testcase_status"] == "QUALITY REVIEW" for row in testcase_rows
    )

    summary = {
        "generator": model,
        "overall_score": (
            round(sum(model_scores) / len(model_scores), 4)
            if model_scores and all(s is not None for s in model_scores)
            else None
        ),
        "passed_testcases": sum(
            row["testcase_verdict"] == "PASS" for row in testcase_rows
        ),
        "total_testcases": len(testcase_rows),
        "quality_gate": (
            "PASS"
            if testcase_rows
            and all(
                row["testcase_verdict"] == "PASS" for row in testcase_rows
            )
            else "FAIL"
        ),
        "technical_error_count": technical_error_count,
        "quality_fail_count": quality_fail_count,
        "quality_review_count": quality_review_count,
        "metric_scores": metric_scores,
    }

    return {
        "summary": summary,
        "testcase_rows": testcase_rows,
        "detail_rows": detail_rows,
        "failures": failures,
    }


def evaluate_generators(models, judge, responses, dataset):
    source_by_id = {
        str(row["Test_ID"]): row for _, row in dataset.iterrows()
    }
    # Truths extraction only depends on the shared source context, not on
    # which generator produced the answer, so it is cached once per test_id
    # and reused across generators instead of being recomputed for each one.
    truths_cache = TruthsCache()

    detail_rows = []
    failures = []
    summaries = []
    testcase_rows = []

    workers = min(config.EVALUATION_CONCURRENCY, len(models))

    console.phase_banner(2, TOTAL_STEPS, "Evaluating responses with DeepEval")
    console.evaluation_intro(
        len(models), workers, len(config.METRIC_NAMES), judge
    )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                evaluate_one_generator,
                model,
                judge,
                responses,
                source_by_id,
                truths_cache,
            ): model
            for model in models
        }

        for future in as_completed(futures):
            model = futures[future]
            try:
                result = future.result()
                summaries.append(result["summary"])
                testcase_rows.extend(result["testcase_rows"])
                detail_rows.extend(result["detail_rows"])
                failures.extend(result["failures"])
                console.generator_evaluation_summary(
                    model,
                    result["summary"]["passed_testcases"],
                    result["summary"]["total_testcases"],
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                console.generator_evaluation_failed(model, error)
                raise

    return {
        "summary": summaries,
        "testcase_rows": testcase_rows,
        "detail_rows": detail_rows,
        "failures": failures,
    }


def build_summary(verdict, suggestions, summaries, run_id):
    rows = [
        {"section": "Run ID", "value": run_id},
        {"section": "Final Verdict", "value": verdict},
        {
            "section": "Sample Size",
            "value": (
                f"{max((row.get('total_testcases', 0) for row in summaries), default=0)} "
                "test cases in this run. A small smoke sample should not be "
                "treated as stable model-ranking evidence."
            ),
        },
        {
            "section": "Testcase Gate",
            "value": (
                f"Weighted score >= {config.QUALITY_THRESHOLD:.2f} for PASS; "
                f"{config.WARNING_THRESHOLD:.2f} to <{config.QUALITY_THRESHOLD:.2f} "
                f"for REVIEW; below {config.WARNING_THRESHOLD:.2f} for FAIL. "
                "Hallucination, Faithfulness and Correctness must not fall "
                "below their configured REVIEW thresholds."
            ),
        },
        {
            "section": "Metric Thresholds",
            "value": " | ".join(
                f"{name}: PASS >= {values['pass']:.2f}, "
                f"REVIEW >= {values['review']:.2f}"
                for name, values in config.METRIC_THRESHOLDS.items()
            ),
        },
        {
            "section": "Metric Roles",
            "value": " | ".join(
                f"{name}: {role}"
                for name, role in config.METRIC_ROLES.items()
            ),
        },
        {
            "section": "Selection Logic",
            "value": (
                "Quality gate first, followed by weighted quality score and "
                "testcase pass rate. Operational measures such as latency, "
                "technical errors and response stability remain separate."
            ),
        },
        {
            "section": "Prompt/Test Data Improvements",
            "value": (
                " | ".join(suggestions)
                if suggestions else "No repeated failure pattern was detected."
            ),
        },
    ]

    for gen_summary in summaries:
        row = gen_summary.copy()
        for name, score in row.pop("metric_scores", {}).items():
            row[f"{name}_score"] = round(score, 4) if score is not None else None
        rows.append(row)
    return rows


def build_configuration(discovered, run_id, prompt):
    return [
        {"key": "Run ID", "value": run_id},
        {"key": "Python Version", "value": provenance.python_version()},
        {
            "key": "DeepEval Version",
            "value": provenance.package_version("deepeval"),
        },
        {"key": "Ollama Base URL", "value": config.OLLAMA_BASE_URL},
        {"key": "Temperature", "value": config.TEMPERATURE},
        {"key": "Generators", "value": ", ".join(discovered["generators"])},
        {
            "key": "Generator Digests",
            "value": provenance.model_digests(
                discovered["generators"], discovered["server_models"],
            ),
        },
        {"key": "Judge Provider", "value": config.JUDGE_PROVIDER},
        {"key": "Judge", "value": ", ".join(discovered["judges"])},
        {
            "key": "Judge Digests",
            "value": provenance.model_digests(
                discovered["judges"], discovered["server_models"],
            ),
        },
        {"key": "Generator Concurrency", "value": config.GENERATOR_CONCURRENCY},
        {"key": "Metric Concurrency", "value": config.JUDGE_CONCURRENCY},
        {"key": "Evaluation Concurrency", "value": config.EVALUATION_CONCURRENCY},
        {"key": "Dataset", "value": str(config.DATASET_PATH)},
        {
            "key": "Dataset SHA256",
            "value": provenance.sha256_file(config.DATASET_PATH),
        },
        {"key": "Prompt SHA256", "value": provenance.sha256_text(prompt)},
        {"key": "Test Case Limit", "value": config.TEST_CASE_LIMIT},
        {"key": "Retries", "value": config.RETRIES},
        {"key": "Request Timeout (s)", "value": config.REQUEST_TIMEOUT},
        {
            "key": "Metric Thresholds",
            "value": " | ".join(
                f"{name}: PASS {values['pass']:.2f}, "
                f"REVIEW {values['review']:.2f}"
                for name, values in config.METRIC_THRESHOLDS.items()
            ),
        },
    ]


def build_run_summary(run_id, phase_timings, summaries, dataset_size, judge):
    total = phase_timings.get("Total run time", 0.0)
    technical_errors = sum(
        int(row.get("technical_error_count") or 0) for row in summaries
    )
    quality_failures = sum(
        int(row.get("quality_fail_count") or 0) for row in summaries
    )
    passed = sum(int(row.get("passed_testcases") or 0) for row in summaries)

    rows = [
        {"section": "Run ID", "value": run_id},
        {"section": "Dataset Test Cases", "value": dataset_size},
        {"section": "Generators", "value": len(summaries)},
        {"section": "Judge Provider", "value": config.JUDGE_PROVIDER},
        {"section": "Judge", "value": judge},
        {"section": "Total Testcases Passed", "value": passed},
        {"section": "Technical Errors", "value": technical_errors},
        {"section": "Quality-Fail Testcases", "value": quality_failures},
        {"section": "Runtime Target (s)", "value": config.RUNTIME_TARGET_SECONDS},
        {
            "section": "Runtime Target Status",
            "value": (
                "PASS" if total <= config.RUNTIME_TARGET_SECONDS else "NOT MET"
            ),
        },
    ]

    for label, seconds in phase_timings.items():
        percentage = (
            round(seconds / total * 100, 2)
            if total and label != "Total run time"
            else ""
        )
        rows.append({
            "section": label,
            "value": f"{seconds:.2f}s"
            + (f" ({percentage:.2f}%)" if percentage != "" else ""),
        })

    return rows


def summarize_metric_durations(detail_rows):
    durations = {}
    for row in detail_rows:
        duration = row.get("duration_seconds")
        if isinstance(duration, (int, float)):
            durations.setdefault(row["metric"], []).append(duration)

    return {
        name: round(sum(values) / len(values), 2)
        for name, values in durations.items()
    }


def main():
    _fix_console_encoding()
    run_id = provenance.new_run_id()
    run_started = time.perf_counter()
    phase_timings = {}

    try:
        console.banner("DeepEval Multi-Generator Evaluation")
        console.run_id_line(run_id)

        phase_started = time.perf_counter()
        print("Discovering Ollama models...")
        discovered = discover_models()
        models = discovered["generators"]
        judge = discovered["judges"][0]
        phase_timings["Model discovery"] = time.perf_counter() - phase_started

        console.discovery_line(models, judge, config.JUDGE_PROVIDER)
        console.dataset_loading_line(config.DATASET_PATH.name)

        phase_started = time.perf_counter()
        dataset = load_dataset()
        prompt = load_prompt()
        phase_timings["Dataset load"] = time.perf_counter() - phase_started

        console.dataset_loaded_line(
            len(dataset), config.TEST_CASE_LIMIT, config.TEMPERATURE
        )

        phase_started = time.perf_counter()
        responses = generate_responses(models, dataset, prompt, run_id)
        response_file = save_generator_responses(
            responses, config.RESPONSE_ROOT, run_id=run_id
        )
        phase_timings["Generation phase"] = time.perf_counter() - phase_started
        console.file_written("Generator workbook", response_file)

        phase_started = time.perf_counter()
        results = evaluate_generators(models, judge, responses, dataset)
        phase_timings["Evaluation phase"] = time.perf_counter() - phase_started

        for collection in ("summary", "testcase_rows", "detail_rows", "failures"):
            for row in results.get(collection, []):
                if isinstance(row, dict):
                    row.setdefault("run_id", run_id)

        verdict = recommend(
            results["summary"],
            minimum_testcases=config.MINIMUM_TESTCASES_FOR_RANKING,
        )
        suggestions = improvement_suggestions(results["failures"])

        console.phase_banner(3, TOTAL_STEPS, "Building report")

        phase_started = time.perf_counter()
        # Report generation and total run time aren't known until the
        # report is built, so build once with placeholder timings, then
        # rebuild with the measured values - report generation is well
        # under a second, so writing it twice costs nothing meaningful.
        report_file = save_report(
            build_summary(verdict, suggestions, results["summary"], run_id),
            results["testcase_rows"],
            results["detail_rows"],
            results["failures"],
            build_configuration(discovered, run_id, prompt),
            config.REPORT_ROOT,
            run_summary=build_run_summary(
                run_id,
                {**phase_timings, "Report generation": 0.0, "Total run time": 0.0},
                results["summary"],
                len(dataset),
                judge,
            ),
            run_id=run_id,
        )
        phase_timings["Report generation"] = time.perf_counter() - phase_started
        phase_timings["Total run time"] = time.perf_counter() - run_started

        report_file = save_report(
            build_summary(verdict, suggestions, results["summary"], run_id),
            results["testcase_rows"],
            results["detail_rows"],
            results["failures"],
            build_configuration(discovered, run_id, prompt),
            config.REPORT_ROOT,
            run_summary=build_run_summary(
                run_id, phase_timings, results["summary"], len(dataset), judge,
            ),
            run_id=run_id,
        )

        console.completion_summary(report_file, verdict)
        console.timing_report(
            phase_timings,
            summarize_metric_durations(results["detail_rows"]),
        )

    except Exception as exc:
        console.run_failed(exc)
        raise


if __name__ == "__main__":
    main()
