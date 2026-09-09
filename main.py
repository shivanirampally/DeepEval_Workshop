from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import pandas as pd

import config
from analysis.recommendation import improvement_suggestions, recommend
from config import discover_models
from deepeval_framework.evaluation import evaluate_one
from deepeval_framework.scoring import testcase_verdict, weighted_score
from generators.ollama_client import OllamaClient
from reporting.excel import save_generator_responses, save_report


def load_dataset():
    dataset = pd.read_excel(config.DATASET_PATH, sheet_name="Test_Data")
    if config.TEST_CASE_LIMIT > len(dataset):
        raise ValueError(
            f"test_case_limit={config.TEST_CASE_LIMIT} but dataset contains "
            f"only {len(dataset)} test cases."
        )
    return dataset.head(config.TEST_CASE_LIMIT).copy()


def load_prompt():
    prompts = pd.read_excel(
        config.DATASET_PATH, sheet_name="Prompt_Versions"
    )
    return str(prompts.iloc[0]["Prompt"])


def build_prompt(template, row):
    return template.format(
        source=str(row["Source"]),
        question=str(row["Question"]),
    )


def _error_rows(model, dataset, error):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return [
        {
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


def generate_for_model(model, dataset, prompt_template):
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


def generate_responses(models, dataset, prompt):
    responses = {}
    workers = min(config.GENERATOR_CONCURRENCY, len(models))

    print(f"Generation : {workers} generators in parallel")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(generate_for_model, model, dataset, prompt): model
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
                print(
                    f"  ✓ {model}: {completed}/{len(dataset)} "
                    "responses completed"
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                responses[model] = _error_rows(model, dataset, error)
                print(f"  ✗ {model}: generation failed - {error}")

    return responses


def evaluate_generators(models, judge, responses, dataset):
    source_by_id = {
        str(row["Test_ID"]): row for _, row in dataset.iterrows()
    }
    detail_rows = []
    failures = []
    summaries = []
    testcase_rows = []

    for model_index, model in enumerate(models, start=1):
        print(f"\n[{model_index}/{len(models)}] Evaluating {model}")
        model_scores = []
        model_testcase_rows = []

        for test_index, response in enumerate(responses[model], start=1):
            test_id = str(response["test_id"])
            metrics = evaluate_one(
                response,
                source_by_id[test_id],
                judge,
                config.OLLAMA_BASE_URL,
            )
            score = weighted_score(metrics)
            verdict = testcase_verdict(metrics)

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
                    })

            testcase_row = {
                "generator": model,
                "test_id": test_id,
                "testcase_verdict": verdict,
                **{
                    f"{name}_score": result["score"]
                    for name, result in metrics.items()
                },
            }
            testcase_rows.append(testcase_row)
            model_testcase_rows.append(testcase_row)
            model_scores.append(score)

            print(
                f"    [{test_index}/{len(responses[model])}] "
                f"{test_id}: {verdict}"
            )

        valid_scores = [score for score in model_scores if score is not None]
        metric_scores = {}
        for name in config.METRIC_NAMES:
            values = [
                row["score"]
                for row in detail_rows
                if row["generator"] == model
                and row["metric"] == name
                and row["score"] is not None
            ]
            if values:
                metric_scores[name] = sum(values) / len(values)

        summaries.append({
            "generator": model,
            "overall_score": (
                round(sum(valid_scores) / len(valid_scores), 4)
                if valid_scores else None
            ),
            "passed_testcases": sum(
                row["testcase_verdict"] == "PASS"
                for row in model_testcase_rows
            ),
            "total_testcases": len(model_testcase_rows),
            "quality_gate": (
                "PASS"
                if model_testcase_rows
                and all(
                    row["testcase_verdict"] == "PASS"
                    for row in model_testcase_rows
                )
                else "FAIL"
            ),
            "metric_scores": metric_scores,
        })

    return {
        "summary": summaries,
        "testcase_rows": testcase_rows,
        "detail_rows": detail_rows,
        "failures": failures,
    }


def build_summary(verdict, suggestions, summaries):
    rows = [
        {"section": "Final Verdict", "value": verdict},
        {
            "section": "Testcase Gate",
            "value": (
                "Weighted score >= 0.80 for PASS; 0.60 to <0.80 for REVIEW; "
                "below 0.60 for FAIL. Hallucination, Faithfulness and "
                "Correctness must not fall below their configured REVIEW thresholds."
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
                "Quality gate first, followed by weighted quality score, "
                "testcase pass rate and grounding metrics. "
                "Single-judge mode is used with an independent judge model."
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

    for summary in summaries:
        row = summary.copy()
        for name, score in row.pop("metric_scores", {}).items():
            row[f"{name}_score"] = round(score, 4)
        rows.append(row)
    return rows


def build_configuration(discovered):
    return [
        {"key": "Ollama Base URL", "value": config.OLLAMA_BASE_URL},
        {"key": "Temperature", "value": config.TEMPERATURE},
        {"key": "Generators", "value": ", ".join(discovered["generators"])},
        {"key": "Judge", "value": ", ".join(discovered["judges"])},
        {"key": "Generator Concurrency", "value": config.GENERATOR_CONCURRENCY},
        {"key": "Metric Concurrency", "value": config.JUDGE_CONCURRENCY},
        {"key": "Dataset", "value": str(config.DATASET_PATH)},
        {"key": "Test Case Limit", "value": config.TEST_CASE_LIMIT},
        {
            "key": "Metric Thresholds",
            "value": " | ".join(
                f"{name}: PASS {values['pass']:.2f}, "
                f"REVIEW {values['review']:.2f}"
                for name, values in config.METRIC_THRESHOLDS.items()
            ),
        },
    ]


def main():
    try:
        print("=" * 60)
        print("DeepEval Multi-Generator Evaluation")
        print("=" * 60)

        print("Discovering Ollama models...")
        discovered = discover_models()
        models = discovered["generators"]
        judge = discovered["judges"][0]

        print("Generators : " + ", ".join(models))
        print("Judge      : " + judge)
        print(f"Test cases : loading from {config.DATASET_PATH.name}")

        dataset = load_dataset()
        prompt = load_prompt()

        print(
            f"Test cases : {len(dataset)} "
            f"(limit={config.TEST_CASE_LIMIT})"
        )
        print(f"Temperature: {config.TEMPERATURE}")

        responses = generate_responses(models, dataset, prompt)
        response_file = save_generator_responses(
            responses, config.RESPONSE_ROOT
        )
        print(f"Generator workbook: {response_file}")

        results = evaluate_generators(
            models, judge, responses, dataset
        )
        verdict = recommend(results["summary"])
        suggestions = improvement_suggestions(results["failures"])

        report_file = save_report(
            build_summary(verdict, suggestions, results["summary"]),
            results["testcase_rows"],
            results["detail_rows"],
            results["failures"],
            build_configuration(discovered),
            config.REPORT_ROOT,
        )

        print("\n" + "=" * 60)
        print("Evaluation Complete")
        print("=" * 60)
        print(f"Final report: {report_file}")
        print(verdict)
        print("=" * 60)

    except Exception as exc:
        print("\n" + "=" * 60)
        print("Evaluation failed")
        print("=" * 60)
        print(f"{type(exc).__name__}: {exc}")
        print("=" * 60)
        raise


if __name__ == "__main__":
    main()
