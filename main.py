from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd

from analysis.recommendation import improvement_suggestions, recommend
import config
from config import discover_models
from deepeval_framework.evaluation import evaluate_one
from deepeval_framework.scoring import is_testcase_passed, weighted_score
from generators.ollama_client import OllamaClient
from reporting.excel import save_generator_responses, save_report


def load_dataset():
    return pd.read_excel(config.DATASET_PATH, sheet_name="Test_Data")


def load_prompt():
    prompts = pd.read_excel(
        config.DATASET_PATH,
        sheet_name="Prompt_Versions",
    )
    row = prompts.iloc[0]
    return str(row["Prompt"])


def build_prompt(template, row):
    return template.format(
        source=str(row["Source"]),
        question=str(row["Question"]),
    )


def generate_for_model(model, dataset, prompt_template):
    client = OllamaClient(
        config.OLLAMA_BASE_URL,
        config.REQUEST_TIMEOUT,
        config.RETRIES,
        config.TEMPERATURE,
    )
    rows = []

    for _, row in dataset.iterrows():
        prompt = build_prompt(prompt_template, row)
        result = client.generate(model, prompt)
        rows.append({
            "test_id": row["Test_ID"],
            "model": model,
            "source": row["Source"],
            "question": row["Question"],
            "prompt": prompt,
            "response": result["response"],
            "status": result["status"],
            "error": result["error"],
            "duration_seconds": result["duration_seconds"],
        })

    return rows


def main():
    print("Discovering models from Ollama /api/tags...")
    discovered = discover_models()

    print("Generators:", ", ".join(discovered["generators"]))
    print("Judges:", ", ".join(discovered["judges"]))

    dataset = load_dataset()
    prompt = load_prompt()

    response_rows = {}
    with ThreadPoolExecutor(
        max_workers=min(config.GENERATOR_CONCURRENCY, len(discovered["generators"]))
    ) as pool:
        futures = {
            pool.submit(generate_for_model, model, dataset, prompt): model
            for model in discovered["generators"]
        }
        for future in as_completed(futures):
            model = futures[future]
            response_rows[model] = future.result()

    response_file = save_generator_responses(
        response_rows,
        config.RESPONSE_ROOT,
    )
    print(f"Generator responses saved: {response_file}")

    source_by_id = {
        str(row["Test_ID"]): row
        for _, row in dataset.iterrows()
    }

    detail_rows = []
    failures = []
    generator_summary = []

    for model in discovered["generators"]:
        model_rows = response_rows[model]
        all_testcase_results = []

        for response_row in model_rows:
            testcase_id = str(response_row["test_id"])
            source_row = source_by_id[testcase_id]

            judge_results = []
            for judge in discovered["judges"]:
                metrics = evaluate_one(
                    response_row,
                    source_row,
                    judge,
                    config.OLLAMA_BASE_URL,
                )
                score = weighted_score(metrics)
                passed = is_testcase_passed(metrics)
                judge_results.append({
                    "judge": judge,
                    "score": score,
                    "passed": passed,
                    "metrics": metrics,
                })

                for metric_name, metric_result in metrics.items():
                    detail_rows.append({
                        "generator": model,
                        "judge": judge,
                        "test_id": testcase_id,
                        "metric": metric_name,
                        "score": metric_result["score"],
                        "passed": metric_result["passed"],
                        "status": metric_result["status"],
                        "reason": metric_result["reason"],
                        "error": metric_result["error"],
                    })
                    if metric_result["status"] != "COMPLETED" or not metric_result["passed"]:
                        failures.append({
                            "generator": model,
                            "judge": judge,
                            "test_id": testcase_id,
                            "metric": metric_name,
                            "score": metric_result["score"],
                            "reason": metric_result["reason"],
                        })

            testcase_score = (
                sum(item["score"] for item in judge_results if item["score"] is not None)
                / len([item for item in judge_results if item["score"] is not None])
                if any(item["score"] is not None for item in judge_results)
                else None
            )
            testcase_pass = all(item["passed"] for item in judge_results)
            all_testcase_results.append({
                "test_id": testcase_id,
                "score": testcase_score,
                "passed": testcase_pass,
                "judge_scores": [item["score"] for item in judge_results],
            })

        passed_count = sum(item["passed"] for item in all_testcase_results)
        total = len(all_testcase_results)
        overall = (
            sum(item["score"] for item in all_testcase_results if item["score"] is not None)
            / len([item for item in all_testcase_results if item["score"] is not None])
            if any(item["score"] is not None for item in all_testcase_results)
            else None
        )

        agreement_values = []
        for testcase in all_testcase_results:
            scores = testcase.get("judge_scores", [])
            if len(scores) >= 2 and all(score is not None for score in scores):
                agreement_values.append(1 - abs(scores[0] - scores[1]))

        metric_scores = []
        for name in config.METRIC_NAMES:
            values = [
                r["score"] for r in detail_rows
                if r["generator"] == model and r["score"] is not None and r["metric"] == name
            ]
            if values:
                metric_scores.append((name, sum(values) / len(values)))

        metric_map = dict(metric_scores)
        generator_summary.append({
            "generator": model,
            "overall_score": round(overall, 4) if overall is not None else None,
            "passed_testcases": passed_count,
            "total_testcases": total,
            "quality_gate": "PASS" if passed_count == total and total else "FAIL",
            "judge_agreement": round(sum(agreement_values) / len(agreement_values), 4) if agreement_values else None,
            "metric_scores": metric_map,
        })

    verdict = recommend(generator_summary)
    suggestions = improvement_suggestions(failures)

    judge_rows = []
    detail_frame = pd.DataFrame(detail_rows)
    if not detail_frame.empty:
        grouped = detail_frame.dropna(subset=["score"]).groupby(
            ["generator", "judge", "metric"]
        )["score"].mean().reset_index()
        for row in grouped.to_dict("records"):
            judge_rows.append({
                "generator": row["generator"],
                "judge": row["judge"],
                "metric": row["metric"],
                "average_score": round(float(row["score"]), 4),
            })

    summary = [
        {"section": "Final Verdict", "value": verdict},
        {
            "section": "Testcase Gate",
            "value": f"Every configured metric must be >= {config.QUALITY_THRESHOLD:.2f} for a testcase to PASS.",
        },
        {
            "section": "Review Band",
            "value": f"Scores from {config.WARNING_THRESHOLD:.2f} to below {config.QUALITY_THRESHOLD:.2f} require review.",
        },
        {
            "section": "Metric Roles",
            "value": " | ".join(f"{name}: {role}" for name, role in config.METRIC_ROLES.items()),
        },
        {
            "section": "Selection Logic",
            "value": "Quality gate first, then weighted quality score, testcase pass count, grounding metrics, and judge agreement. A close result is reported as no clear winner.",
        },
        {
            "section": "Prompt/Test Data Improvements",
            "value": " | ".join(suggestions) if suggestions else "No repeated failure pattern was detected.",
        },
    ]
    for row in generator_summary:
        for metric_name, metric_score in row.pop("metric_scores").items():
            row[f"{metric_name}_score"] = round(metric_score, 4)
        summary.append(row)

    configuration = [
        {"key": "Ollama Base URL", "value": config.OLLAMA_BASE_URL},
        {"key": "Temperature", "value": config.TEMPERATURE},
        {"key": "Generators", "value": ", ".join(discovered["generators"])},
        {"key": "Judges", "value": ", ".join(discovered["judges"])},
        {"key": "Generator Concurrency", "value": config.GENERATOR_CONCURRENCY},
        {"key": "Judge/Metric Concurrency", "value": config.JUDGE_CONCURRENCY},
        {"key": "Dataset", "value": str(config.DATASET_PATH)},
        {"key": "Metric Roles", "value": " | ".join(f"{name}: {role}" for name, role in config.METRIC_ROLES.items())},
    ]

    detail_frame = pd.DataFrame(detail_rows)
    testcase_rows = []
    if not detail_frame.empty:
        for (generator, test_id), group in detail_frame.groupby(["generator", "test_id"]):
            row = {"generator": generator, "test_id": test_id}
            for metric_name in config.METRIC_NAMES:
                values = group.loc[group["metric"] == metric_name, "score"].dropna()
                row[f"{metric_name}_score"] = round(float(values.mean()), 4) if not values.empty else None
            row["is_testcase_passed"] = bool(
                group["passed"].fillna(False).all()
                and (group["status"] == "COMPLETED").all()
            )
            testcase_rows.append(row)

    report_file = save_report(
        summary,
        testcase_rows,
        detail_rows,
        failures,
        configuration,
        judge_rows,
        config.REPORT_ROOT,
    )

    print(f"Final report saved: {report_file}")
    print(verdict)


if __name__ == "__main__":
    main()
