from deepeval.dataset import EvaluationDataset, Golden
from deepeval.evaluate import AsyncConfig
from deepeval.metrics import (
    PlanAdherenceMetric,
    StepEfficiencyMetric,
    TaskCompletionMetric,
)
from deepeval.models import OllamaModel

from config.settings import OLLAMA_BASE_URL, TRAJECTORY_THRESHOLD
from trajectory.agent import multi_generator_agent


METRIC_NAMES = (
    "task_completion",
    "step_efficiency",
    "plan_adherence",
)


def create_metrics(judge):
    return {
        "task_completion": TaskCompletionMetric(
            threshold=TRAJECTORY_THRESHOLD,
            model=judge,
            async_mode=False,
            include_reason=True,
        ),
        "step_efficiency": StepEfficiencyMetric(
            threshold=TRAJECTORY_THRESHOLD,
            model=judge,
            async_mode=False,
            include_reason=True,
        ),
        "plan_adherence": PlanAdherenceMetric(
            threshold=TRAJECTORY_THRESHOLD,
            model=judge,
            async_mode=False,
            include_reason=True,
        ),
    }


def collect_metric_results(metrics):
    results = {}

    for name, metric in metrics.items():
        score = metric.score
        results[name] = {
            "score": round(float(score), 4) if score is not None else None,
            "passed": bool(metric.is_successful()) if score is not None else False,
            "status": "COMPLETED" if score is not None else "ERROR",
            "reason": metric.reason or "",
            "error": "",
        }

    return results


def run_trajectory(dataset, generators, judge_name, preferred_generator=None):
    judge = OllamaModel(
        model=judge_name,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )
    metrics = create_metrics(judge)

    goldens = [
        Golden(
            input=str(row["Question"]),
            expected_output=str(row["Golden_Answer"]),
            context=[str(row["Source"])],
        )
        for _, row in dataset.iterrows()
    ]
    eval_dataset = EvaluationDataset(goldens=goldens)

    rows = []
    iterator = iter(
        eval_dataset.evals_iterator(
            metrics=list(metrics.values()),
            async_config=AsyncConfig(run_async=False),
        )
    )

    index = 0
    try:
        golden = next(iterator)

        while True:
            row = dataset.iloc[index]

            final, outputs, selected_model = multi_generator_agent(
                golden.input,
                str(row["Source"]),
                str(row["Golden_Answer"]),
                generators,
                preferred_generator=preferred_generator,
            )

            try:
                next_golden = next(iterator)
                metric_results = collect_metric_results(metrics)
            except StopIteration:
                metric_results = collect_metric_results(metrics)
                next_golden = None

            rows.append(
                {
                    "test_id": row["Test_ID"],
                    "question": golden.input,
                    "selected_output": final,
                    "generator_path": " -> ".join(
                        item["model"] for item in outputs
                    ),
                    "generator_count": len(outputs),
                    "preferred_generator_from_project_1": preferred_generator or "",
                    "selected_generator": selected_model,
                    "selection_matches_project_1": bool(
                        preferred_generator
                        and selected_model == preferred_generator
                    ),
                    "task_completion_score": metric_results["task_completion"]["score"],
                    "task_completion_verdict": (
                        "PASS"
                        if metric_results["task_completion"]["passed"]
                        else "FAIL"
                    ),
                    "task_completion_reason": metric_results["task_completion"]["reason"],
                    "step_efficiency_score": metric_results["step_efficiency"]["score"],
                    "step_efficiency_verdict": (
                        "PASS"
                        if metric_results["step_efficiency"]["passed"]
                        else "FAIL"
                    ),
                    "step_efficiency_reason": metric_results["step_efficiency"]["reason"],
                    "plan_adherence_score": metric_results["plan_adherence"]["score"],
                    "plan_adherence_verdict": (
                        "PASS"
                        if metric_results["plan_adherence"]["passed"]
                        else "FAIL"
                    ),
                    "plan_adherence_reason": metric_results["plan_adherence"]["reason"],
                }
            )

            if next_golden is None:
                break

            golden = next_golden
            index += 1

    except StopIteration:
        return rows

    return rows
