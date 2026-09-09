from datetime import datetime
from pathlib import Path

import pandas as pd


def save(rows, root):
    directory = Path(root) / datetime.now().strftime("%Y-%m-%d")
    directory.mkdir(parents=True, exist_ok=True)

    path = directory / (
        f"trajectory_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    )

    results = pd.DataFrame(rows)

    summary_rows = [
        {"section": "Objective", "value": "Validate the Project 1 multi-generator selection trajectory."},
        {"section": "Generator Path", "value": "llama3:instruct -> qwen2.5-coder:14b -> sqlcoder:15b"},
        {"section": "Trajectory Judge", "value": "gpt-oss:20b (preferred)"},
        {"section": "Trajectory Metrics", "value": "Task Completion, Step Efficiency, Plan Adherence"},
        {"section": "Project 1 Alignment", "value": "Selected generator is compared with the latest Project 1 preferred generator."},
        {"section": "Test Cases", "value": len(results)},
    ]

    if not results.empty:
        summary_rows.extend(
            [
                {
                    "section": "Project 1 Selection Matches",
                    "value": int(results["selection_matches_project_1"].sum()),
                },
                {
                    "section": "Task Completion Pass Rate",
                    "value": f"{results['task_completion_verdict'].eq('PASS').mean():.0%}",
                },
                {
                    "section": "Step Efficiency Pass Rate",
                    "value": f"{results['step_efficiency_verdict'].eq('PASS').mean():.0%}",
                },
                {
                    "section": "Plan Adherence Pass Rate",
                    "value": f"{results['plan_adherence_verdict'].eq('PASS').mean():.0%}",
                },
            ]
        )

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(summary_rows).to_excel(
            writer, sheet_name="Executive Summary", index=False
        )

        if results.empty:
            results.to_excel(writer, sheet_name="Testcase Results", index=False)
        else:
            results.to_excel(writer, sheet_name="Testcase Results", index=False)

            alignment = results[
                [
                    "test_id",
                    "preferred_generator_from_project_1",
                    "selected_generator",
                    "selection_matches_project_1",
                    "generator_path",
                    "generator_count",
                ]
            ]
            alignment.to_excel(
                writer, sheet_name="Project 1 Alignment", index=False
            )

            metric_columns = [
                "test_id",
                "task_completion_score",
                "task_completion_verdict",
                "task_completion_reason",
                "step_efficiency_score",
                "step_efficiency_verdict",
                "step_efficiency_reason",
                "plan_adherence_score",
                "plan_adherence_verdict",
                "plan_adherence_reason",
            ]
            results[metric_columns].to_excel(
                writer, sheet_name="Trajectory Metrics", index=False
            )

        configuration = pd.DataFrame(
            [
                {"key": "Generators", "value": "llama3:instruct, qwen2.5-coder:14b, sqlcoder:15b"},
                {"key": "Judge", "value": "gpt-oss:20b"},
                {"key": "Threshold", "value": 0.70},
                {"key": "Project 1 Source", "value": "Latest generator_comparison_*.xlsx report"},
                {"key": "Dataset", "value": "Same Project 1 hallucination benchmark"},
            ]
        )
        configuration.to_excel(
            writer, sheet_name="Configuration", index=False
        )

    return path
