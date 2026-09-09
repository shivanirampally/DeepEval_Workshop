from datetime import datetime
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font

from analysis.failure_analysis import verdict
from config import (
    GENERATOR_MODEL,
    JUDGE_MODEL,
    METRIC_WEIGHTS,
    QUALITY_THRESHOLD,
    REVIEW_THRESHOLD,
)
from prompts import BASE_PROMPT, IMPROVED_PROMPT

def _pct(value):
    return "" if pd.isna(value) else round(float(value) * 100, 1)

def save_baseline_report(details, dataset, failures, recommendations):
    report_dir = Path("reports") / datetime.now().strftime("%Y-%m-%d")
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"v1_baseline_evaluation_{datetime.now():%H%M%S}.xlsx"

    detailed = details.copy()
    detailed["Score (%)"] = detailed["score"].apply(_pct)
    detailed["Verdict"] = detailed["score"].apply(verdict)

    testcase = (
        detailed.groupby("test_id", as_index=False)
        .agg(
            Overall_Score=("score", "mean"),
            Metrics_Passed=("passed", "sum"),
            Metrics_Evaluated=("passed", "count"),
        )
    )
    testcase["Overall Score (%)"] = testcase["Overall_Score"].apply(_pct)
    testcase["Verdict"] = testcase.apply(
        lambda row: "PASS"
        if row["Metrics_Passed"] == row["Metrics_Evaluated"]
        and row["Overall_Score"] >= QUALITY_THRESHOLD
        else "FAIL",
        axis=1,
    )

    metric_summary = (
        detailed.groupby("metric", as_index=False)
        .agg(
            Average_Score=("score", "mean"),
            Passed=("passed", "sum"),
            Evaluated=("passed", "count"),
        )
    )
    metric_summary["Average Score (%)"] = metric_summary["Average_Score"].apply(_pct)
    metric_summary["Verdict"] = metric_summary["Average_Score"].apply(verdict)

    total = len(testcase)
    passed = int((testcase["Verdict"] == "PASS").sum())
    overall = float(detailed["score"].dropna().mean()) if detailed["score"].notna().any() else 0

    summary = pd.DataFrame([
        ["Objective", "Run the V1 baseline, identify failure patterns, and use evidence to improve the prompt."],
        ["Input dataset", "Project 1 hallucination benchmark (same file, unchanged)."],
        ["Test cases", total],
        ["Testcases passed", f"{passed}/{total}"],
        ["Overall evaluation score", f"{overall * 100:.1f}%"],
        ["Generator", GENERATOR_MODEL],
        ["Independent judge", JUDGE_MODEL],
        ["Metric gate", f"{QUALITY_THRESHOLD:.0%} = PASS; {REVIEW_THRESHOLD:.0%} = REVIEW boundary"],
        ["Next step", "Use failure analysis to design targeted synthetic test cases and improve the prompt."],
    ], columns=["Item", "Value"])

    failures_df = pd.DataFrame(
        failures,
        columns=[
            "Test Case", "Metric", "Score (%)",
            "Verdict", "Reason", "Recommended Action",
        ],
    )

    recommendations_df = pd.DataFrame(recommendations)

    prompt_df = pd.DataFrame([
        ["Baseline prompt", BASE_PROMPT],
        ["Improved prompt", IMPROVED_PROMPT],
        ["Implementation note",
         "The improved prompt is derived from baseline failure patterns. It is documented as an improvement "
         "implementation, not as a quantitatively validated result until it is rerun."],
    ], columns=["Item", "Content"])

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Executive Summary", index=False)
        testcase[
            ["test_id", "Overall Score (%)", "Metrics_Passed", "Metrics_Evaluated", "Verdict"]
        ].to_excel(writer, sheet_name="Testcase Results", index=False)
        metric_summary[
            ["metric", "Average Score (%)", "Passed", "Evaluated", "Verdict"]
        ].to_excel(writer, sheet_name="Metric Summary", index=False)
        detailed[
            ["test_id", "metric", "Score (%)", "Verdict", "reason", "status", "error", "response"]
        ].to_excel(writer, sheet_name="Detailed Results", index=False)
        failures_df.to_excel(writer, sheet_name="Failure Analysis", index=False)
        recommendations_df.to_excel(writer, sheet_name="Recommendations", index=False)
        prompt_df.to_excel(writer, sheet_name="Prompt Improvement", index=False)
        dataset.to_excel(writer, sheet_name="Baseline Input Data", index=False)

    _format(path)
    return path

def _format(path):
    workbook = load_workbook(path)
    for sheet in workbook.worksheets:
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        for cell in sheet[1]:
            cell.font = Font(bold=True)
        for row in sheet.iter_rows():
            for cell in row:
                cell.alignment = Alignment(vertical="top", wrap_text=True)
        for column in sheet.columns:
            width = min(max(max(len(str(c.value or "")) for c in column) + 2, 12), 55)
            sheet.column_dimensions[column[0].column_letter].width = width
    workbook.save(path)
