from datetime import datetime
from pathlib import Path

import pandas as pd

import config
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


METRIC_DISPLAY_NAMES = {
    "hallucination": "Hallucination",
    "faithfulness": "Faithfulness",
    "correctness": "Correctness",
    "completeness": "Completeness",
    "answer_relevancy": "Answer Relevancy",
    "bias": "Bias",
}


def _percent(value):
    if value is None or value == "":
        return ""
    return round(float(value) * 100, 1)


def _metric_verdict(score, passed, metric=None):
    if score is None:
        return "ERROR"

    thresholds = config.METRIC_THRESHOLDS.get(
        metric, {"pass": 0.80, "review": 0.60}
    )
    if float(score) >= thresholds["pass"]:
        return "PASS"
    if float(score) >= thresholds["review"]:
        return "REVIEW"
    return "FAIL"


def _metric_interpretation(metric, score, passed):
    if score is None:
        return "Metric evaluation did not complete."

    percent = _percent(score)
    thresholds = config.METRIC_THRESHOLDS.get(
        metric, {"pass": 0.80, "review": 0.60}
    )
    pass_threshold = thresholds["pass"]
    review_threshold = thresholds["review"]

    if passed:
        return (
            f"{percent}% evaluation score - passed the "
            f"{pass_threshold:.2f} quality threshold."
        )

    if float(score) >= review_threshold:
        return (
            f"{percent}% evaluation score - in the REVIEW band "
            f"({review_threshold:.2f} to <{pass_threshold:.2f})."
        )

    return (
        f"{percent}% evaluation score - below the REVIEW band "
        "and marked FAIL."
    )


def _format_sheet(ws):
    ws.freeze_panes = "A2"

    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.fill = PatternFill(fill_type="solid", fgColor="D9EAF7")
        cell.alignment = Alignment(vertical="top", wrap_text=True)

    for row in ws.iter_rows():
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)

    for column_cells in ws.columns:
        width = 12
        for cell in column_cells:
            if cell.value is not None:
                width = max(width, min(len(str(cell.value)) + 2, 45))
        ws.column_dimensions[get_column_letter(column_cells[0].column)].width = width


def _write_dataframe(ws, frame):
    if frame.empty:
        ws.append(["No data"])
        return

    for row in frame.itertuples(index=False, name=None):
        ws.append(list(row))


def save_generator_responses(response_rows, output_root):
    now = datetime.now()
    output_dir = Path(output_root) / now.strftime("%Y-%m-%d")
    output_dir.mkdir(parents=True, exist_ok=True)

    file_path = (
        output_dir
        / f"generator_responses_{now.strftime('%Y%m%d_%H%M%S')}.xlsx"
    )

    with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
        for model, rows in response_rows.items():
            frame = pd.DataFrame(rows)
            sheet_name = str(model).replace(":", "_")[:31]
            frame.to_excel(writer, sheet_name=sheet_name, index=False)

    return file_path


def save_report(
    summary,
    testcase_rows,
    detail_rows,
    failures,
    configuration,
    report_root,
):
    now = datetime.now()
    output_dir = Path(report_root) / now.strftime("%Y-%m-%d")
    output_dir.mkdir(parents=True, exist_ok=True)

    file_path = (
        output_dir
        / f"generator_comparison_{now.strftime('%Y%m%d_%H%M%S')}.xlsx"
    )

    workbook = Workbook()
    workbook.remove(workbook.active)

    # ---------------------------------------------------------
    # Executive Summary
    # ---------------------------------------------------------
    ws = workbook.create_sheet("Executive Summary")

    ws.append(["Section", "Value"])
    for item in summary:
        ws.append([item.get("section", ""), item.get("value", "")])

    ws.append([])
    ws.append(
        [
            "Score interpretation",
            "DeepEval scores are displayed as percentages for readability. "
            "A score is an evaluation score, not a literal probability "
            "that a percentage of the response is hallucinated.",
        ]
    )
    ws.append(
        [
            "Hallucination interpretation",
            "For Hallucination, a higher score represents a better "
            "source-grounded / non-hallucinating result. A 0.80 score is "
            "shown as 80% evaluation score and 20 percentage points below "
            "the ideal 100% score; it is not reported as '20% of the "
            "response is hallucinated'.",
        ]
    )

    _format_sheet(ws)

    # ---------------------------------------------------------
    # Generator Comparison
    # ---------------------------------------------------------
    ws = workbook.create_sheet("Generator Comparison")

    detail_frame = pd.DataFrame(detail_rows)

    generators = []
    if not detail_frame.empty and "generator" in detail_frame:
        generators = list(dict.fromkeys(detail_frame["generator"].tolist()))

    metrics = list(METRIC_DISPLAY_NAMES.keys())

    headers = ["Generator"]
    for metric in metrics:
        headers.extend(
            [
                f"{METRIC_DISPLAY_NAMES[metric]} Score (%)",
                f"{METRIC_DISPLAY_NAMES[metric]} Verdict",
            ]
        )
    headers.extend(
        [
            "Overall Score (%)",
            "Testcases Passed",
            "Testcases Failed",
            "Quality Gate",
        ]
    )
    ws.append(headers)

    summary_by_generator = {
        row.get("generator"): row
        for row in summary
        if row.get("generator")
    }

    for generator in generators:
        row = [generator]
        generator_summary = summary_by_generator.get(generator, {})

        for metric in metrics:
            metric_rows = detail_frame[
                detail_frame["generator"].astype(str) == str(generator)
            ]
            metric_rows = metric_rows[metric_rows["metric"] == metric]

            scores = pd.to_numeric(
                metric_rows["score"], errors="coerce"
            ).dropna()

            score = float(scores.mean()) if not scores.empty else None

            passed_values = metric_rows["passed"].tolist()
            passed = bool(passed_values) and all(
                bool(value) for value in passed_values
            )

            row.extend(
                [
                    _percent(score),
                    _metric_verdict(score, passed, metric),
                ]
            )

        overall = generator_summary.get("overall_score")
        passed_count = generator_summary.get("passed_testcases", "")
        total_count = generator_summary.get("total_testcases", "")
        failed_count = (
            total_count - passed_count
            if isinstance(total_count, (int, float))
            and isinstance(passed_count, (int, float))
            else ""
        )

        row.extend(
            [
                _percent(overall),
                passed_count,
                failed_count,
                generator_summary.get("quality_gate", ""),
            ]
        )

        ws.append(row)

    _format_sheet(ws)

    # ---------------------------------------------------------
    # Testcase Comparison
    # ---------------------------------------------------------
    ws = workbook.create_sheet("Testcase Comparison")

    if testcase_rows:
        testcase_frame = pd.DataFrame(testcase_rows)

        ordered_columns = ["generator", "test_id"]
        for metric in metrics:
            score_column = f"{metric}_score"
            if score_column in testcase_frame.columns:
                ordered_columns.append(score_column)

        if "testcase_verdict" in testcase_frame.columns:
            ordered_columns.append("testcase_verdict")
        elif "is_testcase_passed" in testcase_frame.columns:
            ordered_columns.append("is_testcase_passed")

        testcase_frame = testcase_frame[
            [column for column in ordered_columns if column in testcase_frame]
        ].copy()

        rename_map = {
            "generator": "Generator",
            "test_id": "Test Case",
            "testcase_verdict": "Testcase Gate",
            "is_testcase_passed": "Testcase Gate",
        }

        for metric in metrics:
            rename_map[f"{metric}_score"] = (
                f"{METRIC_DISPLAY_NAMES[metric]} Score (%)"
            )

        testcase_frame.rename(columns=rename_map, inplace=True)

        for column in testcase_frame.columns:
            if column.endswith("Score (%)"):
                testcase_frame[column] = testcase_frame[column].apply(_percent)

        if "Testcase Gate" in testcase_frame.columns and testcase_frame["Testcase Gate"].dtype == bool:
            testcase_frame["Testcase Gate"] = testcase_frame[
                "Testcase Gate"
            ].map({True: "PASS", False: "FAIL"})

        _write_dataframe(ws, testcase_frame)
    else:
        ws.append(["No testcase results"])

    _format_sheet(ws)

    # ---------------------------------------------------------
    # Detailed Metric Reasons
    # ---------------------------------------------------------
    ws = workbook.create_sheet("Detailed Metric Reasons")

    detail_headers = [
        "Generator",
        "Judge",
        "Test Case",
        "Metric",
        "Score (%)",
        "Metric Verdict",
        "Passed Quality Threshold",
        "Gap to Ideal (%)",
        "Reason",
        "Status",
        "Error",
        "Interpretation",
    ]
    ws.append(detail_headers)

    for item in detail_rows:
        metric = str(item.get("metric", "")).lower()
        score = item.get("score")
        passed = bool(item.get("passed", False))

        if score is not None:
            gap_to_ideal = round((1 - float(score)) * 100, 1)
        else:
            gap_to_ideal = ""

        ws.append(
            [
                item.get("generator", ""),
                item.get("judge", ""),
                item.get("test_id", ""),
                METRIC_DISPLAY_NAMES.get(metric, item.get("metric", "")),
                _percent(score),
                _metric_verdict(score, passed, metric),
                "PASS" if passed else "FAIL",
                gap_to_ideal,
                item.get("reason", ""),
                item.get("status", ""),
                item.get("error", ""),
                _metric_interpretation(metric, score, passed),
            ]
        )

    _format_sheet(ws)

    # ---------------------------------------------------------
    # Failures
    # ---------------------------------------------------------
    ws = workbook.create_sheet("Failures")

    failure_headers = [
        "Generator",
        "Judge",
        "Test Case",
        "Metric",
        "Score (%)",
        "Verdict",
        "Reason",
    ]
    ws.append(failure_headers)

    for item in failures:
        score = item.get("score")
        passed = False

        ws.append(
            [
                item.get("generator", ""),
                item.get("judge", ""),
                item.get("test_id", ""),
                METRIC_DISPLAY_NAMES.get(
                    str(item.get("metric", "")).lower(),
                    item.get("metric", ""),
                ),
                _percent(score),
                _metric_verdict(score, passed, metric),
                item.get("reason", ""),
            ]
        )

    _format_sheet(ws)

    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------
    ws = workbook.create_sheet("Configuration")
    ws.append(["Configuration", "Value"])

    for item in configuration:
        ws.append([item.get("key", ""), item.get("value", "")])

    ws.append([])
    ws.append(
        [
            "Reporting rule",
            "Individual metric scores are displayed as percentages. "
            "Each metric uses its configured PASS and REVIEW thresholds.",
        ]
    )
    ws.append(
        [
            "Testcase rule",
            "Overall weighted score >= 0.80 is PASS; 0.60 to <0.80 is "
            "REVIEW; below 0.60 is FAIL. Hallucination, Faithfulness and "
            "Correctness must not fall below their REVIEW thresholds.",
        ]
    )

    _format_sheet(ws)


    workbook.save(file_path)

    return file_path
