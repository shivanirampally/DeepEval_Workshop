from openpyxl import load_workbook

from reporting.excel import save_report


def test_failures_sheet_uses_each_rows_own_metric_threshold(tmp_path):
    # detail_rows ends on a "bias" row so that, under the pre-fix bug, the
    # Failures sheet would reuse "bias" thresholds (pass=0.80, review=0.60)
    # for every row instead of each row's own metric.
    detail_rows = [
        {
            "generator": "gen1", "judge": "judge1", "test_id": "t1",
            "metric": "bias", "score": 0.95, "verdict": "PASS",
            "passed": True, "status": "COMPLETED", "reason": "",
            "error": "", "duration_seconds": 1.0,
        },
    ]

    # Same score, different metrics -> different thresholds ->
    # different expected verdicts:
    #   completeness: pass=0.70 -> 0.72 is PASS
    #   hallucination: pass=0.80, review=0.60 -> 0.72 is REVIEW
    failures = [
        {
            "generator": "gen1", "judge": "judge1", "test_id": "t1",
            "metric": "completeness", "score": 0.72, "verdict": "REVIEW",
            "reason": "reason-completeness",
        },
        {
            "generator": "gen1", "judge": "judge1", "test_id": "t1",
            "metric": "hallucination", "score": 0.72, "verdict": "REVIEW",
            "reason": "reason-hallucination",
        },
    ]

    file_path = save_report(
        summary=[],
        testcase_rows=[],
        detail_rows=detail_rows,
        failures=failures,
        configuration=[],
        report_root=tmp_path,
    )

    workbook = load_workbook(file_path)
    sheet = workbook["Failures"]
    rows = list(sheet.iter_rows(min_row=2, values_only=True))

    by_reason = {row[6]: row for row in rows}

    assert by_reason["reason-completeness"][5] == "PASS"
    assert by_reason["reason-hallucination"][5] == "REVIEW"
