from pathlib import Path

import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font

SHEET_ORDER = [
    "01_Run_Summary", "02_V1_Evaluation", "03_Failure_Analysis",
    "04_Human_Approval", "05_Synthetic_Data", "06_Synthetic_V1",
    "07_Prompt_Versions", "08_V2_Evaluation", "09_Synthetic_V2",
    "10_V1_vs_V2", "11_Audit_Log",
]


def _format_workbook(path):
    wb = load_workbook(path)
    for ws in wb.worksheets:
        ws.freeze_panes = "A2"
        if ws.max_row and ws.max_column:
            ws.auto_filter.ref = ws.dimensions
        for cell in ws[1]:
            cell.font = Font(bold=True)
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        for row in ws.iter_rows():
            for cell in row:
                cell.alignment = Alignment(vertical="top", wrap_text=True)
        for column_cells in ws.columns:
            values = [len(str(c.value)) for c in column_cells if c.value is not None]
            if values:
                ws.column_dimensions[column_cells[0].column_letter].width = min(max(max(values) + 2, 12), 60)

    existing = {ws.title: ws for ws in wb.worksheets}
    wb._sheets = [existing[name] for name in SHEET_ORDER if name in existing]
    wb.save(path)


def _write_with_openpyxl(path, sheet_name, frame):
    path = Path(path)
    if path.exists():
        try:
            wb = load_workbook(path)
        except PermissionError as exc:
            raise PermissionError(
                f"Workbook is locked: {path}. Close it in Excel and retry the lifecycle."
            ) from exc
    else:
        wb = Workbook()
        wb.remove(wb.active)

    if sheet_name in wb.sheetnames:
        del wb[sheet_name]
    ws = wb.create_sheet(sheet_name)

    if frame.empty:
        ws.append(["No data"])
    else:
        ws.append(list(frame.columns))
        for row in frame.itertuples(index=False, name=None):
            ws.append(list(row))

    try:
        wb.save(path)
    except PermissionError as exc:
        raise PermissionError(
            f"Workbook is locked: {path}. Close it in Excel and retry the lifecycle."
        ) from exc

    _format_workbook(path)


def write_sheet(path, sheet_name, data):
    if isinstance(data, pd.DataFrame):
        frame = data.copy()
    elif isinstance(data, list):
        frame = pd.DataFrame(data)
    else:
        frame = pd.DataFrame([data])
    _write_with_openpyxl(path, sheet_name, frame)


def read_sheet(path, sheet_name):
    return pd.read_excel(path, sheet_name=sheet_name)


def save_summary(path, rows):
    write_sheet(path, "01_Run_Summary", rows)


def save_evaluation(path, sheet_name, details):
    write_sheet(path, sheet_name, details)


def save_failure_analysis(path, failures, recommendations):
    write_sheet(path, "03_Failure_Analysis", failures)
    wb = load_workbook(path)
    ws = wb["03_Failure_Analysis"]
    start = ws.max_row + 3
    ws.cell(start, 1, "Recommendations")
    ws.cell(start, 1).font = Font(bold=True)
    rec = pd.DataFrame(recommendations)
    if rec.empty:
        rec = pd.DataFrame([{
            "Metric": "", "Failure Count": 0,
            "Recommended Action": "No recurring failure pattern detected.",
        }])
    for j, col in enumerate(rec.columns, 1):
        ws.cell(start + 1, j, col).font = Font(bold=True)
    for i, values in enumerate(rec.itertuples(index=False, name=None), start + 2):
        for j, value in enumerate(values, 1):
            ws.cell(i, j, value)
    wb.save(path)
    _format_workbook(path)


def _append_row(path, sheet_name, row):
    if Path(path).exists() and sheet_name in load_workbook(path, read_only=True).sheetnames:
        old = pd.read_excel(path, sheet_name=sheet_name)
        frame = pd.concat([old, pd.DataFrame([row])], ignore_index=True)
    else:
        frame = pd.DataFrame([row])
    write_sheet(path, sheet_name, frame)


def save_approval(path, stage, status, decision, note):
    _append_row(path, "04_Human_Approval", {
        "Stage": stage, "Status": status, "Decision": decision, "Note": note,
    })


def save_prompt_version(path, version, prompt_text, status):
    _append_row(path, "07_Prompt_Versions", {
        "Version": version, "Status": status, "Prompt": prompt_text,
    })


def _summary(frame, score_name):
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["metric", score_name])
    working = frame.copy()
    working["score"] = pd.to_numeric(working["score"], errors="coerce")
    return working.groupby("metric", dropna=False)["score"].mean().reset_index().rename(columns={"score": score_name})


def _testcase_pass_rate(frame):
    if frame is None or frame.empty:
        return 0.0, 0, 0
    working = frame.copy()
    working["passed"] = working["passed"].fillna(False).astype(bool)
    grouped = working.groupby("test_id")["passed"].all()
    total = len(grouped)
    passed = int(grouped.sum())
    return (passed / total if total else 0.0), passed, total


def save_comparison(path, v1, v2, synthetic_v1=None, synthetic_v2=None):
    a = _summary(v1, "V1 Score")
    b = _summary(v2, "V2 Score")
    comparison = pd.merge(a, b, on="metric", how="outer")
    if not comparison.empty:
        comparison["Change"] = comparison["V2 Score"] - comparison["V1 Score"]
        comparison["Result"] = comparison["Change"].apply(
            lambda x: "Improved" if x > 0.001 else "Declined" if x < -0.001 else "No material change"
        )

    v1_rate, v1_passed, v1_total = _testcase_pass_rate(v1)
    v2_rate, v2_passed, v2_total = _testcase_pass_rate(v2)
    if comparison.empty:
        comparison = pd.DataFrame([{"metric": "No metric results"}])

    comparison["V1 Testcase Pass Rate"] = v1_rate
    comparison["V2 Testcase Pass Rate"] = v2_rate
    comparison["V1 Passed Testcases"] = v1_passed
    comparison["V2 Passed Testcases"] = v2_passed
    comparison["Regression Flag"] = comparison.apply(
        lambda r: "YES" if r.get("Change") is not None and pd.notna(r.get("Change")) and float(r["Change"]) < -0.001 else "NO",
        axis=1,
    )

    _append_row(path, "10_V1_vs_V2", {"Comparison": "Original Dataset", "V1 Testcase Pass Rate": v1_rate, "V2 Testcase Pass Rate": v2_rate,
                                      "V1 Passed Testcases": v1_passed, "V2 Passed Testcases": v2_passed,
                                      "Overall Decision": "REVIEW V2 CANDIDATE"})
    write_sheet(path, "10_V1_vs_V2", comparison)

    wb = load_workbook(path)
    ws = wb["10_V1_vs_V2"]
    start = ws.max_row + 3
    ws.cell(start, 1, "Synthetic Dataset")
    ws.cell(start, 1).font = Font(bold=True)
    syn = pd.merge(_summary(synthetic_v1, "V1 Score"), _summary(synthetic_v2, "V2 Score"), on="metric", how="outer")
    if syn.empty:
        ws.cell(start + 1, 1, "No synthetic comparison data available")
    else:
        syn["Change"] = syn["V2 Score"] - syn["V1 Score"]
        syn["Result"] = syn["Change"].apply(
            lambda x: "Improved" if x > 0.001 else "Declined" if x < -0.001 else "No material change"
        )
        for j, col in enumerate(syn.columns, 1):
            ws.cell(start + 1, j, col).font = Font(bold=True)
        for i, values in enumerate(syn.itertuples(index=False, name=None), start + 2):
            for j, value in enumerate(values, 1):
                ws.cell(i, j, value)
    wb.save(path)
    _format_workbook(path)


def append_audit(path, timestamp, stage, action, status, details=""):
    _append_row(path, "11_Audit_Log", {
        "Timestamp": timestamp,
        "Stage": stage,
        "Action": action,
        "Status": status,
        "Details": details,
    })
