from pathlib import Path

import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font

SHEET_ORDER = [
    "01_Run_Summary", "02_V1_Evaluation", "03_Failure_Analysis",
    "04_Human_Approval", "05_Synthetic_Data", "06_Synthetic_V1",
    "07_Prompt_Versions", "08_V2_Evaluation", "09_Synthetic_V2",
    "10_V1_vs_V2", "11_Audit_Log", "12_LLM_Call_Timings",
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

    # Reorder tabs to match the lifecycle stage order regardless of the order
    # sheets happened to be written in. Any sheet name outside SHEET_ORDER
    # would silently vanish here, but the app only ever writes these eleven.
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


def _append_rows(path, sheet_name, rows):
    """Append one or more rows to a sheet in a single workbook open/save.

    Approvals, audit events and timing rows can happen many times per run, and
    the old approach (read the sheet into pandas, concat, then hand the whole
    thing to write_sheet -- which deletes and recreates the sheet and runs a
    full workbook reformat) meant every single append reopened, rewrote and
    reformatted the *entire* workbook. This appends in place instead: open
    once, create the sheet with a header row if it's new, append the given
    rows, save once. No separate "does this sheet exist" probe is needed
    either, since there's only one open here to check against.
    """
    if not rows:
        return
    path = Path(path)
    try:
        wb = load_workbook(path) if path.exists() else Workbook()
    except PermissionError as exc:
        raise PermissionError(
            f"Workbook is locked: {path}. Close it in Excel and retry the lifecycle."
        ) from exc
    if not path.exists():
        wb.remove(wb.active)

    if sheet_name not in wb.sheetnames:
        ws = wb.create_sheet(sheet_name)
        ws.append(list(rows[0].keys()))
        for cell in ws[1]:
            cell.font = Font(bold=True)
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws = wb[sheet_name]
    headers = [cell.value for cell in ws[1]]
    for row in rows:
        ws.append([row.get(header, "") for header in headers])
        for cell in ws[ws.max_row]:
            cell.alignment = Alignment(vertical="top", wrap_text=True)

    try:
        wb.save(path)
    except PermissionError as exc:
        raise PermissionError(
            f"Workbook is locked: {path}. Close it in Excel and retry the lifecycle."
        ) from exc


def _append_row(path, sheet_name, row):
    _append_rows(path, sheet_name, [row])


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


def parse_excel_bool(value):
    """Normalize a 'passed'-style cell to a real bool.

    Excel round-trips should already hand back a native bool, but a manually
    edited cell can come back as the literal text "TRUE"/"FALSE"/"yes" -- and
    plain bool("FALSE") is True (any non-empty string is truthy), which would
    silently flip a failed row to "passed". Check the actual text instead.

    Public (not `_`-prefixed) because evaluation/failure_analysis.py needs the
    exact same normalization when it re-reads a saved evaluation sheet on
    --resume -- one place to fix this bug in, not two.
    """
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().upper() in {"TRUE", "1", "YES", "PASS"}


def _testcase_pass_rate(frame):
    if frame is None or frame.empty:
        return 0.0, 0, 0
    working = frame.copy()
    working["passed"] = working["passed"].map(parse_excel_bool)
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
        # +/-0.001 is just noise-filtering for float rounding, not a real threshold.
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
    # One-glance signal on top of the per-metric Regression Flag column above:
    # did the overall testcase pass rate get worse, better, or stay the same.
    comparison["V1 vs V2 Pass-Rate Change"] = v2_rate - v1_rate
    comparison["Overall Regression"] = "YES" if v2_rate < v1_rate else "NO"

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


def append_llm_call_timings(path, rows):
    """Append many LLM call timing rows in one workbook open/save.

    A stage with a generation call plus six metrics is six-plus rows; batching
    them here instead of calling this once per row is what keeps a run's
    timing bookkeeping from becoming its own bottleneck as testcases scale up.
    Each row: {"Stage": ..., "Call": ..., "Model": ..., "Time Taken (s)": ...}
    """
    _append_rows(path, "12_LLM_Call_Timings", rows)


def append_llm_call_timing(path, stage, call, model, seconds):
    """Single-row convenience wrapper around append_llm_call_timings."""
    append_llm_call_timings(path, [{
        "Stage": stage,
        "Call": call,
        "Model": model,
        "Time Taken (s)": seconds,
    }])


def format_workbook(path):
    """Public entry point for a one-off full formatting/reordering pass.

    The lightweight appends above skip this on every call for speed, so the
    caller runs it once at the end of a completed lifecycle to make sure the
    workbook a human actually opens is fully polished (freeze panes, column
    widths, tab order) even though intermediate audit/timing rows weren't.
    """
    _format_workbook(path)
