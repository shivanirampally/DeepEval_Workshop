from pathlib import Path

import pandas as pd
import pytest

from config import SYNTHETIC_TEST_CASE_COUNT
from evaluation.failure_analysis import analyze
from execution.main import main_resume, save_state
from governance_observability.excel_reporting import write_sheet
from testdata.synthetic_data import validate_human_approved_synthetic_dataset


def test_technical_error_is_not_prompt_failure():
    details = pd.DataFrame([{
        "test_id": "T001", "metric": "hallucination", "score": None,
        "passed": False, "status": "ERROR", "reason": "", "error": "timeout",
    }])
    failures, _ = analyze(details)
    assert failures.empty
    assert failures.attrs["technical_errors"]


def test_technical_error_does_not_mask_a_real_failure():
    details = pd.DataFrame([
        {"test_id": "T001", "metric": "hallucination", "score": 0.5, "passed": False,
         "status": "COMPLETED", "reason": "bad", "error": ""},
        {"test_id": "T002", "metric": "bias", "score": None, "passed": False,
         "status": "ERROR", "reason": "", "error": "timeout"},
    ])
    failures, _ = analyze(details)
    assert len(failures) == 1
    assert failures.iloc[0]["test_id"] == "T001"
    assert len(failures.attrs["technical_errors"]) == 1


def test_human_approval_is_required():
    # validate_human_approved_synthetic_dataset checks the row count against the
    # live SYNTHETIC_TEST_CASE_COUNT config value first -- build a frame with
    # exactly that many rows rather than a hardcoded 1, otherwise this test
    # silently depends on whatever config.py's value happens to be at run time
    # (see the same fix already applied in test_scalability.py).
    frame = pd.DataFrame([{
        "Test_ID": f"SYN{i:03d}", "Source": "S", "Question": "Q",
        "Golden_Answer": "A", "Human_Reviewed": "PENDING",
    } for i in range(1, SYNTHETIC_TEST_CASE_COUNT + 1)])
    with pytest.raises(ValueError):
        validate_human_approved_synthetic_dataset(frame)
    frame["Human_Reviewed"] = "APPROVED"
    assert len(validate_human_approved_synthetic_dataset(frame)) == SYNTHETIC_TEST_CASE_COUNT


def test_resume_cannot_bypass_a_human_rejection(tmp_path):
    run_dir = tmp_path / "run_rejected"
    run_dir.mkdir()
    workbook = run_dir / "AI_validation_run_test.xlsx"
    write_sheet(workbook, "02_V1_Evaluation", pd.DataFrame([{"test_id": "T001"}]))
    # Mirrors what a real "N" answer at a human gate persists: status STOPPED,
    # next_stage pointing at whatever comes after the gate.
    save_state(run_dir, "run_test", "HUMAN_APPROVAL_FAILURE_ANALYSIS", "SYNTHETIC_DATA", "STOPPED")
    with pytest.raises(RuntimeError):
        main_resume(run_dir)
