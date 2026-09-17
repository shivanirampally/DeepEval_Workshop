import pandas as pd
import pytest

from testdata import loader


def _write_dataset(path, rows):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name="Test_Data", index=False)


def _patch(monkeypatch, path, limit=1):
    monkeypatch.setattr(loader.config, "DATASET_PATH", path)
    monkeypatch.setattr(loader.config, "TEST_CASE_LIMIT", limit)


def test_load_dataset_accepts_well_formed_rows(tmp_path, monkeypatch):
    path = tmp_path / "data.xlsx"
    _write_dataset(path, {
        "Test_ID": ["T1"], "Source": ["s"],
        "Question": ["q"], "Golden_Answer": ["a"],
    })
    _patch(monkeypatch, path)

    dataset = loader.load_dataset()

    assert len(dataset) == 1


def test_load_dataset_rejects_blank_test_id(tmp_path, monkeypatch):
    path = tmp_path / "data.xlsx"
    _write_dataset(path, {
        "Test_ID": [""], "Source": ["s"],
        "Question": ["q"], "Golden_Answer": ["a"],
    })
    _patch(monkeypatch, path)

    with pytest.raises(ValueError, match="blank Test_ID"):
        loader.load_dataset()


def test_load_dataset_rejects_duplicate_test_id(tmp_path, monkeypatch):
    path = tmp_path / "data.xlsx"
    _write_dataset(path, {
        "Test_ID": ["T1", "T1"], "Source": ["s1", "s2"],
        "Question": ["q1", "q2"], "Golden_Answer": ["a1", "a2"],
    })
    _patch(monkeypatch, path, limit=2)

    with pytest.raises(ValueError, match="duplicate Test_ID"):
        loader.load_dataset()


def test_load_dataset_rejects_blank_required_field(tmp_path, monkeypatch):
    path = tmp_path / "data.xlsx"
    _write_dataset(path, {
        "Test_ID": ["T1"], "Source": [None],
        "Question": ["q"], "Golden_Answer": ["a"],
    })
    _patch(monkeypatch, path)

    with pytest.raises(ValueError, match="blank values in required column: Source"):
        loader.load_dataset()


def test_load_dataset_rejects_missing_columns(tmp_path, monkeypatch):
    path = tmp_path / "data.xlsx"
    _write_dataset(path, {"Test_ID": ["T1"], "Source": ["s"]})
    _patch(monkeypatch, path)

    with pytest.raises(ValueError, match="missing required columns"):
        loader.load_dataset()
