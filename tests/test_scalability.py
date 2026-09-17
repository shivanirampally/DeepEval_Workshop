import pandas as pd
import pytest

from config import SYNTHETIC_TEST_CASE_COUNT
from testdata.synthetic_data import _parse_json_array, validate_synthetic_dataset


def test_parse_json_array_accepts_plain_json():
    value = _parse_json_array('[{"question":"Q","golden_answer":"A"}]')
    assert value[0]["question"] == "Q"


def test_parse_json_array_accepts_fenced_json():
    value = _parse_json_array('```json\n[{"question":"Q","golden_answer":"A"}]\n```')
    assert value[0]["golden_answer"] == "A"


def test_parse_json_array_accepts_text_wrapped_json():
    value = _parse_json_array('Sure, here you go:\n[{"question":"Q","golden_answer":"A"}]\nHope that helps!')
    assert value[0]["question"] == "Q"


def test_parse_json_array_skips_unrelated_brackets():
    # A citation like "[1]" before the real array, and a bracketed word after
    # it, are both themselves valid JSON -- picking the first '[' regardless
    # of shape would silently return [1] instead of the real array.
    text = (
        'Sure! Here is the case, based on source [1]:\n'
        '[{"question":"Q1","golden_answer":"A1"}]\n'
        'Let me know if you need more [details].'
    )
    value = _parse_json_array(text)
    assert value == [{"question": "Q1", "golden_answer": "A1"}]


def test_parse_json_array_rejects_non_object_array():
    with pytest.raises(ValueError):
        _parse_json_array("Here are the numbers: [1, 2, 3]")


def test_synthetic_dataset_contract():
    # validate_synthetic_dataset checks against the live SYNTHETIC_TEST_CASE_COUNT
    # config value, so build a frame with exactly that many rows rather than a
    # hardcoded count -- otherwise this test silently depends on whatever
    # config.py's default (or the local .env) happens to be at run time.
    frame = pd.DataFrame([{
        "Test_ID": f"SYN{i:03d}",
        "Source": "source",
        "Question": "question",
        "Golden_Answer": "answer",
        "Human_Reviewed": "PENDING",
    } for i in range(1, SYNTHETIC_TEST_CASE_COUNT + 1)])
    assert len(validate_synthetic_dataset(frame)) == SYNTHETIC_TEST_CASE_COUNT
