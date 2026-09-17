from observability.accounting import build_llm_accounting


def test_build_llm_accounting_counts_and_percentiles():
    responses = {
        "gen1": [
            {"status": "COMPLETED", "duration_seconds": 2.0},
            {"status": "COMPLETED", "duration_seconds": 4.0},
            {"status": "ERROR", "duration_seconds": ""},
        ],
    }
    detail_rows = [
        {"llm_call_timings": [
            {"step": "Truths", "duration_seconds": 10.0},
            {"step": "Claims", "duration_seconds": 20.0},
        ]},
    ]

    result = build_llm_accounting(responses, detail_rows)

    assert result["Generator LLM Calls"] == 2
    assert result["Judge LLM Calls"] == 2
    assert result["Total LLM Calls"] == 4
    assert result["LLM Call Maximum"] == "20.00s"


def test_build_llm_accounting_handles_empty_run():
    result = build_llm_accounting({}, [])

    assert result["Total LLM Calls"] == 0
    assert result["LLM Call P50"] == "n/a"
    assert result["LLM Call Maximum"] == "n/a"
