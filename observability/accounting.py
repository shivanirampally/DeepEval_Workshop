"""LLM-call accounting and latency statistics for a run.

Observability concern: how many generator/judge LLM calls did this run
actually make, and how long did they take (P50/P95/max)? Complements the
per-metric average in the Timing Report with run-level totals that make
cost/latency outliers visible.
"""


def _percentile(values, percentile):
    if not values:
        return None
    values = sorted(float(v) for v in values)
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * percentile / 100.0
    low = int(rank)
    high = min(low + 1, len(values) - 1)
    fraction = rank - low
    return values[low] + (values[high] - values[low]) * fraction


def _fmt(value):
    return "n/a" if value is None else f"{value:.2f}s"


def build_llm_accounting(responses, detail_rows):
    """Return run-level LLM accounting from generator and judge call timings.

    Generator calls are one Ollama generation per completed response. Judge
    calls come from the internal DeepEval model.generate instrumentation
    (evaluation/runner.py's per-call logging). Metric evaluation durations
    are not counted here - those are the outer wall-clock time per metric,
    already reported separately; this is about individual LLM calls.
    """
    generator_durations = [
        float(row["duration_seconds"])
        for rows in responses.values()
        for row in rows
        if row.get("status") == "COMPLETED"
        and isinstance(row.get("duration_seconds"), (int, float))
    ]

    judge_durations = [
        float(call["duration_seconds"])
        for item in detail_rows
        for call in (item.get("llm_call_timings") or [])
        if isinstance(call.get("duration_seconds"), (int, float))
    ]

    all_durations = generator_durations + judge_durations

    return {
        "Generator LLM Calls": len(generator_durations),
        "Judge LLM Calls": len(judge_durations),
        "Total LLM Calls": len(all_durations),
        "Generator Raw LLM Time": _fmt(sum(generator_durations)),
        "Judge Raw LLM Time": _fmt(sum(judge_durations)),
        "Total Raw LLM Time": _fmt(sum(all_durations)),
        "LLM Call P50": _fmt(_percentile(all_durations, 50)),
        "LLM Call P95": _fmt(_percentile(all_durations, 95)),
        "LLM Call Maximum": _fmt(max(all_durations) if all_durations else None),
        "Generator Call P50": _fmt(_percentile(generator_durations, 50)),
        "Generator Call P95": _fmt(_percentile(generator_durations, 95)),
        "Judge Call P50": _fmt(_percentile(judge_durations, 50)),
        "Judge Call P95": _fmt(_percentile(judge_durations, 95)),
    }
