"""Live run console output.

Observability concern: this module owns every print() the run produces, so
"what does watching a live run look like" is one file, not statements
scattered across main.py and the evaluation runner. It renders data handed
to it by the other layers - it does not compute anything itself.
"""


def banner(title):
    print("=" * 60)
    print(title)
    print("=" * 60)


def run_id_line(run_id):
    print(f"Run ID     : {run_id}")


def discovery_line(models, judge, judge_provider):
    print("Generators : " + ", ".join(models))
    print(f"Judge      : {judge} (provider={judge_provider})")


def dataset_loading_line(filename):
    print(f"Test cases : loading from {filename}")


def dataset_loaded_line(count, limit, temperature):
    print(f"Test cases : {count} (limit={limit})")
    print(f"Temperature: {temperature}")


def phase_banner(step, total_steps, title):
    print()
    print("-" * 60)
    print(f"STEP {step}/{total_steps}: {title}")
    print("-" * 60)


def generator_concurrency_note(workers, total):
    if workers >= total:
        note = f"{total} generators running concurrently"
    elif workers == 1:
        note = (
            f"{total} generators running one at a time "
            "(concurrency=1 - avoids reloading different models on the "
            "shared Ollama server)"
        )
    else:
        note = f"{total} generators running {workers} at a time"
    print(note)


def generator_completed(model, completed, total, avg_duration):
    avg_note = f", {avg_duration:.1f}s avg" if avg_duration is not None else ""
    print(f"  ✓ {model}: {completed}/{total} responses completed{avg_note}")


def generator_failed(model, error):
    print(f"  ✗ {model}: generation failed - {error}")


def evaluation_intro(generator_count, workers, metric_count, judge):
    print(
        f"{generator_count} generator evaluation workers; {workers} active "
        f"at a time. Each testcase runs {metric_count} metrics concurrently "
        f"against judge={judge}."
    )
    print(
        "Note: client-side concurrency does not imply backend parallelism; "
        "a single-request-at-a-time server (e.g. default Ollama) will "
        "serialize judge calls regardless of this setting."
    )


def judge_call(context_label, step, elapsed_seconds):
    """One internal DeepEval judge call, printed as it completes.

    context_label is "[generator][test_id][metric]"; step is the DeepEval
    schema class name for that call (Truths, Claims, Verdicts, ...
    ScoreReason, or "response" when no schema was used) - this is the real
    internal step, not a guess, since it's read from the schema DeepEval
    itself passed to the judge call.
    """
    print(f"      {context_label} {step}: {elapsed_seconds:.1f}s")


def testcase_result(model, index, total, test_id, status):
    print(f"    [{model}] [{index}/{total}] {test_id}: {status}")


def metric_failure_detail(name, verdict, score, reason):
    score_text = f"{score:.3f}" if score is not None else "n/a"
    print(f"      -> {name}: {verdict}, score={score_text}")
    reason_text = " ".join(str(reason or "").split())
    if len(reason_text) > 280:
        reason_text = reason_text[:277] + "..."
    if reason_text:
        print(f"         Judge reason: {reason_text}")


def generator_evaluation_summary(model, passed, total):
    print(f"  ✓ {model}: {passed}/{total} testcases passed")


def generator_evaluation_failed(model, error):
    print(f"  ✗ {model}: evaluation failed - {error}")


def file_written(label, path):
    print(f"{label}: {path}")


def completion_summary(report_file, verdict):
    print()
    banner("Evaluation Complete")
    print(f"Final report: {report_file}")
    print(verdict)
    print("=" * 60)


def timing_report(phase_timings, metric_durations):
    total = phase_timings.get("Total run time", 0.0)
    print("\n" + "=" * 60)
    print("Timing Report")
    print("=" * 60)
    for label, seconds in phase_timings.items():
        percentage = (
            f" ({seconds / total * 100:.1f}%)"
            if total and label != "Total run time"
            else ""
        )
        print(f"  {label:<24}: {seconds:>8.2f}s{percentage}")

    if metric_durations:
        print("\n  Avg metric evaluation duration:")
        for name, seconds in sorted(
            metric_durations.items(),
            key=lambda item: item[1],
            reverse=True,
        ):
            print(f"    {name:<20}: {seconds:>6.2f}s")

    print("=" * 60)


def run_failed(exc):
    print()
    banner("Evaluation failed")
    print(f"{type(exc).__name__}: {exc}")
    print("=" * 60)
