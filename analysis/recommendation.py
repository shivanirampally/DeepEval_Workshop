def recommend(generator_rows, minimum_testcases=10):
    """Return a conservative generator comparison statement.

    A small smoke sample is useful for validating the framework but is not
    enough evidence for a stable generator ranking - claiming a "winner"
    from 1-3 test cases would be statistically meaningless. Technical
    evaluation errors (judge timeouts, rate limits, connection failures)
    also invalidate a comparison entirely, since an affected generator's
    score isn't a real quality measurement.
    """
    if not generator_rows:
        return "No generator could be evaluated."

    total_testcases = max(
        (int(row.get("total_testcases") or 0) for row in generator_rows),
        default=0,
    )

    technical_error_rows = [
        row for row in generator_rows
        if int(row.get("technical_error_count") or 0) > 0
    ]
    if technical_error_rows:
        affected = ", ".join(
            row.get("generator", "unknown")
            for row in technical_error_rows
        )
        return (
            "No final winner: technical evaluation errors prevented a "
            f"reliable comparison for {affected}."
        )

    if total_testcases < minimum_testcases:
        return (
            f"No clear winner in this {total_testcases}-test sample. "
            f"The sample is below the {minimum_testcases}-test recommendation "
            "for a stable generator comparison; use the larger benchmark and "
            "repeated runs to assess quality stability before selecting a "
            "preferred generator."
        )

    ranked = sorted(
        generator_rows,
        key=lambda row: (
            row.get("quality_gate") == "PASS",
            row.get("overall_score")
            if row.get("overall_score") is not None else -1,
            row.get("passed_testcases") or 0,
        ),
        reverse=True,
    )

    best = ranked[0]
    if best.get("overall_score") is None:
        return (
            "No final winner: technical evaluation errors prevented "
            "a reliable comparison."
        )

    if len(ranked) > 1:
        second = ranked[1]
        second_score = second.get("overall_score")
        gap = (
            best["overall_score"] - second_score
            if second_score is not None else None
        )
        if (
            gap is not None
            and gap < 0.03
            and best.get("passed_testcases") == second.get("passed_testcases")
        ):
            return (
                "No clear winner. The leading generators are too close on "
                "quality; use latency, response stability, and human review "
                "as secondary evidence."
            )

    metric_scores = best.get("metric_scores", {})
    if metric_scores and len(set(metric_scores.values())) == 1:
        score = next(iter(metric_scores.values()))
        return (
            f"Preferred generator: {best['generator']}. "
            f"It passed {best['passed_testcases']}/{best['total_testcases']} "
            f"testcases with weighted quality score "
            f"{best['overall_score']:.3f}. "
            f"All configured metrics scored {score:.3f}."
        )

    weak = sorted(
        metric_scores.items(),
        key=lambda item: item[1],
    )[:2]
    weak_text = ", ".join(
        f"{name}={score:.3f}" for name, score in weak
    )

    return (
        f"Preferred generator: {best['generator']}. "
        f"It passed {best['passed_testcases']}/{best['total_testcases']} "
        f"testcases with weighted quality score "
        f"{best['overall_score']:.3f}. "
        f"The two weakest areas were {weak_text}."
    )


def improvement_suggestions(failures):
    suggestions = []

    for row in failures:
        metric = row.get("metric", "")
        if metric in {"hallucination", "faithfulness"}:
            suggestions.append(
                "Strengthen source grounding: answer only from the supplied "
                "source and explicitly state when the source does not contain "
                "the answer."
            )
        elif metric == "correctness":
            suggestions.append(
                "Require factual claims to be checked against the supplied "
                "source and golden answer before they are included."
            )
        elif metric == "completeness":
            suggestions.append(
                "Require the response to cover every information point needed "
                "by the question while avoiding unsupported additions."
            )
        elif metric == "answer_relevancy":
            suggestions.append(
                "Require a direct answer first and remove information that "
                "does not help answer the question."
            )
        elif metric == "bias":
            suggestions.append(
                "Require neutral, evidence-based wording and avoid unsupported "
                "judgments about people, groups, or intent."
            )

    return sorted(set(suggestions))
