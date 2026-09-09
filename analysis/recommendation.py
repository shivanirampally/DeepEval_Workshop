def recommend(generator_rows):
    if not generator_rows:
        return "No generator could be evaluated."

    ranked = sorted(
        generator_rows,
        key=lambda row: (
            row["quality_gate"] == "PASS",
            row["overall_score"] if row["overall_score"] is not None else -1,
            row["passed_testcases"],
            row.get("judge_agreement", 0),
        ),
        reverse=True,
    )

    best = ranked[0]
    if best["overall_score"] is None:
        return "No final winner: technical evaluation errors prevented a reliable comparison."

    if len(ranked) > 1:
        second = ranked[1]
        gap = best["overall_score"] - (second["overall_score"] or 0)
        if gap < 0.03 and best["passed_testcases"] == second["passed_testcases"]:
            return (
                "No clear winner. The leading generators are too close on quality; "
                "use latency, response stability, and human review as the final tie-breaker."
            )

    weak = sorted(best.get("metric_scores", {}).items(), key=lambda item: item[1])[:2]
    weak_text = ", ".join(f"{name}={score:.3f}" for name, score in weak)

    return (
        f"Preferred generator: {best['generator']}. "
        f"It passed {best['passed_testcases']}/{best['total_testcases']} testcases "
        f"with weighted quality score {best['overall_score']:.3f}. "
        f"Cross-judge agreement={best.get('judge_agreement', 0):.3f}. "
        f"The two weakest areas were {weak_text}."
    )


def improvement_suggestions(failures):
    suggestions = []
    for row in failures:
        metric = row.get("metric", "")
        if metric in {"hallucination", "faithfulness"}:
            suggestions.append(
                "Strengthen source grounding: answer only from the supplied source and explicitly state when the source does not contain the answer."
            )
        elif metric == "correctness":
            suggestions.append(
                "Require factual claims to be checked against the supplied source and golden answer before they are included."
            )
        elif metric == "completeness":
            suggestions.append(
                "Require the response to cover every information point needed by the question while avoiding unsupported additions."
            )
        elif metric == "answer_relevancy":
            suggestions.append(
                "Require a direct answer first and remove information that does not help answer the question."
            )
        elif metric == "bias":
            suggestions.append(
                "Require neutral, evidence-based wording and avoid unsupported judgments about people, groups, or intent."
            )
    return sorted(set(suggestions))
