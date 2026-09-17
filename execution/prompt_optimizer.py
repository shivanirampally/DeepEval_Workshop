from time import perf_counter

from config import OPTIMIZER_MODEL, TEMPERATURE
from execution.ollama_client import ollama_generate
from testdata.prompts import OPTIMIZER_PROMPT


def optimize_prompt(original_prompt, failure_rows, synthetic_df, output_path):
    """Ask the optimizer model for one candidate V2 prompt -- V1 stays untouched either way.

    Returns (prompt_text, seconds_spent_calling_the_optimizer_model) so the
    caller can report the LLM time without re-parsing the log stream.
    """
    failure_text = "\n".join(
        f"{r['test_id']} | {r['metric']} | {r['verdict']} | "
        f"{r['reason']} | {r['recommended_action']}"
        for _, r in failure_rows.iterrows()
    )
    synthetic_text = synthetic_df.to_csv(index=False)

    prompt = OPTIMIZER_PROMPT
    prompt = prompt.replace("{original_prompt}", original_prompt)
    prompt = prompt.replace(
        "{failure_analysis}",
        failure_text or "No repeated failure pattern was identified.",
    )
    prompt = prompt.replace("{synthetic_context}", synthetic_text)

    started = perf_counter()
    optimized = ollama_generate(OPTIMIZER_MODEL, prompt, temperature=TEMPERATURE)
    elapsed = round(perf_counter() - started, 3)

    # The rest of the lifecycle formats this prompt with .format(source=..., question=...),
    # so a candidate missing either placeholder would blow up much later and be confusing to trace.
    if "{source}" not in optimized or "{question}" not in optimized:
        raise ValueError(
            "Optimizer returned a prompt without {source} and {question} placeholders."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(optimized.strip() + "\n", encoding="utf-8")
    return optimized.strip(), elapsed
