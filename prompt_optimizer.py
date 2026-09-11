from config import OPTIMIZER_MODEL, TEMPERATURE
from llm_client import ollama_generate
from prompts import OPTIMIZER_PROMPT

def optimize_prompt(original_prompt, failure_rows, synthetic_df, output_path):
    """Create one candidate V2 prompt without changing V1."""
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

    optimized = ollama_generate(OPTIMIZER_MODEL, prompt, temperature=TEMPERATURE)

    if "{source}" not in optimized or "{question}" not in optimized:
        raise ValueError(
            "Optimizer returned a prompt without {source} and {question} placeholders."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(optimized.strip() + "\n", encoding="utf-8")
    return optimized.strip()
