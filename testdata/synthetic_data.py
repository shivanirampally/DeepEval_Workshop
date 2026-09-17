from datetime import datetime
import json
import re

import pandas as pd

from config import (
    OLLAMA_BASE_URL,
    OPTIMIZER_MODEL,
    SYNTHETIC_MODE,
    SYNTHETIC_MODEL,
    SYNTHETIC_TEST_CASE_COUNT,
)
from execution.ollama_client import ollama_generate
from governance_observability.logging_config import log_llm_timing


def validate_synthetic_dataset(frame):
    """Guard rails for the human-reviewed synthetic testcase sheet before it feeds back into the lifecycle."""
    required = {"Test_ID", "Source", "Question", "Golden_Answer", "Human_Reviewed"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Synthetic dataset is missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Synthetic dataset is empty.")
    if len(frame) != SYNTHETIC_TEST_CASE_COUNT:
        raise ValueError(
            f"Expected exactly {SYNTHETIC_TEST_CASE_COUNT} synthetic testcase(s), "
            f"but found {len(frame)}."
        )
    if frame["Test_ID"].duplicated().any():
        raise ValueError("Synthetic dataset contains duplicate Test_ID values.")
    for column in ["Source", "Question", "Golden_Answer"]:
        if frame[column].fillna("").astype(str).str.strip().eq("").any():
            raise ValueError(f"Synthetic dataset contains blank {column} values.")
    return frame


def validate_human_approved_synthetic_dataset(frame):
    """Same checks as validate_synthetic_dataset, plus: every row must actually
    carry a recorded human approval. Call this wherever the lifecycle is about
    to *use* the synthetic dataset (Synthetic V1/V2 evaluation, optimizer) --
    never where it's merely being displayed for review, since it's expected
    to still say PENDING at that point."""
    frame = validate_synthetic_dataset(frame)
    reviewed = frame["Human_Reviewed"].fillna("").astype(str).str.strip().str.upper()
    if not reviewed.eq("APPROVED").all():
        raise ValueError("Synthetic dataset has not been human-approved.")
    return frame


def _first_json_array_of_objects(text):
    """Scan for the first '[' that decodes into a non-empty JSON array of
    objects, skipping any '[' that parses as valid JSON but isn't that shape
    (a stray citation like "[1]" is itself valid JSON -- an array containing
    one integer -- so it has to be rejected by shape, not just by validity,
    or it gets mistaken for the real array before we ever reach it)."""
    decoder = json.JSONDecoder()
    start = 0
    while True:
        start = text.find("[", start)
        if start == -1:
            return None
        try:
            value, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            start += 1
            continue
        if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
            return value
        start += 1


def _parse_json_array(text):
    """Pull a JSON array of {question, golden_answer} objects out of a model response.

    Models routinely wrap JSON in a ```json fence or add a stray sentence before/after
    it (e.g. "Sure, here you go: [...] Let me know if you need more [details]."), so
    this strips fences first, then -- if a straight json.loads fails -- scans for the
    first bracket that decodes into the expected shape (a list of objects) and ignores
    everything else. Grabbing from the first '[' to the LAST ']' in the whole response
    (a naive regex) would overshoot into unrelated brackets in the surrounding prose;
    stopping at the FIRST '[' regardless of shape would just as easily grab a stray
    "[1]"-style citation before the model ever gets to the real array.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        value = _first_json_array_of_objects(cleaned)
        if value is None:
            raise ValueError("Synthetic generator did not return a JSON array of objects.")
        return value
    if not isinstance(value, list):
        raise ValueError("Synthetic generator response must be a JSON array.")
    return value


def _direct_synthetic_generation(sources, failure_rows):
    """One direct JSON-mode request for the whole batch of synthetic testcases.

    DeepEval's Synthesizer does this too, but internally: context construction,
    golden generation, and style/complexity evolutions each cost their own LLM
    call, all reported under a single wrapped timing entry. Asking the model
    directly for the JSON shape we actually need collapses that down to one
    real call -- much faster, at the cost of losing the Synthesizer's own
    calibration/evolution steps. See SYNTHETIC_MODE=deepeval to fall back.
    """
    failure_text = "\n".join(
        f"- {row['metric']}: {row['reason']}" for _, row in failure_rows.iterrows()
    )
    prompt = f"""Create exactly {SYNTHETIC_TEST_CASE_COUNT} targeted synthetic QA test case(s) from the supplied source.

Purpose:
- Stress the failure patterns found during the V1 evaluation.
- Stay completely grounded in the supplied source.
- Do not introduce outside facts.
- Each case must have a question and a source-supported golden answer.

Observed V1 failure patterns:
{failure_text or "- No detailed failure reason was available."}

Return ONLY valid JSON in this exact shape, nothing else:
[
  {{"question": "...", "golden_answer": "..."}}
]

Source:
{chr(10).join(sources)}
"""
    # Two different failure modes need two different messages: the call itself
    # failing (network/server/model issue) vs. the call succeeding but the
    # model's text not being parsable JSON (a prompt/response formatting
    # problem). Pointing someone at "check the server" for a malformed
    # response wastes their troubleshooting time.
    try:
        with log_llm_timing("synthetic_data_generation", SYNTHETIC_MODEL) as timing:
            text = ollama_generate(SYNTHETIC_MODEL, prompt, temperature=0)
    except Exception as exc:
        raise RuntimeError(
            "Synthetic data generation failed while calling the configured Ollama "
            f"model ({SYNTHETIC_MODEL} at {OLLAMA_BASE_URL}). "
            "Verify that the server is reachable and the model is available."
        ) from exc
    try:
        goldens = _parse_json_array(text)
    except ValueError as exc:
        raise RuntimeError(
            f"Synthetic data generation got a response from {SYNTHETIC_MODEL}, but it "
            f"wasn't parsable as the expected JSON array of {{question, golden_answer}} "
            f"objects: {exc}"
        ) from exc
    return goldens, timing["elapsed"]


def _deepeval_synthetic_generation(sources):
    """Original path: DeepEval's own Synthesizer. Slower, but more heavily
    calibrated -- kept available via SYNTHETIC_MODE=deepeval for comparison
    or if the direct JSON path ever produces lower-quality synthetic cases."""
    from deepeval.models import OllamaModel
    from deepeval.synthesizer import Synthesizer

    model = OllamaModel(model=OPTIMIZER_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
    synthesizer = Synthesizer(model=model, async_mode=False)

    try:
        with log_llm_timing("synthetic_data_generation", OPTIMIZER_MODEL) as timing:
            goldens = synthesizer.generate_goldens_from_contexts(
                contexts=[sources],
                include_expected_output=True,
                max_goldens_per_context=SYNTHETIC_TEST_CASE_COUNT,
            )
    except Exception as exc:
        raise RuntimeError(
            "Synthetic data generation failed while calling the configured Ollama "
            f"synthesizer ({OPTIMIZER_MODEL} at {OLLAMA_BASE_URL}). "
            "Verify that the server is reachable and the model is available."
        ) from exc

    goldens = [
        {"question": g.input, "golden_answer": g.expected_output or ""}
        for g in list(goldens)[:SYNTHETIC_TEST_CASE_COUNT]
    ]
    return goldens, timing["elapsed"]


def generate_synthetic_dataset(dataset, failure_rows, output_path):
    """Generate the configured number of targeted goldens from V1 failure sources.

    Returns (output_path, seconds_spent_generating) so the caller can log/report
    the LLM time without re-deriving it from the log stream.
    """
    if failure_rows.empty:
        raise ValueError("No V1 failures are available for targeted synthesis.")

    # Only pull sources from testcases that actually failed -- we want synthetic
    # questions that stress the same weak spots, not a random sample of the dataset.
    failed_ids = {str(v) for v in failure_rows["test_id"].dropna().tolist()}
    source_rows = dataset[dataset["Test_ID"].astype(str).isin(failed_ids)]
    sources = [str(v) for v in source_rows["Source"].dropna().unique()]
    if not sources:
        raise ValueError("No source documents were found for the V1 failure cases.")

    if SYNTHETIC_MODE == "direct":
        goldens, elapsed = _direct_synthetic_generation(sources, failure_rows)
    else:
        goldens, elapsed = _deepeval_synthetic_generation(sources)

    if len(goldens) != SYNTHETIC_TEST_CASE_COUNT:
        raise RuntimeError(
            f"Synthetic generator returned {len(goldens)} case(s); "
            f"expected {SYNTHETIC_TEST_CASE_COUNT}."
        )

    created = datetime.now().isoformat(timespec="seconds")
    rows = []
    for index, golden in enumerate(goldens, start=1):
        if not isinstance(golden, dict):
            raise ValueError("Each synthetic testcase must be a JSON object.")
        question = str(golden.get("question", "")).strip()
        golden_answer = str(golden.get("golden_answer", "")).strip()
        if not question or not golden_answer:
            raise ValueError("Synthetic generator returned a blank question or golden_answer.")
        rows.append({
            "Test_ID": f"SYN{index:03d}",
            "Scenario": "Targeted synthetic case from V1 failure sources",
            "Source": "\n\n".join(sources),
            "Question": question,
            "Golden_Answer": golden_answer,
            "Generated_At": created,
            "Human_Reviewed": "PENDING",
        })

    frame = validate_synthetic_dataset(pd.DataFrame(rows))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_excel(output_path, index=False)
    return output_path, elapsed
