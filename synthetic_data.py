
from datetime import datetime

import pandas as pd
from deepeval.models import OllamaModel
from deepeval.synthesizer import Synthesizer

from config import OLLAMA_BASE_URL, OPTIMIZER_MODEL, SYNTHETIC_TEST_CASE_COUNT


def validate_synthetic_dataset(frame):
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


def generate_synthetic_dataset(dataset, failure_rows, output_path):
    """Generate the configured number of targeted goldens from V1 failure sources."""
    if failure_rows.empty:
        raise ValueError("No V1 failures are available for targeted synthesis.")

    failed_ids = {str(v) for v in failure_rows["test_id"].dropna().tolist()}
    source_rows = dataset[dataset["Test_ID"].astype(str).isin(failed_ids)]
    sources = [str(v) for v in source_rows["Source"].dropna().unique()]
    if not sources:
        raise ValueError("No source documents were found for the V1 failure cases.")

    model = OllamaModel(model=OPTIMIZER_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
    synthesizer = Synthesizer(model=model, async_mode=False)

    try:
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

    goldens = list(goldens)[:SYNTHETIC_TEST_CASE_COUNT]
    if len(goldens) != SYNTHETIC_TEST_CASE_COUNT:
        raise RuntimeError(
            f"Synthesizer returned {len(goldens)} golden(s); "
            f"expected {SYNTHETIC_TEST_CASE_COUNT}."
        )

    created = datetime.now().isoformat(timespec="seconds")
    rows = []
    for index, golden in enumerate(goldens, start=1):
        rows.append({
            "Test_ID": f"SYN{index:03d}",
            "Scenario": "Targeted synthetic case from V1 failure sources",
            "Source": "\n\n".join(sources),
            "Question": golden.input,
            "Golden_Answer": golden.expected_output or "",
            "Generated_At": created,
            "Human_Reviewed": "PENDING",
        })

    frame = validate_synthetic_dataset(pd.DataFrame(rows))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_excel(output_path, index=False)
    return output_path
