from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

from analysis.failure_analysis import analyze
from config import (
    DATASET_PATH,
    GENERATOR_MODEL,
    OLLAMA_BASE_URL,
    REQUEST_TIMEOUT,
    TEMPERATURE,
)
from deepeval_framework.evaluator import evaluate_response
from prompts import BASE_PROMPT
from reporting.excel import save_baseline_report

def load_data():
    return pd.read_excel(DATASET_PATH)

def generate(row):
    prompt = BASE_PROMPT.format(
        source=str(row["Source"]),
        question=str(row["Question"]),
    )
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": GENERATOR_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": TEMPERATURE},
        },
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()["response"].strip()

def main():
    dataset = load_data()
    generated = {}

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(generate, row): str(row["Test_ID"])
            for _, row in dataset.iterrows()
        }
        for future in as_completed(futures):
            generated[futures[future]] = future.result()

    rows = []
    for _, row in dataset.iterrows():
        test_id = str(row["Test_ID"])
        results = evaluate_response(row, generated[test_id])
        for metric, result in results.items():
            rows.append({
                "test_id": test_id,
                "metric": metric,
                "score": result["score"],
                "passed": result["passed"],
                "status": result["status"],
                "reason": result["reason"],
                "error": result["error"],
                "response": generated[test_id],
            })

    details = pd.DataFrame(rows)
    failures, recommendations = analyze(details)
    report = save_baseline_report(
        details,
        dataset,
        failures,
        recommendations,
    )

    print("V1 baseline evaluation completed.")
    print(f"Report: {report}")
    print("\nFailure-analysis recommendations:")
    for item in recommendations:
        print(
            f"- {item['Metric']}: "
            f"{item['Recommended Action']}"
        )

if __name__ == "__main__":
    main()
