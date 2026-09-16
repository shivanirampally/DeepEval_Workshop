"""Benchmark dataset loading.

Testdata layer: owns reading the benchmark workbook and building the
per-row generator prompt. Validates shape (required columns, requested
sample size) before anything downstream depends on it.
"""
import pandas as pd

import config


REQUIRED_TEST_DATA_COLUMNS = {"Test_ID", "Source", "Question", "Golden_Answer"}


def load_dataset():
    dataset = pd.read_excel(config.DATASET_PATH, sheet_name="Test_Data")

    if config.TEST_CASE_LIMIT > len(dataset):
        raise ValueError(
            f"test_case_limit={config.TEST_CASE_LIMIT} but dataset contains "
            f"only {len(dataset)} test cases."
        )

    missing = REQUIRED_TEST_DATA_COLUMNS.difference(dataset.columns)
    if missing:
        raise ValueError(
            "Test_Data is missing required columns: "
            + ", ".join(sorted(missing))
        )

    return dataset.head(config.TEST_CASE_LIMIT).copy()


def load_prompt():
    prompts = pd.read_excel(config.DATASET_PATH, sheet_name="Prompt_Versions")
    if prompts.empty or "Prompt" not in prompts.columns:
        raise ValueError("Prompt_Versions must contain at least one Prompt.")
    return str(prompts.iloc[0]["Prompt"])


def build_prompt(template, row):
    return template.format(
        source=str(row["Source"]),
        question=str(row["Question"]),
    )
