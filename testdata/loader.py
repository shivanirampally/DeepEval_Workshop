"""Benchmark dataset loading.

Testdata layer: owns reading the benchmark workbook and building the
per-row generator prompt. Validates shape (required columns, requested
sample size) before anything downstream depends on it.
"""
import pandas as pd

from settings import config


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

    if (
        dataset["Test_ID"].isna().any()
        or dataset["Test_ID"].astype(str).str.strip().eq("").any()
    ):
        raise ValueError("Test_Data contains a blank Test_ID.")

    if dataset["Test_ID"].duplicated().any():
        duplicates = dataset.loc[
            dataset["Test_ID"].duplicated(), "Test_ID"
        ].astype(str).tolist()
        raise ValueError(
            "Test_Data contains duplicate Test_ID values: "
            + ", ".join(duplicates[:10])
        )

    for column in ("Source", "Question", "Golden_Answer"):
        if dataset[column].isna().any():
            raise ValueError(
                f"Test_Data contains blank values in required column: {column}"
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
