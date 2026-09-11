import re

import pandas as pd

from config.model_discovery import discover
from config.settings import (
    DATASET_PATH,
    PREFERRED_GENERATORS,
    PROJECT_1_REPORT_ROOT,
    REPORT_ROOT,
    TEST_CASE_LIMIT,
)
from evaluate import run_trajectory
from reporting.report import save


def _preferred_generator():
    reports = sorted(
        PROJECT_1_REPORT_ROOT.glob("*/generator_comparison_*.xlsx"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )

    if not reports:
        return None

    try:
        summary = pd.read_excel(
            reports[0],
            sheet_name="Executive Summary",
        )

        row = summary.loc[
            summary["Section"] == "Final Verdict"
        ]

        if row.empty:
            return None

        value = str(row.iloc[0]["Value"])

        for generator in PREFERRED_GENERATORS:
            if f"Preferred generator: {generator}" in value:
                return generator

        return None

    except Exception:
        return None


def main():
    dataset = pd.read_excel(
        DATASET_PATH,
        sheet_name="Test_Data",
    )

    if TEST_CASE_LIMIT > 0:
        dataset = dataset.head(TEST_CASE_LIMIT)

    generators, judge = discover()
    preferred = _preferred_generator()

    print(
        "Trajectory generator path:",
        " -> ".join(generators),
    )
    print("Trajectory judge:", judge)
    print(
        "Project 1 preferred generator:",
        preferred or "not available",
    )
    print(
        "Trajectory test case count:",
        len(dataset),
    )

    rows = run_trajectory(
        dataset,
        generators,
        judge,
        preferred_generator=preferred,
    )

    path = save(
        rows,
        REPORT_ROOT,
    )

    print(f"Trajectory report: {path}")

    print(
        "Use this report with Project 1 to explain the "
        "model-selection result: trajectory evaluation shows "
        "the complete multi-generator decision path, while "
        "Project 1's end-to-end metrics explain final response quality."
    )


if __name__ == "__main__":
    main()