from pathlib import Path
import pandas as pd

SOURCE = Path(__file__).resolve().parent / "hallucination_benchmark.xlsx"
OUTPUT = Path(__file__).resolve().parent / "synthetic_test_data.xlsx"

# These scenarios target common baseline gaps found in source-grounded QA.
# The source/question text can be replaced with actual V1 failure patterns
# after the baseline report is reviewed.
SYNTHETIC_CASES = [
    {
        "Test_ID": "SYN01",
        "Scenario": "Missing information",
        "Source": "The policy provides email support from Monday to Friday.",
        "Question": "What is the weekend chat-support number?",
        "Golden_Answer": "The weekend chat-support number is not provided in the source.",
    },
    {
        "Test_ID": "SYN02",
        "Scenario": "Partial information",
        "Source": "The service is available in India and Singapore. Pricing is not listed.",
        "Question": "Where is the service available and what is its price in India?",
        "Golden_Answer": "The service is available in India and Singapore. The price in India is not provided.",
    },
    {
        "Test_ID": "SYN03",
        "Scenario": "Unsupported inference",
        "Source": "The office is open from 9 AM to 6 PM.",
        "Question": "How many employees work in the office?",
        "Golden_Answer": "The number of employees is not provided in the source.",
    },
]

def create_synthetic_data():
    # Keep the original Project 1 benchmark untouched.
    if not SOURCE.exists():
        raise FileNotFoundError(f"Baseline dataset not found: {SOURCE}")

    pd.DataFrame(SYNTHETIC_CASES).to_excel(OUTPUT, index=False)
    print(f"Synthetic test data created: {OUTPUT}")

if __name__ == "__main__":
    create_synthetic_data()
