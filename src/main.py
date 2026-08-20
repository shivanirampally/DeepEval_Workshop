import pandas as pd

from src.llm_model import llm
from src.metrics import evaluate_hallucination
from src.report_generator import generate_report
from src.logger import info, section


DATASET_PATH = "dataset/evaluation_data.xlsx"
REPORT_PATH = "reports/evaluation_report.xlsx"


def main():
    section("DeepEval Workshop")

    try:
        data = pd.read_excel(DATASET_PATH)

        info(f"Loaded {len(data)} evaluation scenarios")

        results = []

        for _, row in data.iterrows():

            scenario_id = row["Scenario_ID"]
            requirement = row["Requirement"]
            context = row["Context"]
            actual_output = row["Actual_Output"]

            section(scenario_id)

            try:
                if pd.isna(actual_output) or not str(actual_output).strip():

                    info("Generating response...")

                    actual_output = llm.invoke(
                        f"""
Requirement:
{requirement}

Context:
{context}

Generate a concise response for this requirement.
"""
                    )

                info("Running hallucination evaluation...")

                result = evaluate_hallucination(
                    input_text=requirement,
                    actual_output=actual_output,
                    context=context
                )

                status = "PASS" if result["success"] else "FAIL"

                info(f"Score: {result['score']}")
                info(f"Status: {status}")

                results.append({
                    "Scenario_ID": scenario_id,
                    "Purpose": row["Purpose"],
                    "Requirement": requirement,
                    "Context": context,
                    "Actual_Output": actual_output,
                    "Score": result["score"],
                    "Status": status,
                    "Reason": result["reason"]
                })

            except Exception as e:

                error_message = str(e).lower()

                if "quota" in error_message or "429" in error_message:
                    reason = "API quota or rate limit exceeded."
                    info("API quota or rate limit exceeded.")

                elif "503" in error_message or "unavailable" in error_message:
                    reason = "Gemini service is temporarily unavailable."
                    info("Gemini service is temporarily unavailable.")

                elif "timeout" in error_message:
                    reason = "Evaluation timed out."
                    info("Evaluation timed out.")

                else:
                    reason = "Unexpected evaluation error."
                    info("Unexpected evaluation error.")

                info("Status: ERROR")

                results.append({
                    "Scenario_ID": scenario_id,
                    "Purpose": row["Purpose"],
                    "Requirement": requirement,
                    "Context": context,
                    "Actual_Output": actual_output,
                    "Score": "",
                    "Status": "ERROR",
                    "Reason": reason
                })

        generate_report(results, REPORT_PATH)

        section("Completed")
        info(f"Report: {REPORT_PATH}")

    except FileNotFoundError:
        info(f"Dataset not found: {DATASET_PATH}")

    except Exception:
        info("Unable to complete evaluation.")


if __name__ == "__main__":
    main()