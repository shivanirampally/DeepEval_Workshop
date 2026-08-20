import os
import pandas as pd

from llm_model import llm
from metrics import evaluate_hallucination
from report_generator import generate_report
from logger import info, section


DATASET_PATH = "dataset/evaluation_data.xlsx"
REPORT_PATH = "reports/evaluation_report.xlsx"


def main():
    section("DeepEval Workshop")

    try:
        data = pd.read_excel(DATASET_PATH)

        info(f"Loaded {len(data)} evaluation scenarios")
        info(f"Model: {os.getenv('GEMINI_MODEL')}")

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
                error_message = str(e)

                if "quota" in error_message.lower() or "429" in error_message:
                    reason = "API quota or rate limit exceeded."
                    info("API quota or rate limit exceeded.")

                elif "503" in error_message or "unavailable" in error_message.lower():
                    reason = "Gemini service temporarily unavailable."
                    info("Gemini service temporarily unavailable.")

                elif "timeout" in error_message.lower():
                    reason = "Evaluation timed out."
                    info("Evaluation timed out.")

                else:
                    reason = error_message
                    info(f"Evaluation error: {error_message}")

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

    except FileNotFoundError as e:
        info(f"File not found: {e}")

    except Exception as e:
        info(f"Execution failed: {e}")


if __name__ == "__main__":
    main()