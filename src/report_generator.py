import pandas as pd


def generate_report(results, output_path="reports/evaluation_report.xlsx"):
    report = pd.DataFrame(results)

    report.to_excel(
        output_path,
        index=False
    )

    print(f"Report saved to: {output_path}")