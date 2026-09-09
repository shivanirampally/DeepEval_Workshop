# Project - Multi Generator Cross-LLM Evaluation
This project sends the same benchmark question, source context and evaluation prompt to three different Ollama generator models. Their responses are then evaluated using the same DeepEval metrics and two independent judge models.

The purpose is to compare generator quality under the same conditions rather than allowing each model to use a different test or evaluation process.

## Model roles
Generators:
llama3:instruct -   general-purpose baseline
qwen2.5-coder:14b - code-oriented candidate
sqlcoder:15b -      SQL-oriented candidate

Each generator receives the same 10 benchmark test cases.

Judges:
gpt-oss:20b

The judge models are intentionally different from the selected generators. This provides an independent cross-LLM evaluation instead of allowing a generator to grade its own output.

## Evaluation approach
For every test case:

Same Question + Same Source + Same Prompt
                    |
        +-----------+-----------+
        |           |           |
        v           v           v
     Generator  Generator  Generator
        1           2           3
        |           |           |
        +-----------+-----------+
                    |
                    v
             Generated Responses
                    |
        +-----------+-----------+
        |                       |
      Judge 1                 Judge 2
        |                       |
    6 metrics               6 metrics
        |                       |
        +-----------+-----------+
                    |
                    v
             Quality Gate
                    |
                    v
          Generator Comparison
                    |
                    v
          Final Recommendation

This keeps the comparison fair:
same dataset
same source context
same prompt
same metric set
same thresholds
independent judges

## Metrics:
Metric	Purpose:
Hallucination: Detect unsupported claims or details that are not grounded in the supplied source
Faithfulness:	Check whether the response remains aligned with the supplied source
Correctness:	Check factual accuracy against the golden answer and source
Completeness:	Check whether important required information was missed
Answer Relevancy:	Check whether the answer directly addresses the question
Bias:	Check whether claims and wording remain neutral and evidence-based

Operational measures such as response time, technical errors and response stability are reported separately. They are not mixed into the semantic quality score.

## Score and verdict interpretation
DeepEval returns metric scores from 0.00 to 1.00.
The Excel report displays these scores as percentages for easier reading:
NOTE: A percentage is an evaluation score, not a literal probability that the same percentage of the response is correct or incorrect.
0.95 -> 95%
0.80 -> 80%
0.65 -> 65%

Hallucination example
If the Hallucination metric returns: 0.80

the report shows:
Hallucination evaluation score: 80%
Gap to ideal score:             20 percentage points
Verdict:                        REVIEW

It should not be interpreted as:
20% of the response is hallucinated

The detailed report also contains the judge's reason so the reviewer can identify which claims or wording caused the lower score.

## Quality gate
>= 0.90 → metric PASS
0.70 to < 0.90 → metric REVIEW
< 0.70 → metric FAIL

A testcase passes only when every configured metric meets the 0.90 threshold.
Both judges evaluate the same generated response.

The generator recommendation considers the quality gate first, followed by weighted semantic score, testcase pass rate and cross-judge agreement.

If the leading generators are too close to distinguish reliably, the report returns:

No clear winner
Reports
Generator response workbook
outputs/generator_responses/YYYY-MM-DD/
generator_responses_YYYYMMDD_HHMMSS.xlsx
Each generator has its own worksheet containing the generated responses and execution information.

Evaluation report
reports/YYYY-MM-DD/
generator_comparison_YYYYMMDD_HHMMSS.xlsx

The report contains:

Executive Summary — final recommendation and evaluation rules
Generator Comparison — generator-level percentage scores and PASS/REVIEW/FAIL verdicts
Testcase Comparison — testcase-level metric scores and testcase quality gate
Detailed Metric Reasons — score, threshold result, gap to ideal, judge reason and technical status
Failures — metrics/testcases that require attention
Configuration — models, thresholds and execution settings
Judge Comparison — comparison of the two independent judges

# Run
Activate the project environment: .\.venv\Scripts\Activate.ps1
Then: python main.py

NOTE: The Ollama server must be reachable from the machine running the POC and must contain the configured generator and judge models.

## Project structure
multigenerators-e2e_evals/
│
├── dataset/
│   └── hallucination_benchmark.xlsx
│
├── deepeval_framework/
│   ├── __init__.py
│   ├── evaluation.py
│   ├── metrics.py
│   └── scoring.py
│
├── generators/
│   ├── __init__.py
│   └── ollama_client.py
│
├── analysis/
│   ├── __init__.py
│   └── recommendation.py
│
├── reporting/
│   ├── __init__.py
│   └── excel.py
│
├── tests/
│   ├── conftest.py
│   └── test_scoring.py
│
├── config.py
├── project_config.json
├── main.py
├── README.md
├── requirements.txt
└── .gitignore
Current validation status

The project has been validated locally for:
Python imports
DeepEval installation
unit tests
Python compilation
configuration loading
framework imports
The full Ollama evaluation requires access to the configured internal Ollama server.