from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent

DATASET_PATH = ROOT / "dataset" / "hallucination_benchmark.xlsx"
REPORT_ROOT = ROOT / "reports"
PROMPT_ROOT = ROOT / "prompts"

# LLM configuration
OLLAMA_BASE_URL = "http://192.168.1.81:11434"

GENERATOR_MODEL = "qwen2.5:14b"
JUDGE_MODEL = "gpt-oss:20b"
OPTIMIZER_MODEL = "gpt-oss:20b"

REQUEST_TIMEOUT = 180
TEMPERATURE = 0

# POC control: one original testcase and one synthetic testcase
# for the whole lifecycle.
TEST_CASE_LIMIT = 1
SYNTHETIC_TEST_CASE_COUNT = 1

if TEST_CASE_LIMIT < 1:
    raise ValueError("TEST_CASE_LIMIT must be at least 1.")

if SYNTHETIC_TEST_CASE_COUNT < 1:
    raise ValueError("SYNTHETIC_TEST_CASE_COUNT must be at least 1.")

QUALITY_THRESHOLD = 0.90
REVIEW_THRESHOLD = 0.70

METRIC_WEIGHTS = {
    "hallucination": 0.25,
    "faithfulness": 0.20,
    "correctness": 0.20,
    "completeness": 0.15,
    "answer_relevancy": 0.10,
    "bias": 0.10,
}

os.environ.setdefault("DEEPEVAL_VERBOSE_MODE", "0")
os.environ.setdefault("LOG_LEVEL", "WARNING")