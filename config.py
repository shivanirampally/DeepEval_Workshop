from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "dataset" / "hallucination_benchmark.xlsx"
SYNTHETIC_DATA_PATH = ROOT / "dataset" / "synthetic_test_data.xlsx"
REPORT_ROOT = ROOT / "reports"

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://192.168.1.81:11434",
).rstrip("/")

REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "180"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))
GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "qwen2.5:14b")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-oss:20b")

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
