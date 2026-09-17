from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent

DATASET_PATH = ROOT / "testdata" / "dataset" / "hallucination_benchmark.xlsx"
REPORT_ROOT = ROOT / "reports"

# Ollama server connection
OLLAMA_BASE_URL = "http://192.168.1.81:11434"

# Generator & Judge model connection
GENERATOR_MODEL = "llama3:instruct"
JUDGE_MODEL = "qwen3-coder:30b"
OPTIMIZER_MODEL = "qwen3-coder:30b"
SYNTHETIC_MODEL = OPTIMIZER_MODEL

REQUEST_TIMEOUT = 180

# A transient 429/5xx or timeout/connection error gets retried this many times
# (exponential backoff) before ollama_generate gives up -- see execution/ollama_client.py.
OLLAMA_MAX_RETRIES = 2
OLLAMA_RETRY_BACKOFF_SECONDS = 1.0
TEMPERATURE = 0.0

GENERATOR_CONCURRENCY = 3
# Measured on our Ollama server (2026-09-17 sweep, N=3 testcases/18 judge calls):
# wall-clock was 87.5s/87.8s/92.2s at concurrency 3/6/12 -- essentially flat --
# while concurrency=1 was slower (102.3s). The server has one real throughput
# ceiling regardless of concurrency; 3 reaches it with half the connections/
# threads that 6 used, so it's the better default. Re-run the sweep if the
# backend changes (more GPUs, a different server, etc.).
JUDGE_CONCURRENCY = 3
SYNTHETIC_MODE = "direct"

TEST_CASE_LIMIT = 10
SYNTHETIC_TEST_CASE_COUNT = 10

if TEST_CASE_LIMIT < 1:
    raise ValueError("TEST_CASE_LIMIT must be at least 1.")

if SYNTHETIC_TEST_CASE_COUNT < 1:
    raise ValueError("SYNTHETIC_TEST_CASE_COUNT must be at least 1.")

if GENERATOR_CONCURRENCY < 1:
    raise ValueError("GENERATOR_CONCURRENCY must be at least 1.")

if JUDGE_CONCURRENCY < 1:
    raise ValueError("JUDGE_CONCURRENCY must be at least 1.")

if SYNTHETIC_MODE not in {"direct", "deepeval"}:
    raise ValueError("SYNTHETIC_MODE must be 'direct' or 'deepeval'.")

if REQUEST_TIMEOUT < 1:
    raise ValueError("REQUEST_TIMEOUT must be at least 1 second.")

if OLLAMA_MAX_RETRIES < 0:
    raise ValueError("OLLAMA_MAX_RETRIES cannot be negative.")

if OLLAMA_RETRY_BACKOFF_SECONDS < 0:
    raise ValueError("OLLAMA_RETRY_BACKOFF_SECONDS cannot be negative.")

if not 0 <= TEMPERATURE <= 2:
    raise ValueError("TEMPERATURE must be between 0 and 2.")

QUALITY_THRESHOLD = 0.90
REVIEW_THRESHOLD = 0.70

if not 0 <= REVIEW_THRESHOLD <= QUALITY_THRESHOLD <= 1:
    raise ValueError("Thresholds must satisfy 0 <= REVIEW_THRESHOLD <= QUALITY_THRESHOLD <= 1.")

METRIC_NAMES = (
    "hallucination",
    "faithfulness",
    "correctness",
    "completeness",
    "answer_relevancy",
    "bias",
)

os.environ.setdefault("DEEPEVAL_VERBOSE_MODE", "0")
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
