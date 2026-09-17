from pathlib import Path
import json
import os

from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault("DEEPEVAL_VERBOSE_MODE", "0")
os.environ.setdefault("LOG_LEVEL", "WARNING")
# Confirmed via a live run: without this, DeepEval blocks process exit for
# up to 10s ("[PostHog] flush timed out") trying to send telemetry. This is
# an internal evaluation tool with no need to phone home, so it's disabled.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

SETTINGS_DIR = Path(__file__).resolve().parent
ROOT = SETTINGS_DIR.parent
CONFIG_PATH = SETTINGS_DIR / "project_config.json"

with CONFIG_PATH.open("r", encoding="utf-8") as file:
    CONFIG = json.load(file)

DATASET_PATH = ROOT / "testdata" / "hallucination_benchmark.xlsx"
RUNS_ROOT = ROOT / "runs"
RESPONSE_ROOT = RUNS_ROOT / "generator_responses"
REPORT_ROOT = RUNS_ROOT / "reports"

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    CONFIG["server"]["base_url"],
).rstrip("/")

REQUEST_TIMEOUT = int(
    os.getenv(
        "REQUEST_TIMEOUT_SECONDS",
        CONFIG["execution"]["request_timeout_seconds"],
    )
)

GENERATOR_CONCURRENCY = int(
    os.getenv(
        "GENERATOR_CONCURRENCY",
        CONFIG["execution"]["generator_concurrency"],
    )
)

JUDGE_CONCURRENCY = int(
    os.getenv(
        "JUDGE_CONCURRENCY",
        CONFIG["execution"]["metric_concurrency"],
    )
)

EVALUATION_CONCURRENCY = int(
    os.getenv(
        "EVALUATION_CONCURRENCY",
        CONFIG["execution"]["evaluation_concurrency"],
    )
)

RUNTIME_TARGET_SECONDS = int(
    os.getenv(
        "RUNTIME_TARGET_SECONDS",
        CONFIG["execution"].get("runtime_target_seconds", 300),
    )
)

RETRIES = int(os.getenv("RETRIES", CONFIG["execution"]["retries"]))
RETRY_BACKOFF_SECONDS = float(
    os.getenv(
        "RETRY_BACKOFF_SECONDS",
        CONFIG["execution"].get("retry_backoff_seconds", 1),
    )
)
TEMPERATURE = float(
    os.getenv("TEMPERATURE", CONFIG["evaluation"]["temperature"])
)

QUALITY_THRESHOLD = float(
    os.getenv(
        "QUALITY_THRESHOLD",
        CONFIG["evaluation"]["quality_threshold"],
    )
)

WARNING_THRESHOLD = float(
    os.getenv(
        "WARNING_THRESHOLD",
        CONFIG["evaluation"]["review_threshold"],
    )
)

TEST_CASE_LIMIT = int(
    os.getenv("TEST_CASE_LIMIT", CONFIG["execution"]["test_case_limit"])
)
if TEST_CASE_LIMIT < 1:
    raise ValueError("test_case_limit must be at least 1")

MINIMUM_TESTCASES_FOR_RANKING = int(
    os.getenv(
        "MINIMUM_TESTCASES_FOR_RANKING",
        CONFIG["execution"].get("minimum_testcases_for_ranking", 10),
    )
)

if RETRIES < 0 or RETRY_BACKOFF_SECONDS < 0:
    raise ValueError("RETRIES and RETRY_BACKOFF_SECONDS must be >= 0")
if MINIMUM_TESTCASES_FOR_RANKING < 1:
    raise ValueError("MINIMUM_TESTCASES_FOR_RANKING must be at least 1")
if RUNTIME_TARGET_SECONDS <= 0:
    raise ValueError("RUNTIME_TARGET_SECONDS must be > 0")
for _name, _value in (
    ("GENERATOR_CONCURRENCY", GENERATOR_CONCURRENCY),
    ("JUDGE_CONCURRENCY", JUDGE_CONCURRENCY),
    ("EVALUATION_CONCURRENCY", EVALUATION_CONCURRENCY),
):
    if _value < 1:
        raise ValueError(f"{_name} must be at least 1")

METRIC_THRESHOLDS = CONFIG["evaluation"]["metric_thresholds"]

PREFERRED_GENERATORS = CONFIG["models"]["generator_preferences"]
PREFERRED_JUDGES = CONFIG["models"]["judge_preferences"]
EXCLUDED_MODELS = tuple(CONFIG["models"]["excluded_patterns"])

GENERATOR_COUNT = int(CONFIG["models"]["generator_count"])
JUDGE_COUNT = int(CONFIG["models"]["judge_count"])

_VALID_JUDGE_PROVIDERS = {"ollama", "gemini", "anthropic"}
JUDGE_PROVIDER = os.getenv(
    "JUDGE_PROVIDER",
    CONFIG["models"].get("judge_provider", "ollama"),
).strip().lower()
if JUDGE_PROVIDER not in _VALID_JUDGE_PROVIDERS:
    raise ValueError(
        "judge_provider must be one of: "
        + ", ".join(sorted(_VALID_JUDGE_PROVIDERS))
    )

GEMINI_JUDGE_MODEL = os.getenv(
    "GEMINI_JUDGE_MODEL",
    CONFIG["models"].get("gemini_judge_model", "gemini-3.6-flash"),
)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

ANTHROPIC_JUDGE_MODEL = os.getenv(
    "ANTHROPIC_JUDGE_MODEL",
    CONFIG["models"].get("anthropic_judge_model", "claude-haiku-4-5-20251001"),
)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

METRIC_NAMES = tuple(
    item["name"]
    for item in CONFIG["evaluation"]["metrics"]
)

METRIC_ROLES = {
    item["name"]: item["purpose"]
    for item in CONFIG["evaluation"]["metrics"]
}

METRIC_WEIGHTS = CONFIG["evaluation"]["metric_weights"]
if set(METRIC_WEIGHTS) != set(METRIC_NAMES):
    raise ValueError("metric_weights must define exactly the configured metrics")
if any(float(weight) < 0 for weight in METRIC_WEIGHTS.values()):
    raise ValueError("metric_weights cannot contain negative values")
if sum(float(weight) for weight in METRIC_WEIGHTS.values()) <= 0:
    raise ValueError("metric_weights must contain at least one positive weight")
