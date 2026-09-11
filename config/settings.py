from pathlib import Path
import json
import os

from dotenv import load_dotenv


load_dotenv()

os.environ.setdefault(
    "DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE",
    "180",
)


ROOT = Path(__file__).resolve().parents[1]

with (Path(__file__).with_name("project_config.json")).open(
    "r",
    encoding="utf-8",
) as file:
    CONFIG = json.load(file)


DATASET_PATH = ROOT / "dataset" / "hallucination_benchmark.xlsx"
REPORT_ROOT = ROOT / "reports"

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

RETRIES = int(
    os.getenv(
        "RETRIES",
        CONFIG["execution"]["retries"],
    )
)

TEST_CASE_LIMIT = int(
    CONFIG["execution"].get("test_case_limit", 0)
)

TEMPERATURE = float(
    os.getenv(
        "TEMPERATURE",
        CONFIG["trajectory"]["temperature"],
    )
)

TRAJECTORY_THRESHOLD = float(
    os.getenv(
        "TRAJECTORY_THRESHOLD",
        CONFIG["trajectory"]["threshold"],
    )
)

PREFERRED_GENERATORS = CONFIG["models"]["generator_preferences"]
PREFERRED_JUDGES = CONFIG["models"]["judge_preferences"]
EXCLUDED_MODELS = tuple(CONFIG["models"]["excluded_patterns"])
GENERATOR_COUNT = int(CONFIG["models"]["generator_count"])
TRAJECTORY_METRICS = CONFIG["trajectory"]["metrics"]

PROJECT_1_REPORT_ROOT = (
    ROOT / CONFIG["project_1_report_root"]
).resolve()

WORKFLOW = CONFIG["workflow"]