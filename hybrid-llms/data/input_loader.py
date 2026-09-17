import hashlib
import json
from pathlib import Path


def find_input_files(input_dir: str) -> list[Path]:
    """Return supported input documents from the configured folder."""
    path = Path(input_dir)
    return sorted([*path.glob("*.docx"), *path.glob("*.pptx")])


def load_expected_answer(file_path: str) -> str:
    """Load an optional '<input-name>.json' sidecar with an expected_answer field."""
    sidecar = Path(file_path).with_suffix(".json")
    if not sidecar.exists():
        return ""

    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        return str(data.get("expected_answer", ""))
    except (OSError, json.JSONDecodeError):
        return ""


def hash_text(text: str) -> str:
    """Return a short content hash for audit correlation without storing raw text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
