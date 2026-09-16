"""Run provenance and reproducibility tracking.

Governance concern: answers "exactly what produced this report" after the
fact - which code run (run_id correlating the response workbook and the
report), which dataset/prompt content (hashed, not just a file path that
could have changed since), which package/interpreter versions, and which
exact model weights (Ollama digest, not just a tag that can be re-pulled to
different weights under the same name).
"""
from datetime import datetime
from pathlib import Path
import hashlib
import importlib.metadata
import platform


def new_run_id():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value):
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def package_version(package_name):
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def python_version():
    return platform.python_version()


def model_digests(model_names, server_models):
    """Format a "name: digest" line per model for the report.

    server_models is discovery's server_models list (name/digest/size_gb
    per model actually seen on the Ollama server this run).
    """
    digest_by_name = {
        item["name"]: item.get("digest", "")
        for item in server_models
    }
    return " | ".join(
        f"{name}: {digest_by_name.get(name, 'N/A') or 'N/A'}"
        for name in model_names
    )
