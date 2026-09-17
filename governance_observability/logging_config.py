from contextlib import contextmanager
from time import perf_counter
import logging
import sys


def configure_logging(log_path):
    # Redirected/piped output on Windows defaults to cp1252, which can't encode
    # some of the Unicode symbols DeepEval's own progress output prints -- force
    # UTF-8 so a redirected run (`python -m execution.main > out.txt`) can't crash on it.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except (ValueError, OSError):
                pass

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


@contextmanager
def log_llm_timing(label, model):
    """Time an LLM call and log it. Yields a dict that holds `elapsed` (seconds) once the call completes."""
    timing = {"elapsed": None}
    started = perf_counter()
    try:
        yield timing
    finally:
        timing["elapsed"] = round(perf_counter() - started, 3)
        logging.info("[LLM CALL] %s | model=%s | time_taken=%.3fs", label, model, timing["elapsed"])
