import statistics
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from data import DocumentProcessingError, extract_text, find_input_files, hash_text, load_expected_answer
from evaluation import EvaluationError, evaluate_response
from execution import RoutingError, generate_response, route_request
from execution.clients import GeminiClient, GeminiError, OllamaClient, OllamaError
from governance import (
    GuardrailError,
    ReportingError,
    append_call_ledger,
    make_final_decision,
    run_guardrail,
    write_audit_log,
    write_excel_report,
)
from settings import Config, ConfigError, configure_logging


def build_clients(config, logger):
    """Create model clients once and reuse them during the run."""
    ollama_client = OllamaClient(
        base_url=config.get("ollama", "base_url"),
        timeout_seconds=config.get("ollama", "timeout_seconds"),
        logger=logger,
        keep_alive=config.get("ollama", "keep_alive", default="30m"),
    )

    gemini_client = GeminiClient(
        api_url=config.get("gemini", "api_url"),
        api_key=config.gemini_api_key(),
        timeout_seconds=config.get("gemini", "timeout_seconds"),
        logger=logger,
    )

    return ollama_client, gemini_client


def process_file(file_path, config, ollama_client, gemini_client, logger, run_id):
    """Run one input document through the complete data -> execution -> evaluation -> governance flow.

    Returns (record, calls) where `record` is the audit/report row for this
    document and `calls` is the list of individual LLM call timings (for the
    call-timing ledger), each already tagged with run_id and input_file.
    """
    started = time.perf_counter()
    input_file = Path(file_path).name
    base = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_file": input_file,
    }
    calls: list[dict] = []

    try:
        extraction_started = time.perf_counter()
        document_text = extract_text(
            file_path,
            int(config.get("execution", "max_document_chars", default=50000)),
        )
        extraction_latency = round(time.perf_counter() - extraction_started, 4)

        route = route_request(document_text, config)

        logger.info(
            "Processing '%s' | intent=%s | model_key=%s | reason=%s",
            input_file,
            route.intent,
            route.model_key,
            route.reason,
        )

        # An ambiguous route is a governance decision, not a quality one:
        # skip generation entirely rather than spend an LLM call on a request
        # the router itself can't confidently classify.
        if route.intent == "MANUAL_REVIEW":
            record = {
                **base,
                "intent": route.intent,
                "route_reason": route.reason,
                "model": "",
                "generation_provider": "",
                "generation_latency_seconds": "",
                "guardrail_decision": "NOT_RUN",
                "guardrail_risk": "MEDIUM",
                "guardrail_source": "NOT_RUN",
                "guardrail_latency_seconds": "",
                "evaluation_verdict": "NOT_RUN",
                "evaluation_score": "",
                "evaluation_latency_seconds": "",
                "total_latency_seconds": round(time.perf_counter() - started, 4),
                "final_decision": "MANUAL_REVIEW",
                "status": "COMPLETED",
                "failure_type": "ROUTING_AMBIGUITY",
                "reason": route.reason,
                "input_hash": hash_text(document_text),
                "extraction_latency_seconds": extraction_latency,
            }
            return record, calls

        response, model, provider, generation_calls = generate_response(
            route,
            document_text,
            config,
            ollama_client,
            gemini_client,
            logger,
        )
        generation_latency = generation_calls[-1]["duration_seconds"]
        calls.extend(_tag(call, run_id, input_file) for call in generation_calls)

        guardrail_started_at = datetime.now(timezone.utc).isoformat()
        guardrail_started = time.perf_counter()
        guardrail = run_guardrail(
            document_text=document_text,
            response=response,
            intent=route.intent,
            ollama_client=ollama_client,
            model=config.get("models", "guardrail"),
            config=config,
            logger=logger,
        )
        guardrail_latency = round(time.perf_counter() - guardrail_started, 4)
        if guardrail.source == "LLM":
            calls.append(
                _tag(
                    {
                        "phase": "guardrail",
                        "provider": "ollama",
                        "model": config.get("models", "guardrail"),
                        "started_at": guardrail_started_at,
                        "duration_seconds": guardrail_latency,
                        "outcome": "success",
                    },
                    run_id,
                    input_file,
                )
            )

        # A blocked response does not need a quality score.
        if guardrail.decision == "BLOCK":
            evaluation_verdict = "NOT_RUN"
            evaluation_score = ""
            evaluation_latency = ""
            reason = guardrail.reason
        else:
            evaluation_started_at = datetime.now(timezone.utc).isoformat()
            expected_answer = load_expected_answer(file_path)
            evaluation = evaluate_response(
                document_text=document_text,
                response=response,
                intent=route.intent,
                ollama_client=ollama_client,
                model=config.get("models", "evaluation"),
                config=config,
                logger=logger,
                expected_answer=expected_answer,
            )
            evaluation_verdict = evaluation.verdict
            evaluation_score = evaluation.overall
            evaluation_latency = evaluation.duration_seconds
            reason = evaluation.reason
            calls.append(
                _tag(
                    {
                        "phase": "evaluation",
                        "provider": "ollama",
                        "model": config.get("models", "evaluation"),
                        "started_at": evaluation_started_at,
                        "duration_seconds": evaluation_latency,
                        "outcome": "success",
                    },
                    run_id,
                    input_file,
                )
            )

        final_decision = make_final_decision(guardrail.decision, evaluation_verdict)

        record = {
            **base,
            "intent": route.intent,
            "route_reason": route.reason,
            "model": model,
            "generation_provider": provider,
            "generation_latency_seconds": generation_latency,
            "guardrail_decision": guardrail.decision,
            "guardrail_risk": guardrail.risk,
            "guardrail_source": guardrail.source,
            "guardrail_latency_seconds": guardrail_latency,
            "evaluation_verdict": evaluation_verdict,
            "evaluation_score": evaluation_score,
            "evaluation_latency_seconds": evaluation_latency,
            "total_latency_seconds": round(time.perf_counter() - started, 4),
            "final_decision": final_decision,
            "status": "COMPLETED",
            "failure_type": "",
            "reason": reason,
            "input_hash": hash_text(document_text),
            "response_hash": hash_text(response),
            "extraction_latency_seconds": extraction_latency,
        }
        return record, calls

    except (
        DocumentProcessingError,
        RoutingError,
        OllamaError,
        GeminiError,
        GuardrailError,
        EvaluationError,
    ) as exc:
        # A technical/infrastructure failure is not a quality or policy verdict:
        # keep it out of PASS/BLOCK/MANUAL_REVIEW so it can't be mistaken for one.
        logger.exception("Processing failed for '%s'", file_path)
        calls.extend(_tag(call, run_id, input_file) for call in getattr(exc, "calls", []))
        record = {
            **base,
            "intent": "ERROR",
            "route_reason": "",
            "model": "",
            "generation_provider": "",
            "generation_latency_seconds": "",
            "guardrail_decision": "ERROR",
            "guardrail_risk": "HIGH",
            "guardrail_source": "ERROR",
            "guardrail_latency_seconds": "",
            "evaluation_verdict": "ERROR",
            "evaluation_score": "",
            "evaluation_latency_seconds": "",
            "total_latency_seconds": round(time.perf_counter() - started, 4),
            "final_decision": "TECHNICAL_ERROR",
            "status": "TECHNICAL_ERROR",
            "failure_type": "TECHNICAL_ERROR",
            "reason": str(exc),
        }
        return record, calls


def _tag(call: dict, run_id: str, input_file: str) -> dict:
    return {**call, "run_id": run_id, "input_file": input_file}


def _print_timing_summary(calls: list[dict]) -> None:
    """Print a per-phase LLM call timing table so run-to-run speed can be tracked at a glance."""
    if not calls:
        return

    by_phase: dict[str, list[float]] = {}
    for call in calls:
        if call["outcome"] != "success":
            continue
        by_phase.setdefault(call["phase"], []).append(float(call["duration_seconds"]))

    header = f"{'Phase':<20}{'Calls':>7}{'Min (s)':>10}{'Avg (s)':>10}{'P50 (s)':>10}{'Max (s)':>10}"
    separator = "-" * len(header)
    lines = [header, separator]
    for phase, durations in sorted(by_phase.items()):
        lines.append(
            f"{phase:<20}{len(durations):>7}"
            f"{min(durations):>10.2f}{statistics.mean(durations):>10.2f}"
            f"{statistics.median(durations):>10.2f}{max(durations):>10.2f}"
        )

    print("\nLLM call timing summary (this run):")
    print("\n".join(lines))


def main() -> int:
    """Orchestrate the POC across the data, execution, evaluation and governance layers."""
    try:
        config = Config()
        logger = configure_logging(
            config.get("logging", "file"),
            config.get("logging", "level", default="INFO"),
        )

        ollama_client, gemini_client = build_clients(config, logger)
        input_files = find_input_files("input")

        if not input_files:
            logger.error("No .docx or .pptx files found in the input folder.")
            return 1

        run_id = uuid.uuid4().hex[:12]
        run_started = time.perf_counter()
        concurrency = int(config.get("execution", "file_concurrency", default=1))

        if concurrency <= 1:
            results = [
                process_file(file_path, config, ollama_client, gemini_client, logger, run_id)
                for file_path in input_files
            ]
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = [
                    pool.submit(
                        process_file,
                        file_path,
                        config,
                        ollama_client,
                        gemini_client,
                        logger,
                        run_id,
                    )
                    for file_path in input_files
                ]
                results = [future.result() for future in as_completed(futures)]

        records = [record for record, _ in results]
        calls = [call for _, file_calls in results for call in file_calls]

        run_metadata = {
            "run_id": run_id,
            "config_version": config.version(),
            "total_requests": len(records),
            "total_run_latency_seconds": round(time.perf_counter() - run_started, 4),
            "file_concurrency": concurrency,
            "generation_models": ", ".join(
                sorted({record.get("model", "") for record in records if record.get("model")})
            ),
            "guardrail_model": config.get("models", "guardrail"),
            "evaluation_model": config.get("models", "evaluation"),
            "gemini_model": config.gemini_model(),
            "general_fallback_model": config.get("models", "general_fallback", default=""),
        }

        write_audit_log(records, config.get("reporting", "audit_file"))
        write_excel_report(records, config.get("reporting", "excel_file"), run_metadata, calls)
        append_call_ledger(calls, config.get("reporting", "call_ledger_file"))

        logger.info(
            "POC completed run_id=%s processed=%d total=%.4fs",
            run_id,
            len(records),
            run_metadata["total_run_latency_seconds"],
        )
        _print_timing_summary(calls)
        return 0

    except (ConfigError, GeminiError, ReportingError) as exc:
        print(f"POC failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Unexpected POC failure: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
