import re
from dataclasses import dataclass


class RoutingError(Exception):
    """Raised when a request cannot be routed."""


@dataclass(frozen=True)
class Route:
    intent: str
    model_key: str
    reason: str


def route_request(text: str, config) -> Route:
    """Select a specialist route using word-boundary keyword matching."""
    if not text or not text.strip():
        raise RoutingError("Cannot route an empty request.")

    normalized = text.lower()

    sql_keywords = config.get("routing", "sql_keywords", default=[])
    coding_keywords = config.get("routing", "coding_keywords", default=[])

    is_sql = _contains_keyword(normalized, sql_keywords)
    is_coding = _contains_keyword(normalized, coding_keywords)

    if is_sql and is_coding:
        return Route(
            intent="MANUAL_REVIEW",
            model_key="general",
            reason="Ambiguous request matched both SQL and coding keywords."
        )

    if is_sql:
        return Route(
            intent="SQL",
            model_key="sql",
            reason="SQL/database keywords were found in the document."
        )

    if is_coding:
        return Route(
            intent="CODING",
            model_key="coding",
            reason="Coding/automation keywords were found in the document."
        )

    return Route(
        intent="GENERAL",
        model_key="general",
        reason="No specialist keyword matched, so the request uses the general model."
    )


def _contains_keyword(text: str, keywords: list[str]) -> bool:
    """Match keywords on word boundaries so substrings can't trigger false positives."""
    return any(
        re.search(rf"(?<![A-Za-z0-9_]){re.escape(keyword.lower())}(?![A-Za-z0-9_])", text)
        for keyword in keywords
    )
