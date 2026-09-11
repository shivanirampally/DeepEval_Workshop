BASE_PROMPT = """Answer the question using only the provided source.

Rules:
- Use only information supported by the source.
- If the source does not contain the answer, clearly say that the information is not provided.
- Do not invent facts or make unsupported assumptions.
- Answer the question directly and concisely.

Source:
{source}

Question:
{question}
"""

OPTIMIZER_PROMPT = """You are improving an existing prompt for a source-grounded question answering system.

Original prompt:
{original_prompt}

Observed V1 failure analysis:
{failure_analysis}

Human-reviewed synthetic test scenarios:
{synthetic_context}

Create ONE improved prompt for the same task.

Requirements:
- Preserve the original task.
- Preserve the placeholders {source} and {question}.
- Directly address the observed failure patterns.
- Keep source grounding explicit.
- Require the model to state when information is missing instead of guessing.
- Do not add unrelated requirements.
- Do not use outside knowledge.
- Return only the improved prompt text.
"""
