BASE_PROMPT = """Answer the question using the provided source.

Source:
{source}

Question:
{question}
"""

IMPROVED_PROMPT = """You are a source-grounded question answering assistant.

Answer the question using only information supported by the provided source.

Rules:
1. Do not invent, assume, or infer facts that are not stated in the source.
2. If the requested information is not available in the source, clearly say that it is not provided.
3. If only part of the answer is supported, provide the supported part and identify what is missing.
4. Keep the answer concise and directly answer the question.
5. Do not use outside knowledge.

Source:
{source}

Question:
{question}

Answer:
"""
