import requests

from deepeval.test_case import LLMTestCase
from deepeval.tracing import observe, update_current_span, update_current_trace

from config.settings import (
    OLLAMA_BASE_URL,
    REQUEST_TIMEOUT,
    TEMPERATURE,
    WORKFLOW,
)


@observe(type="llm")
def generator_step(model, question, source, expected_output):
    prompt = WORKFLOW["prompt_template"].format(
        source=source,
        question=question,
    )

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": TEMPERATURE,
            },
        },
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    output = response.json()["response"].strip()

    update_current_span(
        test_case=LLMTestCase(
            input=question,
            actual_output=output,
            expected_output=expected_output,
            context=[source],
            retrieval_context=[source],
        )
    )

    return output


@observe(type="agent")
def multi_generator_agent(
    question,
    source,
    expected_output,
    generators,
    preferred_generator=None,
):
    plan = (
        "Plan: "
        f"1. Run {generators[0]} against the supplied source and question. "
        f"2. Run {generators[1]} against the supplied source and question. "
        f"3. Run {generators[2]} against the supplied source and question. "
        "4. Compare the three generator outputs. "
        "5. Select the generator preferred by Project 1. "
        "6. Return the selected generator response."
    )

    update_current_trace(
        name="Multi-generator model selection",
        input=question,
        context=[source],
        expected_output=expected_output,
        metadata={
            "selection_policy": plan,
            "configured_generators": generators,
        },
    )

    # Explicitly expose the plan in the trace.
    update_current_span(
        name="Execution plan",
        input=question,
        output=plan,
    )

    outputs = []

    for model in generators:
        output = generator_step(
            model,
            question,
            source,
            expected_output,
        )

        outputs.append(
            {
                "model": model,
                "response": output,
            }
        )

    selected = next(
        (
            item
            for item in outputs
            if item["model"] == preferred_generator
        ),
        outputs[0],
    )

    selection_note = (
        f"Compared {len(outputs)} generator outputs. "
        f"Project 1 preferred generator: "
        f"{preferred_generator or 'not available'}. "
        f"Selected: {selected['model']}."
    )

    update_current_span(
        name="Generator selection",
        input=question,
        output=selection_note,
    )

    final = selected["response"]

    update_current_trace(
        output=final,
        expected_output=expected_output,
    )

    return final, outputs, selected["model"]