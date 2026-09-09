import requests

from deepeval.test_case import LLMTestCase
from deepeval.tracing import observe, update_current_span, update_current_trace

from config.settings import OLLAMA_BASE_URL, REQUEST_TIMEOUT, TEMPERATURE, WORKFLOW


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
            "options": {"temperature": TEMPERATURE},
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
    update_current_trace(
        name="Multi-generator model selection",
        input=question,
        context=[source],
        expected_output=expected_output,
        metadata={
            "selection_policy": WORKFLOW["selection_policy"],
            "configured_generators": generators,
        },
    )

    outputs = []
    for model in generators:
        output = generator_step(
            model,
            question,
            source,
            expected_output,
        )
        outputs.append({"model": model, "response": output})

    selected = next(
        (item for item in outputs if item["model"] == preferred_generator),
        outputs[0],
    )

    selection_note = (
        f"Compared {len(outputs)} generator outputs. "
        f"Project 1 preferred generator: {preferred_generator or 'not available'}. "
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
