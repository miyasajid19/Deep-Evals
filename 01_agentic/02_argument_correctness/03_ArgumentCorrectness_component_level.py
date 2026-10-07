from basemodel import CustomOpenAI
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.tracing import observe, update_current_span
from deepeval.metrics import ArgumentCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall

# ==========================================
# 1. DEFINE THE METRIC
# ==========================================
# When attached to an @observe component, the metric is scored against
# the test case passed to `update_current_span(test_case=...)`. The
# component can be a tool call, a sub-agent, an LLM step, anything.

metric = ArgumentCorrectnessMetric(
    threshold=0.7,
    model=CustomOpenAI(),
    include_reason=True,
)

# ==========================================
# 2. SIMULATED LLM AGENT
# ==========================================
def call_llm(input: str) -> str:
    """Stand-in for an actual LLM call. Returns a canned answer."""
    return f"(mock response to: {input})"


# ==========================================
# 3. COMPONENT WITH @observe(metrics=[metric])
# ==========================================
@observe(metrics=[metric])
def inner_component(input: str):
    # Build the LLMTestCase at runtime so the metric sees what the
    # component actually did. tools_called captures the arguments the
    # agent used for the LLM.
    tools_called = [
        ToolCall(
            name="GetWeather",
            description="Get the current weather for a given city.",
            input_parameters={"city": "Paris", "unit": "celsius"},
        ),
    ]
    actual_output = call_llm(input)

    update_current_span(
        test_case=LLMTestCase(
            input=input,
            actual_output=actual_output,
            tools_called=tools_called,
        )
    )
    return actual_output


@observe()
def llm_app(input: str):
    return inner_component(input)


# ==========================================
# 4. BUILD A DATASET
# ==========================================
dataset = EvaluationDataset(
    goldens=[
        Golden(input="What's the weather in Paris?"),
        Golden(input="What's the weather in Tokyo?"),
        Golden(input="What's the weather in Berlin?"),
    ]
)

# ==========================================
# 5. EVALUATE OVER THE DATASET
# ==========================================
for golden in dataset.evals_iterator():
    llm_app(golden.input)