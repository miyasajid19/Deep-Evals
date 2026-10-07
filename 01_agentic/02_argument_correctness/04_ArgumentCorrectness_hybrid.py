from basemodel import CustomOpenAI
from deepeval.metrics import ArgumentCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval import evaluate
from rich import print

# ==========================================
# 1. DEFINE THE METRIC IN HYBRID MODE
# ==========================================
# `eval_mode="hybrid"` keeps the LLM in the loop for extraction but
# delegates the per-tool-call decision to Jev (System One). The score
# equation and the reason are unchanged — only the judge changes.
# If a Jev call fails, the metric falls back to the LLM.

metric = ArgumentCorrectnessMetric(
    threshold=0.7,
    eval_mode="hybrid",
    model=CustomOpenAI(),
    system_one_model="jev-latest",
    include_reason=True,
)

# ==========================================
# 2. BUILD LLMTestCases WITH tools_called
# ==========================================
test_cases = [
    # All arguments fit the input → Jev should mark both correct.
    LLMTestCase(
        input="What's the weather in Paris and what time is it there?",
        actual_output="It's 18°C and partly cloudy in Paris; it's 14:32 local time.",
        tools_called=[
            ToolCall(
                name="GetWeather",
                description="Get the current weather for a given city.",
                input={"city": "Paris", "unit": "celsius"},
            ),
            ToolCall(
                name="GetLocalTime",
                description="Get the current local time for a given city.",
                input={"city": "Paris"},
            ),
        ],
    ),
    # Argument for the second tool is wrong (Berlin instead of Paris).
    LLMTestCase(
        input="What's the weather in Paris and what time is it there?",
        actual_output="It's 18°C and partly cloudy in Paris; it's 14:32 in Berlin.",
        tools_called=[
            ToolCall(
                name="GetWeather",
                description="Get the current weather for a given city.",
                input={"city": "Paris", "unit": "celsius"},
            ),
            ToolCall(
                name="GetLocalTime",
                description="Get the current local time for a given city.",
                input={"city": "Berlin"},  # wrong city
            ),
        ],
    ),
]

# ==========================================
# 3. RUN END-TO-END EVALUATION
# ==========================================
evaluate(test_cases=test_cases, metrics=[metric])