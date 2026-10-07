from deepeval.metrics import ArgumentCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval import evaluate
from rich import print

# ==========================================
# 1. DEFINE THE METRIC IN SYSTEM_ONE MODE
# ==========================================
# Jev judges the whole metric in one request. It is sent the raw input
# and the structured tools_called, and asked three questions:
#   1. Every tool call has input params that correctly address input  (Noul, weight 2)
#   2. No tool call has params that are missing/irrelevant/wrong        (Noul, weight 1)
#   3. How many tool calls have correct params? (None → All)            (Score, weight 1)
# Each answer becomes a value in [0, 1] and the score is the weighted
# mean. No LLM is called.

metric = ArgumentCorrectnessMetric(
    threshold=0.7,
    eval_mode="system_one",
    system_one_model="jev-latest",
    include_reason=True,
)

# ==========================================
# 2. BUILD LLMTestCases WITH tools_called
# ==========================================
test_cases = [
    # All arguments are on-target for the input.
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
    # Wrong arguments: Jev should mark the run down.
    LLMTestCase(
        input="What's the weather in Paris and what time is it there?",
        actual_output="It's 18°C in Berlin; it's 14:32 in Berlin.",
        tools_called=[
            ToolCall(
                name="GetWeather",
                description="Get the current weather for a given city.",
                input={"city": "Berlin", "unit": "celsius"},
            ),
            ToolCall(
                name="GetLocalTime",
                description="Get the current local time for a given city.",
                input={"city": "Berlin"},
            ),
        ],
    ),
]

# ==========================================
# 3. RUN END-TO-END EVALUATION
# ==========================================
# A `system_one` score comes from Jev's answers, not the LLM-chain
# equation. Keep a metric on one eval mode when comparing results
# across runs.
evaluate(test_cases=test_cases, metrics=[metric])