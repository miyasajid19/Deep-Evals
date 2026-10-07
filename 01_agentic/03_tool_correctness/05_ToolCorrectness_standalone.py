from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall
from rich import print
from basemodel import CustomOpenAI
# ==========================================
# 1. DEFINE THE METRIC
# ==========================================
# Standalone execution is useful for debugging or building a custom
# pipeline, but it does NOT include the testing reports, Confident AI
# integration, or speed / caching that `evaluate()` provides.

metric = ToolCorrectnessMetric(
    should_consider_ordering=True,
    include_reason=True,
    model=CustomOpenAI(),
)

# ==========================================
# 2. BUILD A SINGLE LLMTestCase
# ==========================================
test_case = LLMTestCase(
    input="What's the weather in Tokyo and what time is it there?",
    actual_output="It's 25°C and sunny in Tokyo; the local time is 14:32.",
    tools_called=[
        ToolCall(name="GetWeather"),
        ToolCall(name="GetLocalTime"),
    ],
    expected_tools=[
        ToolCall(name="GetWeather"),
        ToolCall(name="GetLocalTime"),
    ],
)

# ==========================================
# 3. MEASURE ON A SINGLE TEST CASE
# ==========================================
metric.measure(test_case)
print(f"Score:   {metric.score}")
print(f"Reason:  {metric.reason}")
print(f"Passed:  {metric.is_successful()}")