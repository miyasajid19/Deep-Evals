from basemodel import CustomOpenAI
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval import evaluate
from rich import print

# ==========================================
# 1. DEFINE THE METRIC (default behaviour)
# ==========================================
# By default, ToolCorrectnessMetric is DETERMINISTIC — it just compares
# tool NAMES between `tools_called` and `expected_tools`. No LLM is
# called unless you also pass `available_tools` (see example 02).
# Equation: correct_named_tools / total_tools_called.

metric = ToolCorrectnessMetric(model=CustomOpenAI())

# ==========================================
# 2. BUILD LLMTestCases WITH tools_called + expected_tools
# ==========================================
test_cases = [
    # Perfect match: every tool the agent called is in expected_tools
    # and every expected tool was called. Score = 1.0.
    LLMTestCase(
        input="What's the weather in Paris?",
        actual_output="It's 18°C and partly cloudy in Paris.",
        tools_called=[
            ToolCall(name="GetWeather"),
        ],
        expected_tools=[
            ToolCall(name="GetWeather"),
        ],
    ),
    # Extra tool called: the agent invoked an unneeded tool. Score
    # drops because not every tools_called matches expected_tools.
    LLMTestCase(
        input="What's the weather in Paris?",
        actual_output="It's 18°C in Paris.",
        tools_called=[
            ToolCall(name="GetWeather"),
            ToolCall(name="GetLocalTime"),  # not in expected_tools
        ],
        expected_tools=[
            ToolCall(name="GetWeather"),
        ],
    ),
    # Wrong tool name: agent called something entirely different.
    LLMTestCase(
        input="What's the weather in Paris?",
        actual_output="Here's a fun fact about Paris.",
        tools_called=[
            ToolCall(name="GetFunFact"),
        ],
        expected_tools=[
            ToolCall(name="GetWeather"),
        ],
    ),
]

# ==========================================
# 3. RUN END-TO-END EVALUATION
# ==========================================
evaluate(test_cases=test_cases, metrics=[metric])