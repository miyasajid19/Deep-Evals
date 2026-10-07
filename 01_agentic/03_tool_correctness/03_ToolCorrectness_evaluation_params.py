from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.metrics.tool_correctness import ToolCallParams
from deepeval import evaluate
from rich import print

# ==========================================
# 1. DEFINE THE METRIC WITH evaluation_params
# ==========================================
# By default, ToolCorrectnessMetric only matches on tool NAME. Pass
# `evaluation_params` to also match on:
#   - ToolCallParams.INPUT_PARAMETERS: input args must match
#   - ToolCallParams.OUTPUT:           output must match
# With INPUT_PARAMETERS, the score is the proportion of correct
# parameters (name + output still required to be correct).

# Inputs-only: name + input params must match
metric_inputs = ToolCorrectnessMetric(
    evaluation_params=[ToolCallParams.INPUT_PARAMETERS],
)

# Full: name + input params + output must all match
metric_full = ToolCorrectnessMetric(
    evaluation_params=[ToolCallParams.INPUT_PARAMETERS, ToolCallParams.OUTPUT],
)

# ==========================================
# 2. BUILD LLMTestCases
# ==========================================
test_cases = [
    # All three checks agree: name, input, and output match.
    LLMTestCase(
        input="What's the weather in Paris?",
        actual_output="It's 18°C and partly cloudy in Paris.",
        tools_called=[
            ToolCall(
                name="GetWeather",
                input_parameters={"city": "Paris", "unit": "celsius"},
                output={"temp_c": 18, "condition": "partly cloudy"},
            ),
        ],
        expected_tools=[
            ToolCall(
                name="GetWeather",
                input_parameters={"city": "Paris", "unit": "celsius"},
                output={"temp_c": 18, "condition": "partly cloudy"},
            ),
        ],
    ),
    # Name matches but input_parameters differ (wrong city).
    LLMTestCase(
        input="What's the weather in Paris?",
        actual_output="It's 22°C in Berlin.",
        tools_called=[
            ToolCall(
                name="GetWeather",
                input_parameters={"city": "Berlin", "unit": "celsius"},
                output={"temp_c": 22, "condition": "sunny"},
            ),
        ],
        expected_tools=[
            ToolCall(
                name="GetWeather",
                input_parameters={"city": "Paris", "unit": "celsius"},
                output={"temp_c": 18, "condition": "partly cloudy"},
            ),
        ],
    ),
]

# ==========================================
# 3. RUN END-TO-END EVALUATION
# ==========================================
# Both metrics are deterministic; the inputs-only one ignores output,
# the full one insists on name + input + output.
evaluate(test_cases=test_cases, metrics=[metric_inputs, metric_full])

# not working