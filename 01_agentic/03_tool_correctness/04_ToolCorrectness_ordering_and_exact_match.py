from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval import evaluate
from rich import print
from basemodel import CustomOpenAI
# ==========================================
# 1. DEFINE THREE FLAVOURS OF STRICTNESS
# ==========================================
# (a) Default: any-order, subset match. tools_called = [A, B, A] is OK
#     as long as expected_tools ⊆ tools_called.
# (b) should_consider_ordering=True: the call sequence must match
#     expected_tools in order.
# (c) should_exact_match=True: tools_called and expected_tools must
#     be IDENTICAL (counts included). Overrides should_consider_ordering.

metric_any_order = ToolCorrectnessMetric(model=CustomOpenAI())  # default behavior

metric_ordered = ToolCorrectnessMetric(model=CustomOpenAI(), should_consider_ordering=True)

metric_exact = ToolCorrectnessMetric(model=CustomOpenAI(), should_exact_match=True)

# ==========================================
# 2. BUILD LLMTestCases
# ==========================================
test_cases = [
    # Sequence matches → all three should pass.
    LLMTestCase(
        input="Look up weather, then time, then weather again.",
        actual_output="18°C, 14:32, still 18°C.",
        tools_called=[
            ToolCall(name="GetWeather"),
            ToolCall(name="GetLocalTime"),
            ToolCall(name="GetWeather"),
        ],
        expected_tools=[
            ToolCall(name="GetWeather"),
            ToolCall(name="GetLocalTime"),
            ToolCall(name="GetWeather"),
        ],
    ),
    # Out-of-order call. Default passes (any order), ordered fails
    # (sequence is wrong), exact passes (lists still match).
    LLMTestCase(
        input="Look up weather, then time.",
        actual_output="18°C, 14:32.",
        tools_called=[
            ToolCall(name="GetLocalTime"),
            ToolCall(name="GetWeather"),
        ],
        expected_tools=[
            ToolCall(name="GetWeather"),
            ToolCall(name="GetLocalTime"),
        ],
    ),
    # One extra call. Default may still pass (subset match), exact
    # must fail (lists differ in count).
    LLMTestCase(
        input="Look up weather.",
        actual_output="18°C.",
        tools_called=[
            ToolCall(name="GetWeather"),
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
# Compare how each strictness flavour scores the same three test cases.
evaluate(
    test_cases=test_cases,
    metrics=[metric_any_order, metric_ordered, metric_exact],
)