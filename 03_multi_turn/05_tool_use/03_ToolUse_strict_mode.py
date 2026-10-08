from basemodel import CustomOpenAI
from deepeval.test_case import Turn, ConversationalTestCase, ToolCall
from deepeval.metrics import ToolUseMetric

# ==========================================
# 1. THE METRIC (strict mode)
# ==========================================
# `strict_mode=True` collapses the score to binary:
#   - 1 if both tool selection AND arguments are perfect
#   - 0 otherwise
# It also overrides `threshold` to 1, so is_successful()
# matches the binary score exactly. Use it when any
# wrong tool or wrong argument is unacceptable
# (compliance, medical, tightly-scoped agents).

available_tools = [
    ToolCall(
        name="get_weather",
        description="Look up the current weather for a city.",
    ),
    ToolCall(
        name="get_news",
        description="Look up recent news headlines for a topic.",
    ),
]

strict_tool_use = ToolUseMetric(
    available_tools=available_tools,
    threshold=0.5,                      # ignored in strict mode
    model=CustomOpenAI(),
    include_reason=True,
    strict_mode=True,
)

# ==========================================
# 2. RIGHT TOOL + RIGHT ARG — expect score 1
# ==========================================
# Perfect on both halves of the min().

perfect_case = ConversationalTestCase(
    turns=[
        Turn(role="user", content="What's the current weather in Paris?"),
        Turn(
            role="assistant",
            content="It's 18°C and partly cloudy in Paris right now.",
            tools_called=[
                ToolCall(
                    name="get_weather",
                    description="Look up the current weather for a city.",
                    input_parameters={"city": "Paris"},
                    output={"temp_c": 18, "condition": "Partly cloudy"},
                )
            ],
        ),
    ]
)

strict_tool_use.measure(perfect_case)
print("--- right tool, right args ---")
print(f"Score:   {strict_tool_use.score}")          # 1.0
print(f"Reason:  {strict_tool_use.reason}")
print(f"Passed:  {strict_tool_use.is_successful()}")  # True

# ==========================================
# 3. RIGHT TOOL + WRONG ARG — expect score 0
# ==========================================
# The agent picks `get_weather` (right tool) but passes
# the wrong city (London instead of Paris). Selection
# is perfect; arguments are bad. min(1.0, ~0.0) ≈ 0.
# Strict mode drops the score to 0. This is the key
# behavior the min() captures — both halves matter.

wrong_arg_case = ConversationalTestCase(
    turns=[
        Turn(role="user", content="What's the current weather in Paris?"),
        Turn(
            role="assistant",
            content="It's 9°C and raining — bring a coat.",
            tools_called=[
                ToolCall(
                    name="get_weather",
                    description="Look up the current weather for a city.",
                    input_parameters={"city": "London"},                       # wrong arg
                    output={"temp_c": 9, "condition": "Rain"},
                )
            ],
        ),
    ]
)

strict_tool_use.measure(wrong_arg_case)
print("--- right tool, wrong arg ---")
print(f"Score:   {strict_tool_use.score}")          # 0.0
print(f"Reason:  {strict_tool_use.reason}")
print(f"Passed:  {strict_tool_use.is_successful()}")  # False