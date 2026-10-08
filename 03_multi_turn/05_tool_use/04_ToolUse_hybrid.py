from basemodel import CustomOpenAI
from deepeval.test_case import Turn, ConversationalTestCase, ToolCall
from deepeval.metrics import ToolUseMetric

# ==========================================
# 1. THE METRIC (hybrid mode)
# ==========================================
# Hybrid for Tool Use works similarly to hybrid for
# Goal Accuracy: there's no LLM extraction step in the
# hybrid mode itself (the LLM extracts tool calls and
# arguments inline). What hybrid *does* change is the
# per-interaction scoring:
#
#   For each (tool selection, argument correctness) pair,
#   Jev (a System One model) rates each half on a
#   five-level scale mapped onto 0..1, with a short line
#   stating Jev's score and confidence as each score's
#   reason. The LLM still writes the FINAL reason.
#
# If a Jev call fails, the metric silently defers that
# decision to the LLM.

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

hybrid_tool_use = ToolUseMetric(
    available_tools=available_tools,
    threshold=0.5,
    model=CustomOpenAI(),       # LLM still writes the final reason
    include_reason=True,
    eval_mode="hybrid",         # Jev scores selection + args per turn
)

# ==========================================
# 2. THE TEST CASE
# ==========================================
# Same Paris-weather agent as the basic run — right
# tool, right argument.

convo_test_case = ConversationalTestCase(
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

# ==========================================
# 3. RUN STANDALONE
# ==========================================
# metric.confidence reports how confident the system_one
# model was on average across its per-turn calls.

hybrid_tool_use.measure(convo_test_case)
print("---")
print(f"Score:      {hybrid_tool_use.score}")
print(f"Confidence: {hybrid_tool_use.confidence}")
print(f"Reason:     {hybrid_tool_use.reason}")
print(f"Passed:     {hybrid_tool_use.is_successful()}")