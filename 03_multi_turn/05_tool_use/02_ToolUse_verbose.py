from basemodel import CustomOpenAI
from deepeval.test_case import Turn, ConversationalTestCase, ToolCall
from deepeval.metrics import ToolUseMetric

# ==========================================
# 1. THE METRIC (verbose)
# ==========================================
# `verbose_mode=True` prints the intermediate steps:
#   - the tool selection score for each interaction
#   - the argument correctness score for each tool call
#   - the final min(selection, args) score
# This is the fastest way to debug "the agent picked the
# right tool but the result was wrong" — verbose mode
# tells you whether selection or arguments cost you
# points.

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

tool_use = ToolUseMetric(
    available_tools=available_tools,
    threshold=0.5,
    model=CustomOpenAI(),
    include_reason=True,
    verbose_mode=True,
)

# ==========================================
# 2. THE TEST CASE — wrong tool choice
# ==========================================
# The user asks for the weather in Paris. The agent
# picks `get_news` (wrong selection). Arguments to
# `get_news` are fine, but selection is bad, so the
# MIN score is low.

convo_test_case = ConversationalTestCase(
    turns=[
        Turn(role="user", content="What's the current weather in Paris?"),
        Turn(
            role="assistant",
            content="Top headline: Paris hosts a major climate summit this week.",
            tools_called=[
                ToolCall(
                    name="get_news",                                       # wrong tool
                    description="Look up recent news headlines for a topic.",
                    input_parameters={"topic": "Paris"},
                    output={"headline": "Paris hosts a major climate summit this week."},
                )
            ],
        ),
    ]
)

# ==========================================
# 3. RUN STANDALONE
# ==========================================
# Standalone prints metric.score and metric.reason directly,
# skipping the report aggregation that `evaluate()` does.
# Verbose output prints above the score.

tool_use.measure(convo_test_case)
print("---")
print(f"Score:   {tool_use.score}")
print(f"Reason:  {tool_use.reason}")
print(f"Passed:  {tool_use.is_successful()}")