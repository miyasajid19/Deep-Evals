from basemodel import CustomOpenAI
from deepeval.test_case import Turn, ConversationalTestCase, ToolCall
from deepeval.metrics import ToolUseMetric
from deepeval import evaluate

# ==========================================
# 1. THE METRIC
# ==========================================
# ToolUseMetric evaluates an agent on TOOL SELECTION and
# ARGUMENT GENERATION. It's the right metric when you
# care about *which* tool the agent picked from a known
# menu, and whether the arguments it passed were right.
#
# Algorithm (default `llm` eval mode):
#   1) for each assistant turn, compute:
#        - Tool Selection Score       — did the agent pick
#                                        the most appropriate
#                                        tool from `available_tools`?
#        - Argument Correctness Score  — are the arguments
#                                        correct and complete?
#   2) score = min(Tool Selection, Argument Correctness)
#
# The MIN is the key: a perfect tool choice with wrong
# arguments still drags the score down.
#
# `available_tools` is MANDATORY — it's the full menu
# the agent could have chosen from, used to judge
# selection. The metric is `referenceless`: no
# `expected_output` needed.

# ==========================================
# 2. THE TOOL MENU (available_tools)
# ==========================================
# Two tools the agent could pick from: get_weather and
# get_news. Both are described so the metric can tell
# them apart.

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
    available_tools=available_tools,    # ← mandatory
    threshold=0.5,
    model=CustomOpenAI(),
    include_reason=True,
)

# ==========================================
# 3. THE TEST CASE
# ==========================================
# The user asks for the Paris weather. The agent picks
# `get_weather` (right choice) with "Paris" as the city
# argument (right argument). Expect score 1.0.

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
# 4. RUN THE EVAL
# ==========================================
# `evaluate()` runs the metric, prints results, and writes
# a report (Confident AI if configured, else local cache).

evaluate(test_cases=[convo_test_case], metrics=[tool_use])