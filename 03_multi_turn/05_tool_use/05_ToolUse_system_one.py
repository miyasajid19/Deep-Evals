from deepeval.test_case import Turn, ConversationalTestCase, ToolCall
from deepeval.metrics import ToolUseMetric

# ==========================================
# 1. THE METRIC (system_one / Jev-as-a-judge)
# ==========================================
# No LLM is called at all. Jev receives the whole
# conversation (with the tools called in each turn) AND
# your `available_tools`, and answers two questions:
#
#   Q1 (Noul, weight 1): every tool call in tools_called
#                        is the most suitable of
#                        available_tools for what the user
#                        asked — no unnecessary call, no
#                        better-suited tool ignored
#   Q2 (Noul, weight 1): every tool call passes correct,
#                        specific, complete arguments for
#                        the task and tool definition
#
# Final score = weighted mean of the two answers.
#
# NOTE: under `system_one` the two questions are
# AVERAGED, whereas the LLM chain takes the MIN of tool
# selection and argument correctness. Different scoring
# rules across modes — keep a metric on one eval mode
# when you compare results.

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

system_one_tool_use = ToolUseMetric(
    available_tools=available_tools,
    threshold=0.5,
    include_reason=True,
    eval_mode="system_one",
    # No `model=` needed — Jev runs the metric alone.
)

# ==========================================
# 2. THE TEST CASE
# ==========================================
# Same Paris-weather agent as the basic run.

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
# metric.confidence reflects how decisive Jev was. The
# reason lists each question's answer with its
# probability — useful when you want to know *why* Jev
# scored it that way, not just the final number.

system_one_tool_use.measure(convo_test_case)
print("---")
print(f"Score:      {system_one_tool_use.score}")
print(f"Confidence: {system_one_tool_use.confidence}")
print(f"Reason:     {system_one_tool_use.reason}")
print(f"Passed:     {system_one_tool_use.is_successful()}")