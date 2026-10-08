from basemodel import CustomOpenAI
from deepeval.test_case import Turn, ConversationalTestCase, ToolCall
from deepeval.metrics import GoalAccuracyMetric

# ==========================================
# 1. THE METRIC (strict mode)
# ==========================================
# `strict_mode=True` collapses the score to binary:
#   - 1 if the agent reached the goal AND followed a
#     sound plan (both halves at max)
#   - 0 otherwise
# It also overrides `threshold` to 1, so is_successful()
# matches the binary score exactly. Use it when any
# wrong tool / wrong city / wrong plan is unacceptable
# (compliance, medical, tightly-scoped agents).

strict_goal_accuracy = GoalAccuracyMetric(
    threshold=0.5,                      # ignored in strict mode
    model=CustomOpenAI(),
    include_reason=True,
    strict_mode=True,
)

# ==========================================
# 2. RIGHT TOOL, RIGHT CITY — expect score 1
# ==========================================
# The agent calls `get_weather("Paris")` and reports the
# result. Both halves should be at max.

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

strict_goal_accuracy.measure(perfect_case)
print("--- perfect plan + goal ---")
print(f"Score:   {strict_goal_accuracy.score}")          # 1.0
print(f"Reason:  {strict_goal_accuracy.reason}")
print(f"Passed:  {strict_goal_accuracy.is_successful()}")  # True

# ==========================================
# 3. WRONG CITY — expect score 0
# ==========================================
# The agent calls `get_weather("London")` and reports a
# Paris answer based on the wrong data. The plan was
# unsound and the goal wasn't really reached.
# Strict mode drops the score to 0. Compare against the
# non-strict `02_verbose` run, where the same agent
# scores somewhere in the middle.

wrong_city_case = ConversationalTestCase(
    turns=[
        Turn(role="user", content="What's the current weather in Paris?"),
        Turn(
            role="assistant",
            content="It's 9°C and raining in Paris.",
            tools_called=[
                ToolCall(
                    name="get_weather",
                    description="Look up the current weather for a city.",
                    input_parameters={"city": "London"},                            # wrong city
                    output={"temp_c": 9, "condition": "Rain"},
                )
            ],
        ),
    ]
)

strict_goal_accuracy.measure(wrong_city_case)
print("--- wrong city ---")
print(f"Score:   {strict_goal_accuracy.score}")          # 0.0
print(f"Reason:  {strict_goal_accuracy.reason}")
print(f"Passed:  {strict_goal_accuracy.is_successful()}")  # False