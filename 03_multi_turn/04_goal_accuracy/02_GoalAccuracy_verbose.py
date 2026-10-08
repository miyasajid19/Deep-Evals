from basemodel import CustomOpenAI
from deepeval.test_case import Turn, ConversationalTestCase, ToolCall
from deepeval.metrics import GoalAccuracyMetric

# ==========================================
# 1. THE METRIC (verbose)
# ==========================================
# `verbose_mode=True` prints the intermediate steps:
#   - the goal extracted from each user message
#   - the steps taken from each assistant turn
#   - the goal evaluation score and plan evaluation score
#     for each (goal, steps) pair
# This is the fastest way to debug "the agent answered
# but didn't actually do the task". Verbose mode shows
# which part (goal vs plan) cost you points.

goal_accuracy = GoalAccuracyMetric(
    threshold=0.5,
    model=CustomOpenAI(),
    include_reason=True,
    verbose_mode=True,
)

# ==========================================
# 2. THE TEST CASE — agent calls the WRONG tool
# ==========================================
# User wants the weather in Paris. The agent calls
# `get_weather` for London instead, then makes up a Paris
# answer. The plan was unsound (wrong city) and the goal
# wasn't actually reached (the agent never had Paris
# data). Expect a low score on both halves.

convo_test_case = ConversationalTestCase(
    turns=[
        Turn(role="user", content="What's the current weather in Paris?"),
        Turn(
            role="assistant",
            content="It's 9°C and raining in Paris.",
            tools_called=[
                ToolCall(
                    name="get_weather",
                    description="Look up the current weather for a city.",
                    input_parameters={"city": "London"},                              # wrong city
                    output={"temp_c": 9, "condition": "Rain"},
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

goal_accuracy.measure(convo_test_case)
print("---")
print(f"Score:   {goal_accuracy.score}")
print(f"Reason:  {goal_accuracy.reason}")
print(f"Passed:  {goal_accuracy.is_successful()}")