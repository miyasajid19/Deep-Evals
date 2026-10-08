from deepeval.test_case import Turn, ConversationalTestCase, ToolCall
from deepeval.metrics import GoalAccuracyMetric

# ==========================================
# 1. THE METRIC (system_one / Jev-as-a-judge)
# ==========================================
# No LLM is called at all. Jev receives the whole
# conversation (with the tools called in each turn) and
# answers three questions:
#
#   Q1 (Noul, weight 2): every user goal is fully and
#                        correctly achieved in the
#                        assistant turns that follow it
#   Q2 (Score, weight 1): across turns, how fully the
#                            assistant's visible replies
#                            achieve the user goals
#                            (not achieved → fully achieved)
#   Q3 (Score, weight 1): how clear/complete the plan is
#                          (including tool calls) and how
#                          closely the assistant followed it
#                          (no plan → complete plan, fully followed)
#
# Final score = weighted mean of the three answers.
#
# system_one mode is the right pick when you want the
# cheapest possible judge and don't need to compare
# against an LLM-mode score from the same metric.

system_one_goal_accuracy = GoalAccuracyMetric(
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

system_one_goal_accuracy.measure(convo_test_case)
print("---")
print(f"Score:      {system_one_goal_accuracy.score}")
print(f"Confidence: {system_one_goal_accuracy.confidence}")
print(f"Reason:     {system_one_goal_accuracy.reason}")
print(f"Passed:     {system_one_goal_accuracy.is_successful()}")