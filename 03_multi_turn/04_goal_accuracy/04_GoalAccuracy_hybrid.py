from basemodel import CustomOpenAI
from deepeval.test_case import Turn, ConversationalTestCase, ToolCall
from deepeval.metrics import GoalAccuracyMetric

# ==========================================
# 1. THE METRIC (hybrid mode)
# ==========================================
# Hybrid for Goal Accuracy works differently from the
# other hybrid modes: there's no LLM extraction step
# (the LLM extracts goals + plans in one go and emits
# them inline). What hybrid *does* change is the
# per-interaction scoring:
#
#   For each (goal, plan) interaction, Jev (a System One
#   model) rates BOTH halves on a five-level scale that
#   the metric maps onto 0..1. Each interaction's reason
#   states Jev's score and confidence.
#
# The LLM still writes the FINAL reason (rolling up the
# per-interaction results). If a Jev call fails, the LLM
# makes that decision instead.

hybrid_goal_accuracy = GoalAccuracyMetric(
    threshold=0.5,
    model=CustomOpenAI(),       # LLM still writes the final reason
    include_reason=True,
    eval_mode="hybrid",         # Jev scores each interaction
)

# ==========================================
# 2. THE TEST CASE
# ==========================================
# Same Paris-weather agent as the basic run — right tool,
# right city, goal reached.

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
# model was on average across its per-interaction calls.

hybrid_goal_accuracy.measure(convo_test_case)
print("---")
print(f"Score:      {hybrid_goal_accuracy.score}")
print(f"Confidence: {hybrid_goal_accuracy.confidence}")
print(f"Reason:     {hybrid_goal_accuracy.reason}")
print(f"Passed:     {hybrid_goal_accuracy.is_successful()}")