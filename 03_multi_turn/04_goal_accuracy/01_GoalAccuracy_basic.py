from basemodel import CustomOpenAI
from deepeval.test_case import Turn, ConversationalTestCase, ToolCall
from deepeval.metrics import GoalAccuracyMetric
from deepeval import evaluate

# ==========================================
# 1.1. THE METRIC
# ==========================================
# GoalAccuracyMetric evaluates an LLM agent on PLANNING
# and EXECUTION. It's the right metric when the user
# gives the agent a task that requires picking the right
# tool and following through.
#
# Algorithm (default `llm` eval mode):
#   1) extract the underlying goal from each user message
#   2) extract the steps it actually took from each
#      assistant turn (including `tools_called`)
#   3) for each (goal, steps) pair, compute:
#        - goal evaluation score   — was the goal reached?
#        - plan evaluation score   — was the plan sound and
#                                    was it followed?
#   4) score = mean(goal score, plan score)
#
# Tool calls go on `Turn.tools_called`, and the metric
# factors them into the plan reconstruction. The metric
# is `referenceless`: no `expected_output` needed.

goal_accuracy = GoalAccuracyMetric(
    threshold=0.5,
    model=CustomOpenAI(),
    include_reason=True,
)

# ==========================================
# 2.1. THE TEST CASE
# ==========================================
# A weather agent. The user asks for Paris weather; the
# agent calls the right tool and reports the result.
# Goal + plan both succeed.

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
# 3.1. RUN THE EVAL
# ==========================================
# `evaluate()` runs the metric, prints results, and writes
# a report (Confident AI if configured, else local cache).

evaluate(test_cases=[convo_test_case], metrics=[goal_accuracy])