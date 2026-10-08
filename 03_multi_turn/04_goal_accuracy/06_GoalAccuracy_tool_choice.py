from basemodel import CustomOpenAI
from deepeval.test_case import Turn, ConversationalTestCase, ToolCall
from deepeval.metrics import GoalAccuracyMetric

# ==========================================
# 1. THE METRIC
# ==========================================
# A single-metric run on three Paris-weather agents that
# differ only in which tool they use. The metric
# averages the goal evaluation score and plan
# evaluation score; the wrong-tool agent fails on BOTH
# halves (wrong tool = unsound plan, hallucinated answer
# = goal not actually reached).

goal_accuracy = GoalAccuracyMetric(
    threshold=0.5,
    model=CustomOpenAI(),
    include_reason=True,
)

# ==========================================
# 2. SHARED TASK
# ==========================================
# Every case below has the same user message:
# "What's the current weather in Paris?"

user_turn = Turn(role="user", content="What's the current weather in Paris?")

# ==========================================
# 3. CASE A — right tool, right city
# ==========================================
# Goal reached, plan sound. Expect score ≈ 1.0.

right_tool_case = ConversationalTestCase(
    turns=[
        user_turn,
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
# 4. CASE B — wrong city (same tool)
# ==========================================
# Plan unsound (asked for London), goal not really
# reached (Paris data was never fetched).

wrong_city_case = ConversationalTestCase(
    turns=[
        user_turn,
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

# ==========================================
# 5. CASE C — no tool at all (pure hallucination)
# ==========================================
# Plan was absent — no tool call. The reply is plausible
# but the agent has no evidence. Goal not reached.

no_tool_case = ConversationalTestCase(
    turns=[
        user_turn,
        Turn(
            role="assistant",
            content="It's 22°C and sunny in Paris, perfect beach day!",
            # No tools_called — the agent just made something up.
        ),
    ]
)

# ==========================================
# 6. RUN THE METRIC ON ALL THREE CASES
# ==========================================
# Watch the score drop as the agent's plan gets worse.

def run(label, case):
    goal_accuracy.measure(case)
    print(f"--- {label} ---")
    print(f"  Score:  {goal_accuracy.score:.3f}")
    print(f"  Passed: {goal_accuracy.is_successful()}")
    print(f"  Reason: {goal_accuracy.reason}")

run("A — right tool, right city",     right_tool_case)
run("B — right tool, wrong city",     wrong_city_case)
run("C — no tool at all",             no_tool_case)