from basemodel import CustomOpenAI
from deepeval.test_case import Turn, ConversationalTestCase, ToolCall
from deepeval.metrics import ToolUseMetric

# ==========================================
# 1. THE METRIC
# ==========================================
# A single-metric run on three Paris-weather agents that
# differ only in tool choice / arguments. The metric's
# final score is the MIN of tool selection and argument
# correctness, so any failure on either half drags the
# whole score down.

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
)

# ==========================================
# 2. SHARED TASK
# ==========================================
user_turn = Turn(role="user", content="What's the current weather in Paris?")

# ==========================================
# 3. CASE A — right tool, right args
# ==========================================
# Both halves at max. min(1.0, 1.0) = 1.0.

right_case = ConversationalTestCase(
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
# 4. CASE B — right tool, wrong args
# ==========================================
# Selection is perfect; arguments are bad. min(1.0, ~0.0) ≈ 0.

wrong_args_case = ConversationalTestCase(
    turns=[
        user_turn,
        Turn(
            role="assistant",
            content="It's 9°C and raining — bring a coat.",
            tools_called=[
                ToolCall(
                    name="get_weather",
                    description="Look up the current weather for a city.",
                    input_parameters={"city": "London"},                       # wrong arg
                    output={"temp_c": 9, "condition": "Rain"},
                )
            ],
        ),
    ]
)

# ==========================================
# 5. CASE C — wrong tool
# ==========================================
# The agent picks `get_news` instead of `get_weather`.
# min(~0.0, ~1.0) ≈ 0.

wrong_tool_case = ConversationalTestCase(
    turns=[
        user_turn,
        Turn(
            role="assistant",
            content="Top headline: Paris hosts a major climate summit this week.",
            tools_called=[
                ToolCall(
                    name="get_news",                                          # wrong tool
                    description="Look up recent news headlines for a topic.",
                    input_parameters={"topic": "Paris"},
                    output={"headline": "Paris hosts a major climate summit this week."},
                )
            ],
        ),
    ]
)

# ==========================================
# 6. RUN THE METRIC ON ALL THREE CASES
# ==========================================
# Either failure drops the whole score — both halves
# matter.

def run(label, case):
    tool_use.measure(case)
    print(f"--- {label} ---")
    print(f"  Score:  {tool_use.score:.3f}")
    print(f"  Passed: {tool_use.is_successful()}")
    print(f"  Reason: {tool_use.reason}")

run("A — right tool, right args",     right_case)
run("B — right tool, wrong args",     wrong_args_case)
run("C — wrong tool",                 wrong_tool_case)