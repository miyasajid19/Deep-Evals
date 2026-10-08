from basemodel import CustomOpenAI
from deepeval.test_case import Turn, ConversationalTestCase, MCPServer, MCPToolCall
from deepeval.metrics import MCPTaskCompletionMetric

# ==========================================
# 1. THE METRIC
# ==========================================
# A single-metric run on three conversations that share
# the same MCP server menu but differ in whether the
# agent actually completed the task the user asked for.
#
# The metric scores the OUTCOME — primitive selection is
# MultiTurnMCPUseMetric's job. An agent can use the tool
# perfectly and still fail if it doesn't relay the answer
# back to the user.

mcp_servers = [
    MCPServer(
        server_name="weather",
        transport="stdio",
        available_tools=[
            {
                "name": "get_current_weather",
                "description": "Look up the current weather for a city.",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        ],
    )
]

mcp_task_completion = MCPTaskCompletionMetric(
    threshold=0.5,
    model=CustomOpenAI(),
    include_reason=True,
)

# ==========================================
# 2. SHARED INPUT
# ==========================================
user_turn = Turn(role="user", content="What's the current weather in Paris?")

# ==========================================
# 3. CASE A — task fully completed
# ==========================================
# Tool called, result relayed back. Expect score 1.0.

completed_case = ConversationalTestCase(
    turns=[
        user_turn,
        Turn(
            role="assistant",
            content="It's 18°C and partly cloudy in Paris right now.",
            mcp_tools_called=[
                MCPToolCall(
                    name="get_current_weather",
                    args={"city": "Paris"},
                    result={"temp_c": 18, "condition": "Partly cloudy"},
                )
            ],
        ),
    ],
    mcp_servers=mcp_servers,
)

# ==========================================
# 4. CASE B — tool ran but user never got the answer
# ==========================================
# Tool succeeded; assistant says "let me look that up" and
# stops. From the user's perspective the task is unfinished.

unfinished_case = ConversationalTestCase(
    turns=[
        user_turn,
        Turn(
            role="assistant",
            content="Let me look that up.",
            mcp_tools_called=[
                MCPToolCall(
                    name="get_current_weather",
                    args={"city": "Paris"},
                    result={"temp_c": 18, "condition": "Partly cloudy"},
                )
            ],
        ),
    ],
    mcp_servers=mcp_servers,
)

# ==========================================
# 5. CASE C — fabricated answer (no MCP call)
# ==========================================
# Agent never called the tool; just made up a temperature.
# Task is not grounded.

fabricated_case = ConversationalTestCase(
    turns=[
        user_turn,
        Turn(
            role="assistant",
            content="It's 22°C and sunny in Paris today.",
            # No mcp_tools_called — agent invented the answer.
        ),
    ],
    mcp_servers=mcp_servers,
)

# ==========================================
# 6. RUN THE METRIC ON ALL THREE CASES
# ==========================================
def run(label, case):
    mcp_task_completion.measure(case)
    print(f"--- {label} ---")
    print(f"  Score:  {mcp_task_completion.score:.3f}")
    print(f"  Passed: {mcp_task_completion.is_successful()}")
    print(f"  Reason: {mcp_task_completion.reason}")

run("A — task fully completed",        completed_case)
run("B — tool ran, no answer relayed",  unfinished_case)
run("C — fabricated answer",           fabricated_case)