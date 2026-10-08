from basemodel import CustomOpenAI
from deepeval.test_case import Turn, ConversationalTestCase, MCPServer, MCPToolCall
from deepeval.metrics import MCPTaskCompletionMetric

# ==========================================
# 1. THE METRIC (strict mode)
# ==========================================
# `strict_mode=True` collapses the score to binary:
#   - 1 if every interaction's task is completed
#   - 0 if any single interaction is incomplete
# It also overrides `threshold` to 1, so is_successful()
# matches the binary score exactly. Use it when any
# unfinished task breaks the experience (compliance,
# medical, tightly-scoped MCP agents).

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

strict_mcp_task_completion = MCPTaskCompletionMetric(
    threshold=0.5,                      # ignored in strict mode
    model=CustomOpenAI(),
    include_reason=True,
    strict_mode=True,
)

# ==========================================
# 2. EVERY TASK COMPLETED — expect score 1
# ==========================================
perfect_case = ConversationalTestCase(
    turns=[
        Turn(role="user",      content="What's the current weather in Paris?"),
        Turn(
            role="assistant",
            content="It's 18°C and partly cloudy in Paris right now.",
            mcp_tools_called=[
                MCPToolCall(name="get_current_weather", args={"city": "Paris"}, result={"temp_c": 18})
            ],
        ),
        Turn(role="user",      content="And London?"),
        Turn(
            role="assistant",
            content="It's 9°C and raining in London.",
            mcp_tools_called=[
                MCPToolCall(name="get_current_weather", args={"city": "London"}, result={"temp_c": 9})
            ],
        ),
    ],
    mcp_servers=mcp_servers,
)

strict_mcp_task_completion.measure(perfect_case)
print("--- every task completed ---")
print(f"Score:   {strict_mcp_task_completion.score}")          # 1.0
print(f"Reason:  {strict_mcp_task_completion.reason}")
print(f"Passed:  {strict_mcp_task_completion.is_successful()}")  # True

# ==========================================
# 3. ONE TASK UNFINISHED — expect score 0
# ==========================================
# First turn answers the user; second turn never reports
# the result back. Strict mode drops the score to 0.

one_unfinished_case = ConversationalTestCase(
    turns=[
        Turn(role="user",      content="What's the current weather in Paris?"),
        Turn(
            role="assistant",
            content="It's 18°C and partly cloudy in Paris right now.",
            mcp_tools_called=[
                MCPToolCall(name="get_current_weather", args={"city": "Paris"}, result={"temp_c": 18})
            ],
        ),
        Turn(role="user",      content="And London?"),
        Turn(
            role="assistant",
            content="Let me look that up.",
            mcp_tools_called=[
                MCPToolCall(name="get_current_weather", args={"city": "London"}, result={"temp_c": 9})
            ],
        ),
    ],
    mcp_servers=mcp_servers,
)

strict_mcp_task_completion.measure(one_unfinished_case)
print("--- one unfinished task ---")
print(f"Score:   {strict_mcp_task_completion.score}")          # 0.0
print(f"Reason:  {strict_mcp_task_completion.reason}")
print(f"Passed:  {strict_mcp_task_completion.is_successful()}")  # False