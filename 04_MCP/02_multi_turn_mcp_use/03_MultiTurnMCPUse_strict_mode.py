from basemodel import CustomOpenAI
from deepeval.test_case import Turn, ConversationalTestCase, MCPServer, MCPToolCall
from deepeval.metrics import MultiTurnMCPUseMetric

# ==========================================
# 1. THE METRIC (strict mode)
# ==========================================
# `strict_mode=True` collapses the score to binary:
#   - 1 if every MCP interaction is perfectly aligned
#     (right primitive + right arguments)
#   - 0 if any single interaction is misaligned
# It also overrides `threshold` to 1, so is_successful()
# matches the binary score exactly. Use it when any
# wrong MCP call breaks the experience (compliance,
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

strict_multi_turn_mcp_use = MultiTurnMCPUseMetric(
    threshold=0.5,                      # ignored in strict mode
    model=CustomOpenAI(),
    include_reason=True,
    strict_mode=True,
)

# ==========================================
# 2. EVERY INTERACTION PERFECT — expect score 1
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

strict_multi_turn_mcp_use.measure(perfect_case)
print("--- every interaction perfect ---")
print(f"Score:   {strict_multi_turn_mcp_use.score}")          # 1.0
print(f"Reason:  {strict_multi_turn_mcp_use.reason}")
print(f"Passed:  {strict_multi_turn_mcp_use.is_successful()}")  # True

# ==========================================
# 3. ONE WRONG ARG — expect score 0
# ==========================================
# Second interaction calls for New York when the user
# asked about London. Strict mode drops the score to 0.

one_wrong_case = ConversationalTestCase(
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
            content="It's 14°C and clear in New York.",
            mcp_tools_called=[
                MCPToolCall(name="get_current_weather", args={"city": "New York"}, result={"temp_c": 14})  # wrong
            ],
        ),
    ],
    mcp_servers=mcp_servers,
)

strict_multi_turn_mcp_use.measure(one_wrong_case)
print("--- one wrong arg ---")
print(f"Score:   {strict_multi_turn_mcp_use.score}")          # 0.0
print(f"Reason:  {strict_multi_turn_mcp_use.reason}")
print(f"Passed:  {strict_multi_turn_mcp_use.is_successful()}")  # False