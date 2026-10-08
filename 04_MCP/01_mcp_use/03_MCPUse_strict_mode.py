from basemodel import CustomOpenAI
from deepeval.test_case import LLMTestCase, MCPServer, MCPToolCall
from deepeval.metrics import MCPUseMetric

# ==========================================
# 1. THE METRIC (strict mode)
# ==========================================
# `strict_mode=True` collapses the score to binary:
#   - 1 if primitive selection AND arguments are perfect
#   - 0 otherwise
# It also overrides `threshold` to 1, so is_successful()
# matches the binary score exactly. Use it when any
# wrong tool or wrong argument is unacceptable
# (compliance, medical, tightly-scoped MCP agents).

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

strict_mcp_use = MCPUseMetric(
    threshold=0.5,                      # ignored in strict mode
    model=CustomOpenAI(),
    include_reason=True,
    strict_mode=True,
)

# ==========================================
# 2. RIGHT TOOL + RIGHT ARG — expect score 1
# ==========================================
perfect_case = LLMTestCase(
    input="What's the current weather in Paris?",
    actual_output="It's 18°C and partly cloudy in Paris right now.",
    mcp_servers=mcp_servers,
    mcp_tools_called=[
        MCPToolCall(
            name="get_current_weather",
            args={"city": "Paris"},
            result={"temp_c": 18, "condition": "Partly cloudy"},
        )
    ],
)

strict_mcp_use.measure(perfect_case)
print("--- right tool, right args ---")
print(f"Score:   {strict_mcp_use.score}")          # 1.0
print(f"Reason:  {strict_mcp_use.reason}")
print(f"Passed:  {strict_mcp_use.is_successful()}")  # True

# ==========================================
# 3. RIGHT TOOL + WRONG ARG — expect score 0
# ==========================================
# Agent picks the right tool but passes the wrong argument
# (London instead of Paris). Strict mode drops the score
# to 0.

wrong_arg_case = LLMTestCase(
    input="What's the current weather in Paris?",
    actual_output="It's 9°C and raining — bring a coat.",
    mcp_servers=mcp_servers,
    mcp_tools_called=[
        MCPToolCall(
            name="get_current_weather",
            args={"city": "London"},                              # wrong arg
            result={"temp_c": 9, "condition": "Rain"},
        )
    ],
)

strict_mcp_use.measure(wrong_arg_case)
print("--- right tool, wrong arg ---")
print(f"Score:   {strict_mcp_use.score}")          # 0.0
print(f"Reason:  {strict_mcp_use.reason}")
print(f"Passed:  {strict_mcp_use.is_successful()}")  # False