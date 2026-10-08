from deepeval.test_case import LLMTestCase, MCPServer, MCPToolCall
from deepeval.metrics import MCPUseMetric

# ==========================================
# 1. THE METRIC (system_one / Jev-as-a-judge)
# ==========================================
# No LLM is called at all. Jev receives the input, the
# actual_output, the MCP calls made, and the available
# mcp_servers, and answers three questions:
#
#   Q1 (Noul, weight 2): the MCP tools/resources/prompts
#                        called are the right ones, with
#                        no clearly better primitive
#                        missed and no unnecessary call
#   Q2 (Noul, weight 1): every MCP call passes arguments
#                        matching the input schema and
#                        carrying the values the input
#                        asks for
#   Q3 (Score, weight 1): overall how well the agent uses
#                          the MCP servers (wrong →
#                          correct)
#
# Final score = weighted mean of the three answers.

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

system_one_mcp_use = MCPUseMetric(
    threshold=0.5,
    include_reason=True,
    eval_mode="system_one",
    # No `model=` needed — Jev runs the metric alone.
)

# ==========================================
# 2. THE TEST CASE
# ==========================================
test_case = LLMTestCase(
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

# ==========================================
# 3. RUN STANDALONE
# ==========================================
system_one_mcp_use.measure(test_case)
print("---")
print(f"Score:      {system_one_mcp_use.score}")
print(f"Confidence: {system_one_mcp_use.confidence}")
print(f"Reason:     {system_one_mcp_use.reason}")
print(f"Passed:     {system_one_mcp_use.is_successful()}")