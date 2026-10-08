from deepeval.test_case import Turn, ConversationalTestCase, MCPServer, MCPToolCall
from deepeval.metrics import MultiTurnMCPUseMetric

# ==========================================
# 1. THE METRIC (system_one / Jev-as-a-judge)
# ==========================================
# No LLM is called at all. Jev receives the whole
# conversation (each turn's MCP calls) and the available
# mcp_servers, and answers three questions:
#
#   Q1 (Noul, weight 2): in every assistant turn, the MCP
#                        primitives called are the right
#                        ones, with no clearly better
#                        primitive missed
#   Q2 (Noul, weight 1): every MCP call passes arguments
#                        matching the schema and carrying
#                        the values the user asked for
#   Q3 (Score, weight 1): overall how well the assistant
#                          uses the MCP servers across
#                          turns (wrong → correct)
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

system_one_multi_turn_mcp_use = MultiTurnMCPUseMetric(
    threshold=0.5,
    include_reason=True,
    eval_mode="system_one",
    # No `model=` needed — Jev runs the metric alone.
)

# ==========================================
# 2. THE TEST CASE
# ==========================================
convo_test_case = ConversationalTestCase(
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

# ==========================================
# 3. RUN STANDALONE
# ==========================================
system_one_multi_turn_mcp_use.measure(convo_test_case)
print("---")
print(f"Score:      {system_one_multi_turn_mcp_use.score}")
print(f"Confidence: {system_one_multi_turn_mcp_use.confidence}")
print(f"Reason:     {system_one_multi_turn_mcp_use.reason}")
print(f"Passed:     {system_one_multi_turn_mcp_use.is_successful()}")