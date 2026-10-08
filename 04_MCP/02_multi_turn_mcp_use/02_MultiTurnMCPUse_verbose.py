from basemodel import CustomOpenAI
from deepeval.test_case import Turn, ConversationalTestCase, MCPServer, MCPToolCall
from deepeval.metrics import MultiTurnMCPUseMetric

# ==========================================
# 1. THE METRIC (verbose)
# ==========================================
# `verbose_mode=True` prints per-interaction reasoning
# so you can see which turn dragged the score down. This
# is the fastest way to debug "one turn used the wrong
# tool" — verbose mode surfaces the offending interaction.

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

multi_turn_mcp_use = MultiTurnMCPUseMetric(
    threshold=0.5,
    model=CustomOpenAI(),
    include_reason=True,
    verbose_mode=True,
)

# ==========================================
# 2. THE TEST CASE — second turn picks wrong city
# ==========================================
# Two MCP interactions. The first is correct; the second
# calls `get_current_weather` for London when the user
# asked about Paris. Verbose mode will surface which
# interaction failed.

convo_test_case = ConversationalTestCase(
    turns=[
        Turn(role="user",      content="What's the current weather in Paris?"),
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
        Turn(role="user",      content="And in London?"),
        Turn(
            role="assistant",
            content="Let me check that.",
            mcp_tools_called=[
                MCPToolCall(
                    name="get_current_weather",
                    args={"city": "New York"},                  # wrong city
                    result={"temp_c": 14, "condition": "Clear"},
                )
            ],
        ),
    ],
    mcp_servers=mcp_servers,
)

# ==========================================
# 3. RUN STANDALONE
# ==========================================
multi_turn_mcp_use.measure(convo_test_case)
print("---")
print(f"Score:   {multi_turn_mcp_use.score}")
print(f"Reason:  {multi_turn_mcp_use.reason}")
print(f"Passed:  {multi_turn_mcp_use.is_successful()}")