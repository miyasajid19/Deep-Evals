from basemodel import CustomOpenAI
from deepeval.test_case import Turn, ConversationalTestCase, MCPServer, MCPToolCall
from deepeval.metrics import MultiTurnMCPUseMetric
from deepeval import evaluate

# ==========================================
# 1. THE METRIC
# ==========================================
# MultiTurnMCPUseMetric is the multi-turn sibling of
# MCPUseMetric — it judges how well an MCP agent uses
# the primitives available across a whole conversation.
#
# Algorithm (default `llm` eval mode):
#   score = mean(AlignmentScore per MCP interaction)
#
# The denominator is total MCP interactions, so an extra
# noisy or redundant call is held to the same standard
# and drags the average down.
#
# `mcp_servers` is required on the test case. The
# primitives the agent called live INSIDE each
# `Turn.mcp_tools_called`, not on the test case itself.

# ==========================================
# 2. THE MCP SERVER MENU
# ==========================================
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
)

# ==========================================
# 3. THE TEST CASE
# ==========================================
# User asks for current weather in Paris. Agent calls
# `get_current_weather("Paris")` — right tool, right
# argument, single interaction. Expect score 1.0.

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
    ],
    mcp_servers=mcp_servers,
)

# ==========================================
# 4. RUN THE EVAL
# ==========================================
evaluate(test_cases=[convo_test_case], metrics=[multi_turn_mcp_use])