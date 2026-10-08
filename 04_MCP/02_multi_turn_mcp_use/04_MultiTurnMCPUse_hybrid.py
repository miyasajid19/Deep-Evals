from basemodel import CustomOpenAI
from deepeval.test_case import Turn, ConversationalTestCase, MCPServer, MCPToolCall
from deepeval.metrics import MultiTurnMCPUseMetric

# ==========================================
# 1. THE METRIC (hybrid mode)
# ==========================================
# Hybrid for Multi-Turn MCP-Use: the LLM STILL splits the
# conversation into tasks / interactions. For each task,
# Jev (a System One model) rates primitive usage and
# argument correctness on a scale mapped onto 0..1.
#
# The equation (mean per interaction) and the LLM-written
# final reason are unchanged. If a Jev call fails, the
# metric silently defers that decision to the LLM.

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

hybrid_multi_turn_mcp_use = MultiTurnMCPUseMetric(
    threshold=0.5,
    model=CustomOpenAI(),       # LLM still splits the conversation into tasks
    include_reason=True,
    eval_mode="hybrid",         # Jev rates each task's usage + args
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
hybrid_multi_turn_mcp_use.measure(convo_test_case)
print("---")
print(f"Score:      {hybrid_multi_turn_mcp_use.score}")
print(f"Confidence: {hybrid_multi_turn_mcp_use.confidence}")
print(f"Reason:     {hybrid_multi_turn_mcp_use.reason}")
print(f"Passed:     {hybrid_multi_turn_mcp_use.is_successful()}")