from deepeval.test_case import Turn, ConversationalTestCase, MCPServer, MCPToolCall
from deepeval.metrics import MCPTaskCompletionMetric

# ==========================================
# 1. THE METRIC (system_one / Jev-as-a-judge)
# ==========================================
# No LLM is called at all. Jev receives the whole
# conversation (each turn's MCP calls) and answers three
# questions:
#
#   Q1 (Noul, weight 2): every task a user message asks
#                        for is completed by the assistant
#                        (tool calls invisible unless
#                        relayed)
#   Q2 (Noul, weight 1): everything the assistant tells
#                        the user is backed by MCP results
#                        (no fabricated claims)
#   Q3 (Score, weight 1): across turns, how completely
#                          the assistant completes tasks
#                          (not → fully completed)
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

system_one_mcp_task_completion = MCPTaskCompletionMetric(
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
    ],
    mcp_servers=mcp_servers,
)

# ==========================================
# 3. RUN STANDALONE
# ==========================================
system_one_mcp_task_completion.measure(convo_test_case)
print("---")
print(f"Score:      {system_one_mcp_task_completion.score}")
print(f"Confidence: {system_one_mcp_task_completion.confidence}")
print(f"Reason:     {system_one_mcp_task_completion.reason}")
print(f"Passed:     {system_one_mcp_task_completion.is_successful()}")