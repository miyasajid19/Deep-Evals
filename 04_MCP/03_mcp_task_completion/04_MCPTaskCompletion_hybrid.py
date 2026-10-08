from basemodel import CustomOpenAI
from deepeval.test_case import Turn, ConversationalTestCase, MCPServer, MCPToolCall
from deepeval.metrics import MCPTaskCompletionMetric

# ==========================================
# 1. THE METRIC (hybrid mode)
# ==========================================
# Hybrid for MCP Task Completion: the LLM STILL splits
# the conversation into tasks / interactions. For each
# task, Jev (a System One model) rates completion on a
# FOUR-level scale mapped onto 0..1, with a short line
# stating Jev's score and confidence.
#
# The equation and the LLM-written final reason are
# unchanged. If a Jev call fails, the metric silently
# defers that decision to the LLM.

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

hybrid_mcp_task_completion = MCPTaskCompletionMetric(
    threshold=0.5,
    model=CustomOpenAI(),       # LLM still splits the conversation
    include_reason=True,
    eval_mode="hybrid",         # Jev rates each task on a 4-level scale
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
hybrid_mcp_task_completion.measure(convo_test_case)
print("---")
print(f"Score:      {hybrid_mcp_task_completion.score}")
print(f"Confidence: {hybrid_mcp_task_completion.confidence}")
print(f"Reason:     {hybrid_mcp_task_completion.reason}")
print(f"Passed:     {hybrid_mcp_task_completion.is_successful()}")