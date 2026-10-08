from basemodel import CustomOpenAI
from deepeval.test_case import Turn, ConversationalTestCase, MCPServer, MCPToolCall
from deepeval.metrics import MCPTaskCompletionMetric

# ==========================================
# 1. THE METRIC (verbose)
# ==========================================
# `verbose_mode=True` prints per-interaction reasoning so
# you can see which turn the judge considered incomplete.
# This is the fastest way to debug "the agent ran the
# tool but the user never got the answer".

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

mcp_task_completion = MCPTaskCompletionMetric(
    threshold=0.5,
    model=CustomOpenAI(),
    include_reason=True,
    verbose_mode=True,
)

# ==========================================
# 2. THE TEST CASE — agent ran the tool but never reported
# ==========================================
# The tool call succeeded but the assistant never told
# the user the result. From the user's perspective the
# task is incomplete. Verbose mode surfaces which turn
# got flagged.

convo_test_case = ConversationalTestCase(
    turns=[
        Turn(role="user",      content="What's the current weather in Paris?"),
        Turn(
            role="assistant",
            content="Let me look that up.",
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
# 3. RUN STANDALONE
# ==========================================
mcp_task_completion.measure(convo_test_case)
print("---")
print(f"Score:   {mcp_task_completion.score}")
print(f"Reason:  {mcp_task_completion.reason}")
print(f"Passed:  {mcp_task_completion.is_successful()}")