from basemodel import CustomOpenAI
from deepeval.test_case import Turn, ConversationalTestCase, MCPServer, MCPToolCall
from deepeval.metrics import MCPTaskCompletionMetric
from deepeval import evaluate

# ==========================================
# 1. THE METRIC
# ==========================================
# MCPTaskCompletionMetric judges the OUTCOME — did the
# agent accomplish the task, regardless of which
# primitives it used. It's the multi-turn outcome sibling
# of MultiTurnMCPUseMetric (which judges process).
#
# Algorithm (default `llm` eval mode):
#   1) split the conversation into unit interactions
#   2) for each interaction, judge whether the task the
#      user asked is completed as the user sees it
#   3) score = (# tasks satisfied) / (# interactions)
#
# `mcp_servers` is required on the test case. The
# primitives the agent called live INSIDE each
# `Turn.mcp_tools_called`. The metric is `referenceless`:
# no `expected_output` needed.

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

mcp_task_completion = MCPTaskCompletionMetric(
    threshold=0.5,
    model=CustomOpenAI(),
    include_reason=True,
)

# ==========================================
# 3. THE TEST CASE
# ==========================================
# User asks for current Paris weather. The agent calls
# the tool and reports the result. Task completed.

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
evaluate(test_cases=[convo_test_case], metrics=[mcp_task_completion])