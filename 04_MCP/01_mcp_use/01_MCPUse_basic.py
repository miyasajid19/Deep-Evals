from basemodel import CustomOpenAI
from deepeval.test_case import LLMTestCase, MCPServer, MCPToolCall
from deepeval.metrics import MCPUseMetric
from deepeval import evaluate

# ==========================================
# 1. THE METRIC
# ==========================================
# MCPUseMetric evaluates how effectively an MCP-based
# agent makes use of the MCP servers it has access to.
# It judges both PRIMITIVE SELECTION (which tool/resource/
# prompt did the agent call) and ARGUMENT CORRECTNESS
# (what values did the agent pass).
#
# Algorithm (default `llm` eval mode):
#   score = AlignmentScore(Primitives Used, Primitives Available)
#
# `mcp_servers` is required on the test case — it's the
# available primitives list the metric judges selection
# against. The metric is `referenceless`: no
# `expected_output` needed.

# ==========================================
# 2. THE MCP SERVER MENU
# ==========================================
# A weather MCP server that exposes two tools. The agent
# has access to both — `mcp_servers` carries that list.

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
            {
                "name": "get_weather_forecast",
                "description": "Look up the multi-day forecast for a city.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "days": {"type": "integer", "default": 5},
                    },
                    "required": ["city"],
                },
            },
        ],
    )
]

mcp_use = MCPUseMetric(
    threshold=0.5,
    model=CustomOpenAI(),
    include_reason=True,
)

# ==========================================
# 3. THE TEST CASE
# ==========================================
# User asks for Paris's current weather. The agent calls
# `get_current_weather("Paris")` — right tool, right
# argument. Expect score 1.0.

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
# 4. RUN THE EVAL
# ==========================================
# `evaluate()` runs the metric, prints results, and writes
# a report (Confident AI if configured, else local cache).

evaluate(test_cases=[test_case], metrics=[mcp_use])