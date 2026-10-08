from basemodel import CustomOpenAI
from deepeval.test_case import LLMTestCase, MCPServer, MCPToolCall
from deepeval.metrics import MCPUseMetric

# ==========================================
# 1. THE METRIC
# ==========================================
# A single-metric run on three weather agents that differ
# in which tool they called and which values they passed.
# The metric's alignment score reflects both halves —
# selection + arguments.

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
# 2. SHARED INPUT
# ==========================================
input_q = "What's the current weather in Paris?"

# ==========================================
# 3. CASE A — right tool, right args
# ==========================================
right_case = LLMTestCase(
    input=input_q,
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
# 4. CASE B — right tool, wrong args
# ==========================================
wrong_args_case = LLMTestCase(
    input=input_q,
    actual_output="It's 9°C and raining — bring a coat.",
    mcp_servers=mcp_servers,
    mcp_tools_called=[
        MCPToolCall(
            name="get_current_weather",
            args={"city": "London"},                              # wrong arg
            result={"temp_c": 9, "condition": "Rain"},
        )
    ],
)

# ==========================================
# 5. CASE C — wrong tool
# ==========================================
wrong_tool_case = LLMTestCase(
    input=input_q,
    actual_output="It's around 18°C and partly cloudy today in Paris.",
    mcp_servers=mcp_servers,
    mcp_tools_called=[
        MCPToolCall(
            name="get_weather_forecast",                          # wrong tool
            args={"city": "Paris"},
            result={"today": {"temp_c": 18, "condition": "Partly cloudy"}},
        )
    ],
)

# ==========================================
# 6. RUN THE METRIC ON ALL THREE CASES
# ==========================================
def run(label, case):
    mcp_use.measure(case)
    print(f"--- {label} ---")
    print(f"  Score:  {mcp_use.score:.3f}")
    print(f"  Passed: {mcp_use.is_successful()}")
    print(f"  Reason: {mcp_use.reason}")

run("A — right tool, right args",     right_case)
run("B — right tool, wrong args",     wrong_args_case)
run("C — wrong tool",                 wrong_tool_case)