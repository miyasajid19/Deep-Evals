from basemodel import CustomOpenAI
from deepeval.test_case import LLMTestCase, MCPServer, MCPToolCall
from deepeval.metrics import MCPUseMetric

# ==========================================
# 1. THE METRIC (verbose)
# ==========================================
# `verbose_mode=True` prints the intermediate alignment
# score and the judge's reasoning for primitive selection
# and argument correctness. This is the fastest way to
# debug "the agent picked the wrong tool" — verbose mode
# surfaces which primitive should have been called instead.

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
    verbose_mode=True,
)

# ==========================================
# 2. THE TEST CASE — agent picks the WRONG tool
# ==========================================
# User asks for current weather. The agent calls
# `get_weather_forecast` instead of `get_current_weather`,
# then reports "current" weather from the forecast data.
# The primitive selection is suboptimal even though the
# output looks plausible.

test_case = LLMTestCase(
    input="What's the current weather in Paris?",
    actual_output="It's around 18°C and partly cloudy today in Paris.",
    mcp_servers=mcp_servers,
    mcp_tools_called=[
        MCPToolCall(
            name="get_weather_forecast",                                  # wrong tool
            args={"city": "Paris"},
            result={"today": {"temp_c": 18, "condition": "Partly cloudy"}},
        )
    ],
)

# ==========================================
# 3. RUN STANDALONE
# ==========================================
mcp_use.measure(test_case)
print("---")
print(f"Score:   {mcp_use.score}")
print(f"Reason:  {mcp_use.reason}")
print(f"Passed:  {mcp_use.is_successful()}")